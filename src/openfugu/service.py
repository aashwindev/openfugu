"""OpenFugu orchestration service wiring router + conductor."""

from __future__ import annotations

import time
import uuid
from typing import Any

from openfugu.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    OpenFuguMetadata,
    RoutingDecision,
    UsageInfo,
)
from openfugu.conductor.planner import ConductorPlanner
from openfugu.config import AppConfig
from openfugu.router.inference import FuguRouter
from openfugu.workers.pool import WorkerPool

OPENFUGU_MODELS = {"openfugu", "openfugu-ultra", "auto"}


class OpenFuguService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pool = WorkerPool(config.workers)
        self.router = FuguRouter(
            self.pool,
            backbone_name=config.router.backbone,
            checkpoint=config.router.checkpoint,
            temperature=config.router.temperature,
            use_heuristic_fallback=config.router.use_heuristic_fallback,
        )
        self.conductor = ConductorPlanner(
            self.pool,
            base_model=config.conductor.base_model,
            checkpoint=config.conductor.checkpoint,
            max_steps=config.conductor.max_steps,
            use_heuristic_fallback=config.conductor.use_heuristic_fallback,
        )

    def resolve_mode(self, model: str, messages: list[dict[str, Any]]) -> str:
        if model in ("openfugu", "fugu"):
            return "router"
        if model in ("openfugu-ultra", "fugu-ultra", "fugu-ultra-20260615"):
            return "conductor"
        if model == "auto":
            prompt = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    prompt = str(m.get("content", ""))
                    break
            return "conductor" if self.pool.is_complex_prompt(prompt) else "router"
        # Direct worker by name
        if model in {w.name for w in self.pool.workers}:
            return "direct"
        return "router"

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        request_id: str | None = None,
    ) -> ChatCompletionResponse:
        rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        mode = self.resolve_mode(model, messages)

        if mode == "direct":
            worker = self.pool.get_by_name(model)
            response = await self.pool.complete(
                worker.id,
                messages,
                request_id=rid,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.content
            meta = OpenFuguMetadata(
                mode="direct",
                routing_decision=RoutingDecision(
                    mode="direct",
                    worker_id=worker.id,
                    worker_name=worker.name,
                    confidence=1.0,
                    strategy="explicit",
                ),
                workers_used=[worker.name],
                total_cost_usd=response.usage.cost_usd,
            )
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        elif mode == "conductor":
            query = next(
                (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            wf_result = await self.conductor.run(query, messages, request_id=rid)
            content = wf_result.final_answer
            meta = OpenFuguMetadata(
                mode="ultra",
                workflow=wf_result.to_trace(),
                workers_used=wf_result.workers_used,
                total_cost_usd=self.pool.total_cost(),
            )
            usage = UsageInfo()
        else:
            content, decision = await self.router.complete(messages, request_id=rid)
            meta = OpenFuguMetadata(
                mode="router",
                routing_decision=RoutingDecision(
                    mode="router",
                    worker_id=decision.worker_id,
                    worker_name=decision.worker_name,
                    confidence=decision.confidence,
                    strategy=decision.strategy,
                    latency_ms=decision.latency_ms,
                    logits=decision.logits,
                ),
                workers_used=[decision.worker_name],
                total_cost_usd=self.pool.total_cost(),
            )
            last_usage = self.pool.usage_history()[-1] if self.pool.usage_history() else None
            usage = UsageInfo(
                prompt_tokens=last_usage.input_tokens if last_usage else 0,
                completion_tokens=last_usage.output_tokens if last_usage else 0,
                total_tokens=(last_usage.input_tokens + last_usage.output_tokens) if last_usage else 0,
            )

        return ChatCompletionResponse(
            id=rid,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=usage,
            openfugu=meta,
        )
