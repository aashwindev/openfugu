"""LiteLLM worker pool and cost tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openfugu.config import WorkerConfig


@dataclass
class UsageRecord:
    worker_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    request_id: str


@dataclass
class WorkerResponse:
    content: str
    worker_id: int
    worker_name: str
    model: str
    usage: UsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class WorkerPool:
    """Registry of worker models invoked via LiteLLM."""

    def __init__(self, workers: list[WorkerConfig]) -> None:
        if not workers:
            raise ValueError("Worker pool must have at least one worker")
        self._workers = {w.id: w for w in workers}
        self._by_name = {w.name: w for w in workers}
        self._usage: list[UsageRecord] = []

    @property
    def workers(self) -> list[WorkerConfig]:
        return list(self._workers.values())

    @property
    def size(self) -> int:
        return len(self._workers)

    def get(self, worker_id: int) -> WorkerConfig:
        if worker_id not in self._workers:
            raise KeyError(f"Unknown worker id {worker_id}")
        return self._workers[worker_id]

    def get_by_name(self, name: str) -> WorkerConfig:
        if name not in self._by_name:
            raise KeyError(f"Unknown worker name {name}")
        return self._by_name[name]

    def usage_history(self) -> list[UsageRecord]:
        return list(self._usage)

    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self._usage)

    async def complete(
        self,
        worker_id: int,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> WorkerResponse:
        import litellm

        worker = self.get(worker_id)
        kwargs: dict[str, Any] = {
            "model": f"{worker.provider}/{worker.model}",
            "messages": messages,
            "temperature": temperature,
        }
        if worker.api_base:
            kwargs["api_base"] = worker.api_base
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (
            input_tokens * worker.cost_per_input_token
            + output_tokens * worker.cost_per_output_token
        )

        record = UsageRecord(
            worker_name=worker.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            request_id=request_id,
        )
        self._usage.append(record)

        return WorkerResponse(
            content=content,
            worker_id=worker_id,
            worker_name=worker.name,
            model=worker.model,
            usage=record,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def heuristic_route(self, prompt: str) -> int:
        """Capability + cost router when trained checkpoint unavailable."""
        text = prompt.lower()
        scores: list[tuple[int, float]] = []

        for w in self.workers:
            score = 0.0
            if w.prefer_local:
                score += 0.3
            if "code" in text or "python" in text or "debug" in text:
                if "code" in w.capabilities or "debug" in w.capabilities:
                    score += 2.0
            if "math" in text or "prove" in text or "calculate" in text:
                if "reasoning" in w.capabilities:
                    score += 1.5
            if "write" in text or "explain" in text:
                if "chat" in w.capabilities:
                    score += 1.0
            # Prefer lower cost when scores tie
            score -= (w.cost_per_input_token + w.cost_per_output_token) * 1e6 * 0.01
            scores.append((w.id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def is_complex_prompt(self, prompt: str) -> bool:
        """Heuristic for auto mode: route to conductor on complex tasks."""
        text = prompt.lower()
        complex_signals = [
            "implement",
            "build",
            "design",
            "analyze",
            "compare",
            "step by step",
            "multi",
            "test and",
            "refactor",
            "architecture",
        ]
        if len(prompt) > 800:
            return True
        return sum(1 for s in complex_signals if s in text) >= 2
