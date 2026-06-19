"""FastAPI application factory."""

from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from openfugu.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelListResponse,
)
from openfugu.config import AppConfig
from openfugu.service import OPENFUGU_MODELS, OpenFuguService

_bearer = HTTPBearer(auto_error=False)


def create_app(config: AppConfig) -> FastAPI:
    service = OpenFuguService(config)
    app = FastAPI(
        title="OpenFugu",
        description="Open-source LLM orchestration: router + conductor",
        version="0.1.0",
    )

    def verify_auth(
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> None:
        expected = config.server.api_key or os.environ.get("OPENFUGU_API_KEY")
        if not expected:
            return
        if credentials is None or credentials.credentials != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "workers": str(service.pool.size)}

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models(_: None = Depends(verify_auth)) -> ModelListResponse:
        now = int(time.time())
        models = [
            ModelInfo(id=m, created=now, owned_by="openfugu") for m in sorted(OPENFUGU_MODELS)
        ]
        for w in service.pool.workers:
            models.append(ModelInfo(id=w.name, created=now, owned_by="worker"))
        return ModelListResponse(data=models)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        body: ChatCompletionRequest,
        _: None = Depends(verify_auth),
    ) -> ChatCompletionResponse:
        if body.stream:
            raise HTTPException(status_code=501, detail="Streaming not yet implemented")
        messages = [m.model_dump(exclude_none=True) for m in body.messages]
        return await service.chat_completion(
            body.model,
            messages,
            temperature=body.temperature or 0.7,
            max_tokens=body.max_tokens,
        )

    @app.post("/v1/responses")
    async def responses_endpoint(
        body: dict,
        _: None = Depends(verify_auth),
    ) -> dict:
        """OpenAI-compatible responses endpoint — routes to conductor."""
        messages = body.get("input", body.get("messages", []))
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        model = body.get("model", "openfugu-ultra")
        result = await service.chat_completion(model, messages)
        return {
            "id": result.id,
            "object": "response",
            "model": result.model,
            "output": result.choices[0].message.content,
            "openfugu": result.openfugu.model_dump() if result.openfugu else None,
        }

    @app.get("/v1/stats")
    async def stats(_: None = Depends(verify_auth)) -> dict:
        history = service.pool.usage_history()
        return {
            "total_requests": len(history),
            "total_cost_usd": service.pool.total_cost(),
            "by_worker": _aggregate_by_worker(history),
        }

    return app


def _aggregate_by_worker(history: list) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for u in history:
        if u.worker_name not in agg:
            agg[u.worker_name] = {"requests": 0, "tokens": 0, "cost_usd": 0.0}
        agg[u.worker_name]["requests"] += 1
        agg[u.worker_name]["tokens"] += u.input_tokens + u.output_tokens
        agg[u.worker_name]["cost_usd"] += u.cost_usd
    return agg
