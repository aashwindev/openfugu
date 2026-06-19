"""OpenAI-compatible request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RoutingDecision(BaseModel):
    mode: str
    worker_id: int | None = None
    worker_name: str | None = None
    confidence: float | None = None
    strategy: str | None = None
    latency_ms: float | None = None
    logits: list[float] = Field(default_factory=list)


class OpenFuguMetadata(BaseModel):
    mode: str
    routing_decision: RoutingDecision | None = None
    workflow: dict[str, Any] | None = None
    workers_used: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    openfugu: OpenFuguMetadata | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "openfugu"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
