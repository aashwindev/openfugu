"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None


class RouterConfig(BaseModel):
    checkpoint: str | None = None
    backbone: str = "Qwen/Qwen3-0.6B"
    temperature: float = 1.0
    use_heuristic_fallback: bool = True


class ConductorConfig(BaseModel):
    checkpoint: str | None = None
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_steps: int = 5
    max_recursion_depth: int = 2
    use_heuristic_fallback: bool = True


class WorkerConfig(BaseModel):
    id: int
    name: str
    provider: str
    model: str
    api_base: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    description: str = ""
    prefer_local: bool = False


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    default_model: str = "openfugu"
    router: RouterConfig = Field(default_factory=RouterConfig)
    conductor: ConductorConfig = Field(default_factory=ConductorConfig)
    workers: list[WorkerConfig] = Field(default_factory=list)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


def find_default_config() -> Path:
    candidates = [
        Path("config/default.yaml"),
        Path(__file__).resolve().parents[2] / "config" / "default.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("config/default.yaml not found")
