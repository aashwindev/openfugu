"""Fugu router inference: hidden state -> worker selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfugu.workers.pool import WorkerPool


@dataclass
class RouteDecision:
    worker_id: int
    worker_name: str
    confidence: float
    logits: list[float]
    strategy: str
    latency_ms: float = 0.0


class FuguRouter:
    """Fast path router (Fugu / simplified Trinity)."""

    def __init__(
        self,
        pool: WorkerPool,
        *,
        backbone_name: str = "Qwen/Qwen3-0.6B",
        checkpoint: str | None = None,
        temperature: float = 1.0,
        use_heuristic_fallback: bool = True,
        device: str | None = None,
    ) -> None:
        self.pool = pool
        self.temperature = temperature
        self.use_heuristic_fallback = use_heuristic_fallback
        self._backbone = None
        self._head = None
        self._checkpoint_path = checkpoint
        self._backbone_name = backbone_name
        self._device_str = device
        self._loaded = False

    def _get_device(self):
        import torch

        if self._device_str:
            return torch.device(self._device_str)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._head is not None

        try:
            import torch
            from openfugu.router.backbone import RouterBackbone
            from openfugu.router.head import SelectionHead
        except ImportError:
            self._loaded = True
            return False

        device = self._get_device()

        if self._checkpoint_path and Path(self._checkpoint_path).exists():
            try:
                ckpt = torch.load(self._checkpoint_path, map_location=device, weights_only=False)
                self._backbone = RouterBackbone(
                    model_name=ckpt.get("backbone", self._backbone_name),
                    freeze=True,
                ).to(device)
                self._head = SelectionHead(
                    self._backbone.hidden_dim,
                    self.pool.size,
                ).to(device)
                self._head.load_state_dict(ckpt["head"])
                self._loaded = True
                return True
            except Exception:
                pass

        if not self.use_heuristic_fallback:
            try:
                self._backbone = RouterBackbone(model_name=self._backbone_name, freeze=True).to(device)
                self._head = SelectionHead(self._backbone.hidden_dim, self.pool.size).to(device)
                self._loaded = True
                return True
            except Exception:
                return False

        self._loaded = True
        return False

    def route(self, prompt: str) -> RouteDecision:
        import time

        start = time.perf_counter()

        if self._ensure_loaded() and self._backbone and self._head:
            import torch
            import torch.nn.functional as F

            device = self._get_device()
            self._backbone.eval()
            self._head.eval()
            with torch.no_grad():
                out = self._backbone.encode(prompt, device=device)
                logits = self._head(out.hidden_state)[0]
                probs = F.softmax(logits / self.temperature, dim=-1)
                worker_idx = int(torch.argmax(probs).item())
                confidence = float(probs[worker_idx].item())
                logits_list = logits.cpu().tolist()

            worker = self.pool.workers[worker_idx]
            latency = (time.perf_counter() - start) * 1000
            return RouteDecision(
                worker_id=worker.id,
                worker_name=worker.name,
                confidence=confidence,
                logits=logits_list,
                strategy="trained_router",
                latency_ms=latency,
            )

        worker_id = self.pool.heuristic_route(prompt)
        worker = self.pool.get(worker_id)
        latency = (time.perf_counter() - start) * 1000
        return RouteDecision(
            worker_id=worker_id,
            worker_name=worker.name,
            confidence=0.5,
            logits=[],
            strategy="heuristic",
            latency_ms=latency,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str = "",
    ) -> tuple[str, RouteDecision]:
        prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt = str(m.get("content", ""))
                break

        decision = self.route(prompt)
        response = await self.pool.complete(
            decision.worker_id,
            messages,
            request_id=request_id,
        )
        return response.content, decision

    def save_checkpoint(self, path: str | Path) -> None:
        import torch

        if not self._head:
            raise RuntimeError("No head to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self._backbone_name,
                "head": self._head.state_dict(),
                "num_workers": self.pool.size,
            },
            path,
        )
