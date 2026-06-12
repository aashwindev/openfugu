"""Per-turn re-routing for multi-step agentic loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openfugu.router.inference import FuguRouter, RouteDecision
from openfugu.workers.pool import WorkerPool


@dataclass
class TurnRecord:
    turn: int
    state_summary: str
    decision: RouteDecision
    response: str


@dataclass
class MultiStepResult:
    turns: list[TurnRecord] = field(default_factory=list)
    final_answer: str = ""
    workers_used: list[str] = field(default_factory=list)


class MultiStepRouter:
    """Re-route each turn in an agentic loop (up to max_turns)."""

    def __init__(self, router: FuguRouter, max_turns: int = 5) -> None:
        self.router = router
        self.max_turns = max_turns

    async def run(
        self,
        initial_query: str,
        *,
        request_id: str = "",
        terminal_check: Any | None = None,
    ) -> MultiStepResult:
        transcript = f"Task: {initial_query}\n"
        result = MultiStepResult()
        workers_used: list[str] = []

        for turn in range(self.max_turns):
            decision = self.router.route(transcript)
            messages = [{"role": "user", "content": transcript}]
            response = await self.router.pool.complete(
                decision.worker_id,
                messages,
                request_id=request_id,
            )

            if decision.worker_name not in workers_used:
                workers_used.append(decision.worker_name)

            result.turns.append(
                TurnRecord(
                    turn=turn,
                    state_summary=transcript[:200],
                    decision=decision,
                    response=response.content,
                )
            )
            transcript += f"\n[Turn {turn} - {decision.worker_name}]: {response.content}\n"

            if terminal_check and terminal_check(response.content):
                break

        result.final_answer = result.turns[-1].response if result.turns else ""
        result.workers_used = workers_used
        return result
