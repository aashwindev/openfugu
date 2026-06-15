"""Conductor workflow types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowStep:
    step_index: int
    subtask: str
    worker_id: int
    access_list: list[int | str]


@dataclass
class Workflow:
    steps: list[WorkflowStep]
    raw_output: str = ""

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def topology_label(self) -> str:
        if self.num_steps <= 1:
            return "single"
        if self.num_steps == 2:
            return "chain"
        # Simple heuristic: parallel if multiple steps share no deps
        return "tree" if self.num_steps >= 3 else "chain"


@dataclass
class StepResult:
    step_index: int
    subtask: str
    worker_id: int
    worker_name: str
    response: str
    tool_transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowResult:
    workflow: Workflow
    step_results: list[StepResult]
    final_answer: str
    workers_used: list[str]

    def to_trace(self) -> dict[str, Any]:
        return {
            "steps": self.workflow.num_steps,
            "topology": self.workflow.topology_label(),
            "workers_used": self.workers_used,
            "step_results": [
                {
                    "step": r.step_index,
                    "worker": r.worker_name,
                    "subtask": r.subtask,
                }
                for r in self.step_results
            ],
        }
