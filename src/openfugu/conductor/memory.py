"""Intra-workflow isolation and inter-workflow shared memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTranscript:
    """Isolated per-agent tool + message history within a workflow."""

    agent_key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowMemory:
    """Manages isolation within workflow + shared memory across turns."""

    session_id: str
    # Inter-workflow: shared tool artifacts across completed workflows
    _prior_session_artifacts: list[dict[str, Any]] = field(default_factory=list)
    # Current workflow tool artifacts (promoted on clear)
    _current_workflow_artifacts: list[dict[str, Any]] = field(default_factory=list)
    # Intra-workflow: per-agent isolated transcripts
    _agent_transcripts: dict[str, AgentTranscript] = field(default_factory=dict)
    # Step outputs visible via access_list
    _step_outputs: dict[int, tuple[str, str]] = field(default_factory=dict)

    def agent_key(self, step_index: int, worker_id: int) -> str:
        return f"step{step_index}_worker{worker_id}"

    def get_agent_transcript(self, step_index: int, worker_id: int) -> AgentTranscript:
        key = self.agent_key(step_index, worker_id)
        if key not in self._agent_transcripts:
            self._agent_transcripts[key] = AgentTranscript(agent_key=key)
        return self._agent_transcripts[key]

    def record_step_output(self, step_index: int, subtask: str, response: str) -> None:
        self._step_outputs[step_index] = (subtask, response)

    def build_context_messages(
        self,
        original_query: str,
        step: "WorkflowStep",
    ) -> list[dict[str, Any]]:
        from openfugu.conductor.workflow import WorkflowStep

        assert isinstance(step, WorkflowStep)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": original_query},
        ]

        # Access list: prior step outputs only (not tool transcripts)
        for access in step.access_list:
            if access == "all":
                for idx in sorted(self._step_outputs.keys()):
                    if idx < step.step_index:
                        subtask, response = self._step_outputs[idx]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"[Step {idx}] {subtask}\n{response}",
                            }
                        )
            elif isinstance(access, int) and access in self._step_outputs:
                subtask, response = self._step_outputs[access]
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Step {access}] {subtask}\n{response}",
                    }
                )

        # Inter-workflow shared memory: tool artifacts from prior completed workflows only
        if self._prior_session_artifacts:
            artifact_summary = "\n".join(
                f"- {a.get('type', 'tool')}: {a.get('summary', '')}"
                for a in self._prior_session_artifacts[-10:]
            )
            messages.append(
                {
                    "role": "system",
                    "content": f"Prior session tool artifacts:\n{artifact_summary}",
                }
            )

        messages.append({"role": "user", "content": step.subtask})
        return messages

    def record_tool_call(
        self,
        step_index: int,
        worker_id: int,
        tool_call: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        transcript = self.get_agent_transcript(step_index, worker_id)
        transcript.tool_calls.append({"call": tool_call, "result": result})
        self._current_workflow_artifacts.append(
            {
                "type": tool_call.get("name", "tool"),
                "summary": str(result.get("content", ""))[:500],
                "step": step_index,
                "worker_id": worker_id,
            }
        )

    def clear_workflow_state(self) -> None:
        """Reset intra-workflow state; promote tool artifacts to inter-workflow memory."""
        self._prior_session_artifacts.extend(self._current_workflow_artifacts)
        self._current_workflow_artifacts.clear()
        self._agent_transcripts.clear()
        self._step_outputs.clear()
