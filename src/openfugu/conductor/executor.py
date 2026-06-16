"""Sequential workflow executor with memory isolation."""

from __future__ import annotations

import uuid
from typing import Any

from openfugu.conductor.memory import WorkflowMemory
from openfugu.conductor.parser import parse_workflow
from openfugu.conductor.workflow import StepResult, Workflow, WorkflowResult
from openfugu.workers.pool import WorkerPool


class WorkflowExecutor:
    def __init__(self, pool: WorkerPool, max_steps: int = 5) -> None:
        self.pool = pool
        self.max_steps = max_steps
        self._sessions: dict[str, WorkflowMemory] = {}

    def get_session(self, session_id: str | None = None) -> WorkflowMemory:
        sid = session_id or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = WorkflowMemory(session_id=sid)
        return self._sessions[sid]

    async def execute(
        self,
        workflow: Workflow,
        original_query: str,
        *,
        session_id: str | None = None,
        request_id: str = "",
    ) -> WorkflowResult:
        memory = self.get_session(session_id)
        step_results: list[StepResult] = []
        workers_used: list[str] = []

        for step in workflow.steps:
            worker = self.pool.get(step.worker_id)
            messages = memory.build_context_messages(original_query, step)

            # Append agent-isolated prior messages (only this agent's tool history)
            transcript = memory.get_agent_transcript(step.step_index, step.worker_id)
            if transcript.messages:
                messages = messages[:-1] + transcript.messages + [messages[-1]]

            response = await self.pool.complete(
                step.worker_id,
                messages,
                request_id=request_id,
            )

            transcript.messages.append({"role": "assistant", "content": response.content})
            memory.record_step_output(step.step_index, step.subtask, response.content)

            if worker.name not in workers_used:
                workers_used.append(worker.name)

            step_results.append(
                StepResult(
                    step_index=step.step_index,
                    subtask=step.subtask,
                    worker_id=step.worker_id,
                    worker_name=worker.name,
                    response=response.content,
                )
            )

        memory.clear_workflow_state()
        final_answer = step_results[-1].response if step_results else ""

        return WorkflowResult(
            workflow=workflow,
            step_results=step_results,
            final_answer=final_answer,
            workers_used=workers_used,
        )

    async def execute_from_output(
        self,
        conductor_output: str,
        original_query: str,
        **kwargs: Any,
    ) -> WorkflowResult | None:
        workflow = parse_workflow(conductor_output, max_steps=self.max_steps)
        if workflow is None:
            return None
        return await self.execute(workflow, original_query, **kwargs)
