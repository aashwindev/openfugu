"""Conductor planner: trained checkpoint or heuristic fallback."""

from __future__ import annotations

import json
from typing import Any

from openfugu.conductor.executor import WorkflowExecutor
from openfugu.conductor.parser import parse_workflow
from openfugu.conductor.prompts import (
    HEURISTIC_SINGLE_STEP_TEMPLATE,
    build_conductor_system_prompt,
)
from openfugu.conductor.workflow import Workflow, WorkflowResult, WorkflowStep
from openfugu.workers.pool import WorkerPool


class ConductorPlanner:
    def __init__(
        self,
        pool: WorkerPool,
        *,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        checkpoint: str | None = None,
        max_steps: int = 5,
        use_heuristic_fallback: bool = True,
    ) -> None:
        self.pool = pool
        self.base_model = base_model
        self.checkpoint = checkpoint
        self.max_steps = max_steps
        self.use_heuristic_fallback = use_heuristic_fallback
        self.executor = WorkflowExecutor(pool, max_steps=max_steps)
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None or not self.checkpoint:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except ImportError:
            pass

    async def _generate_workflow_text(self, query: str) -> str:
        self._load_model()
        system = build_conductor_system_prompt(self.pool.workers, self.max_steps)

        if self._model is not None and self._tokenizer is not None:
            import torch

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
            return self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        # Heuristic: use strongest worker via pool router for single-step
        worker_id = self.pool.heuristic_route(query)
        if not self.pool.is_complex_prompt(query):
            workflow = Workflow(
                steps=[
                    WorkflowStep(
                        step_index=0,
                        subtask=HEURISTIC_SINGLE_STEP_TEMPLATE.format(query=query),
                        worker_id=worker_id,
                        access_list=[],
                    )
                ]
            )
            return workflow.raw_output or self._workflow_to_text(workflow)

        # Multi-step heuristic: plan → execute
        planner_id = self.pool.heuristic_route("plan architecture " + query)
        executor_id = self.pool.heuristic_route("implement code " + query)
        return (
            f'model_id = [{planner_id}, {executor_id}]\n'
            f'subtasks = ["Analyze and plan an approach: {query[:200]}", '
            f'"Implement the solution based on the plan"]\n'
            f'access_list = [[], ["all"]]'
        )

    @staticmethod
    def _workflow_to_text(workflow: Workflow) -> str:
        ids = [s.worker_id for s in workflow.steps]
        tasks = [s.subtask for s in workflow.steps]
        access = [s.access_list for s in workflow.steps]
        return (
            f"model_id = {ids}\n"
            f"subtasks = {json.dumps(tasks)}\n"
            f"access_list = {access}"
        )

    async def run(
        self,
        query: str,
        messages: list[dict[str, Any]] | None = None,
        *,
        session_id: str | None = None,
        request_id: str = "",
        max_retries: int = 2,
    ) -> WorkflowResult:
        text = query
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user":
                    text = str(m.get("content", ""))
                    break

        conductor_output = await self._generate_workflow_text(text)
        workflow = parse_workflow(conductor_output, max_steps=self.max_steps)

        for attempt in range(max_retries):
            if workflow is not None:
                break
            if attempt < max_retries - 1:
                conductor_output = await self._generate_workflow_text(
                    text + "\n\nOutput valid model_id, subtasks, access_list Python lists."
                )
                workflow = parse_workflow(conductor_output, max_steps=self.max_steps)

        if workflow is None and self.use_heuristic_fallback:
            worker_id = self.pool.heuristic_route(text)
            workflow = Workflow(
                steps=[
                    WorkflowStep(
                        step_index=0,
                        subtask=text,
                        worker_id=worker_id,
                        access_list=[],
                    )
                ],
                raw_output=conductor_output,
            )
        elif workflow is None:
            raise ValueError("Failed to parse conductor workflow output")

        result = await self.executor.execute(
            workflow, text, session_id=session_id, request_id=request_id
        )
        return result
