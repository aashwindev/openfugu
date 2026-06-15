"""Parse Conductor natural-language workflow output."""

from __future__ import annotations

import ast
import re
from typing import Any

from openfugu.conductor.workflow import Workflow, WorkflowStep


def _extract_bracket_list(text: str, var_name: str) -> list[Any] | None:
    pattern = re.compile(rf"{var_name}\s*=\s*", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    start = match.end()
    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(text[start : i + 1])
                except (SyntaxError, ValueError):
                    return None
                return value if isinstance(value, list) else None
    return None


def _extract_lists(output: str) -> tuple[list[int], list[str], list[Any]] | None:
    model_ids = _extract_bracket_list(output, "model_id")
    subtasks = _extract_bracket_list(output, "subtasks")
    access_list = _extract_bracket_list(output, "access_list")

    if model_ids is None or subtasks is None or access_list is None:
        return None
    if not (len(model_ids) == len(subtasks) == len(access_list)):
        return None
    if len(model_ids) == 0:
        return None

    try:
        model_ids_int = [int(m) for m in model_ids]
    except (TypeError, ValueError):
        return None

    subtasks_str = [str(s) for s in subtasks]
    return model_ids_int, subtasks_str, access_list


def parse_workflow(output: str, max_steps: int = 5) -> Workflow | None:
    """Extract workflow from conductor model output."""
    parsed = _extract_lists(output)
    if parsed is None:
        return None

    model_ids, subtasks, access_lists = parsed
    if len(model_ids) > max_steps:
        model_ids = model_ids[:max_steps]
        subtasks = subtasks[:max_steps]
        access_lists = access_lists[:max_steps]

    steps: list[WorkflowStep] = []
    for i, (wid, subtask, access) in enumerate(
        zip(model_ids, subtasks, access_lists, strict=True)
    ):
        normalized_access: list[int | str] = []
        if isinstance(access, list):
            for item in access:
                if isinstance(item, str) and item.lower() == "all":
                    normalized_access.append("all")
                else:
                    try:
                        normalized_access.append(int(item))
                    except (TypeError, ValueError):
                        pass
        steps.append(
            WorkflowStep(
                step_index=i,
                subtask=subtask,
                worker_id=wid,
                access_list=normalized_access,
            )
        )

    return Workflow(steps=steps, raw_output=output)


def is_well_formed(output: str) -> bool:
    return parse_workflow(output) is not None
