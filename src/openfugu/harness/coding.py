"""Mini coding harness for CMA-ES end-to-end router training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodingTask:
    task_id: str
    description: str
    test_code: str
    expected_pass: bool = True


FALLBACK_TASKS: list[CodingTask] = [
    CodingTask(
        task_id="reverse_string",
        description="Write a function reverse_string(s) that reverses a string.",
        test_code="assert reverse_string('hello') == 'olleh'",
    ),
    CodingTask(
        task_id="add_numbers",
        description="Write a function add(a, b) returning a + b.",
        test_code="assert add(2, 3) == 5",
    ),
]


def evaluate_code_submission(code: str, task: CodingTask) -> bool:
    """Execute generated code against a simple assertion."""
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102
        exec(task.test_code, namespace)  # noqa: S102
        return True
    except Exception:
        return False


def terminal_reward(completed: bool) -> float:
    return 1.0 if completed else 0.0
