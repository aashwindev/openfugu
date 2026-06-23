"""Format and correctness reward functions."""

from __future__ import annotations

import re
from typing import Any

from openfugu.conductor.parser import parse_workflow


def grade_exact_match(answer: str, ground_truth: str) -> bool:
    a = _normalize(answer)
    g = _normalize(ground_truth)
    return a == g or g in a


def grade_mmlu(answer: str, ground_truth: str) -> bool:
    """Letter-match for multiple choice."""
    match = re.search(r"\b([A-D])\b", answer.upper())
    if match:
        return match.group(1) == ground_truth.strip().upper()[:1]
    return grade_exact_match(answer, ground_truth)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.-]", "", text)
    return text


def conductor_reward(
    conductor_output: str,
    final_answer: str,
    ground_truth: str,
    *,
    grade_fn: Any = grade_exact_match,
) -> float:
    """Conductor reward: 0 malformed, 1 correct, 0.5 wrong."""
    if parse_workflow(conductor_output) is None:
        return 0.0
    if grade_fn(final_answer, ground_truth):
        return 1.0
    return 0.5


def router_soft_targets(
    rewards: list[float],
    temperature: float = 1.0,
) -> list[float]:
    """Softmax over per-worker mean rewards."""
    import math

    if not rewards:
        return []
    scaled = [r / temperature for r in rewards]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]
