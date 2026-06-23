"""Dataset loaders for router and conductor training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class TrainingExample:
    question: str
    answer: str
    domain: str = "general"
    metadata: dict | None = None


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    path = Path(path)
    if not path.exists():
        return []
    examples: list[TrainingExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                TrainingExample(
                    question=obj["question"],
                    answer=obj.get("answer", obj.get("ground_truth", "")),
                    domain=obj.get("domain", "general"),
                    metadata=obj.get("metadata"),
                )
            )
    return examples


def load_huggingface_math(split: str = "train[:100]") -> list[TrainingExample]:
    try:
        from datasets import load_dataset

        ds = load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)
        return [
            TrainingExample(
                question=row["problem"],
                answer=row["solution"],
                domain="math",
            )
            for row in ds
        ]
    except Exception:
        return _fallback_math_examples()


def load_huggingface_mmlu(config: str = "abstract_algebra", split: str = "test[:100]") -> list[TrainingExample]:
    try:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", config, split=split)
        return [
            TrainingExample(
                question=f"{row['question']}\nChoices: {row['choices']}",
                answer=row["answer"],
                domain="mmlu",
            )
            for row in ds
        ]
    except Exception:
        return []


def build_training_mix(config: dict) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for source in config.get("dataset", []):
        name = source.get("name", "")
        split = source.get("split", "train[:100]")
        if name == "math500" or name == "math":
            examples.extend(load_huggingface_math(split))
        elif name == "mmlu":
            examples.extend(
                load_huggingface_mmlu(source.get("config", "abstract_algebra"), split)
            )
    if not examples:
        examples = _fallback_math_examples()
    return examples


def _fallback_math_examples() -> list[TrainingExample]:
    return [
        TrainingExample("What is 2+2?", "4", domain="math"),
        TrainingExample("Solve x^2 - 5x + 6 = 0", "x=2 or x=3", domain="math"),
        TrainingExample("Derivative of x^3", "3x^2", domain="math"),
        TrainingExample("What is the capital of France?", "Paris", domain="factual"),
        TrainingExample("Write a Python function to reverse a string", "def reverse(s): return s[::-1]", domain="code"),
    ]


def iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
