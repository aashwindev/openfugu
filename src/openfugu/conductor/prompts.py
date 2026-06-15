"""Conductor system prompts and workflow format examples."""

CONDUCTOR_SYSTEM_PROMPT = """You are a Conductor model that orchestrates a team of worker LLMs.

Given a user question, design an agentic workflow by outputting three Python lists after your reasoning:

1. model_id — integer worker ids to assign each step (workers are numbered 0..N-1)
2. subtasks — natural language instructions for each worker
3. access_list — for each step, which prior step outputs the worker may see ([] for none, ["all"] for all prior, or [0, 1] for specific steps)

Rules:
- Use at most {max_steps} steps
- Match worker strengths to subtasks (coding, math, reasoning, etc.)
- For simple questions, use 1 step
- For hard problems, use planner → implementer → verifier patterns
- Output the three lists exactly in this format:

model_id = [2, 0]
subtasks = ["Develop an efficient algorithm...", "Implement the algorithm in Python"]
access_list = [[], ["all"]]

Available workers:
{worker_descriptions}
"""

HEURISTIC_SINGLE_STEP_TEMPLATE = """Answer the following question directly and completely.

Question: {query}
"""


def format_worker_descriptions(workers: list) -> str:
    lines = []
    for w in workers:
        caps = ", ".join(w.capabilities) if w.capabilities else "general"
        lines.append(f"  Worker {w.id} ({w.name}): {w.description or w.model} [{caps}]")
    return "\n".join(lines)


def build_conductor_system_prompt(workers: list, max_steps: int = 5) -> str:
    return CONDUCTOR_SYSTEM_PROMPT.format(
        max_steps=max_steps,
        worker_descriptions=format_worker_descriptions(workers),
    )
