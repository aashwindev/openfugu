"""Per-agent tool call loop with isolation."""

from __future__ import annotations

from typing import Any, Callable, Awaitable


async def run_tool_loop(
    agent_key: str,
    messages: list[dict[str, Any]],
    complete_fn: Callable[..., Awaitable[str]],
    tool_handlers: dict[str, Callable[..., Any]],
    *,
    max_rounds: int = 10,
) -> tuple[str, list[dict[str, Any]]]:
    """Run function-calling loop for a single isolated agent."""
    transcript: list[dict[str, Any]] = []
    current_messages = list(messages)

    for _ in range(max_rounds):
        response = await complete_fn(current_messages)
        transcript.append({"role": "assistant", "content": response})

        # Simple tool detection: lines starting with TOOL:
        if response.strip().startswith("TOOL:"):
            parts = response.strip().split(":", 2)
            tool_name = parts[1].strip() if len(parts) > 1 else ""
            tool_input = parts[2].strip() if len(parts) > 2 else ""
            handler = tool_handlers.get(tool_name)
            if handler:
                result = handler(tool_input)
                tool_msg = {"role": "tool", "content": str(result), "name": tool_name}
                transcript.append(tool_msg)
                current_messages.append({"role": "assistant", "content": response})
                current_messages.append(tool_msg)
                continue

        return response, transcript

    return transcript[-1].get("content", "") if transcript else "", transcript
