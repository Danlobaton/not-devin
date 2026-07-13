"""Runtime guard for proposed tool calls: validation and loop detection."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from .state import AgentState

MAX_INVALID_STRIKES = 3
MAX_SIGNATURE_REPEATS = 3

_SIBLING_REJECTION = "not executed: a sibling tool call in this batch was invalid"


def batch_signature(tool_calls: list[dict]) -> str:
    """Return a stable signature for a batch of proposed tool calls.

    Args:
        tool_calls: Tool-call dicts from an AIMessage, in call order.

    Returns:
        A JSON string of [name, args] pairs, stable across dict key order.
    """
    return json.dumps(
        [[call["name"], call["args"]] for call in tool_calls],
        sort_keys=True,
    )


def build_guard_node(
    tools: Sequence[BaseTool],
) -> Callable[[AgentState], dict]:
    """Build a graph node that vets tool calls before execution.

    The node kills stuck runs (the final identical batch never executes,
    ``terminal_reason="repeated_tool_calls"``), rejects invalid batches
    back to the agent with per-call error ToolMessages, and ends the run
    with ``terminal_reason="invalid_tool_call"`` once ``MAX_INVALID_STRIKES``
    invalid batches have been proposed.

    Args:
        tools: The tools registered on the graph's tool node.

    Returns:
        A node callable producing a partial AgentState update.
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def guard_node(state: AgentState) -> dict:
        tool_calls = state["messages"][-1].tool_calls

        signature = batch_signature(tool_calls)
        if signature == state.get("last_tool_signature"):
            repeats = state.get("signature_repeats", 1) + 1
        else:
            repeats = 1
        signature_updates = {
            "last_tool_signature": signature,
            "signature_repeats": repeats,
        }
        if repeats >= MAX_SIGNATURE_REPEATS:
            return {**signature_updates, "terminal_reason": "repeated_tool_calls"}

        errors: dict[str, str] = {}
        for call in tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                errors[call["id"]] = f"unknown tool: {call['name']}"
                continue
            try:
                tool.tool_call_schema.model_validate(call["args"])
            except ValidationError as error:
                errors[call["id"]] = (
                    f"invalid arguments for {call['name']}: {error}"
                )

        if not errors:
            return signature_updates

        strikes = state.get("invalid_strikes", 0) + 1
        if strikes >= MAX_INVALID_STRIKES:
            return {
                **signature_updates,
                "invalid_strikes": strikes,
                "terminal_reason": "invalid_tool_call",
            }

        rejections = [
            ToolMessage(
                content=errors.get(call["id"], _SIBLING_REJECTION),
                tool_call_id=call["id"],
            )
            for call in tool_calls
        ]
        return {
            **signature_updates,
            "invalid_strikes": strikes,
            "messages": rejections,
        }

    return guard_node
