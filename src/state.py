from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


TerminalReason = Literal[
    "success",
    "iteration_limit",
    "timeout",
    "repeated_tool_calls",
    "approval_denied",
    "invalid_tool_call",
    "provider_failure",
]


class AgentState(TypedDict):
    """Runtime state for one coding-agent run."""

    messages: Annotated[list[AnyMessage], add_messages]
    workspace: str
    iteration: int
    max_iterations: int
    terminal_reason: NotRequired[TerminalReason | None]
