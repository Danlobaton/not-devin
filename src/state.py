from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


TerminalReason = Literal[
    "success",
    "iteration_limit",
    "model_call_budget",
    "token_budget",
    "cost_budget",
    "timeout",
    "repeated_tool_calls",
    "approval_denied",
    "invalid_tool_call",
    "provider_failure",
]


class UsageState(TypedDict):
    """Serializable cumulative model usage."""

    attempts: int
    successful_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reserved_tokens: int
    estimated_cost_usd: float


class AgentState(TypedDict):
    """Runtime state for one coding-agent run."""

    messages: Annotated[list[AnyMessage], add_messages]
    workspace: str
    iteration: int
    max_iterations: int
    terminal_reason: NotRequired[TerminalReason | None]
    usage: NotRequired[UsageState]
    run_started_at: NotRequired[str]
    run_deadline: NotRequired[str]
    last_provider_error: NotRequired[str | None]
