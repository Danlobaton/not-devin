"""Rebuilds AgentState from a run's event log for crash-recovery resume."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, messages_from_dict
from langgraph.graph.message import add_messages

from ..state import AgentState


def apply_delta(state: AgentState, delta: dict) -> AgentState:
    """Apply one node's state update onto an AgentState.

    ``delta["messages"]``, if present, must already be BaseMessage objects
    and merges via ``add_messages``; every other key copies
    last-write-wins, so new state fields never silently drop on resume.

    Args:
        state: The state to update (not mutated).
        delta: One node's partial update.

    Returns:
        A new AgentState with the delta applied.
    """
    updated = dict(state)
    for key, value in delta.items():
        if key == "messages":
            updated["messages"] = add_messages(state["messages"], value)
        else:
            updated[key] = value
    return updated


def rebuild_state(events: list[dict]) -> AgentState:
    """Replay run_started and state_delta events into an AgentState.

    Args:
        events: A list of event dictionaries from the event log. Must contain
            at least one "run_started" event.

    Returns:
        The final AgentState after replaying all state_delta events.

    Raises:
        ValueError: If the event log contains no run_started event.
    """
    run_started = next((event for event in events if event["type"] == "run_started"), None)
    if run_started is None:
        raise ValueError("event log has no run_started event")

    state: AgentState = {
        "messages": [HumanMessage(content=run_started["task"])],
        "workspace": run_started["workspace"],
        "iteration": 0,
        "max_iterations": run_started["max_iterations"],
        "terminal_reason": None,
        "verify_command": run_started.get("verify_command"),
    }

    for event in events:
        if event["type"] != "state_delta":
            continue
        delta = dict(event["delta"])
        if "messages" in delta:
            delta["messages"] = messages_from_dict(delta["messages"])
        state = apply_delta(state, delta)

    return state
