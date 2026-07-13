from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict

from not_devin.event_log.replay import apply_delta, rebuild_state


def test_apply_delta_merges_messages_and_scalars() -> None:
    state = {
        "messages": [HumanMessage(content="fix bug")],
        "workspace": "/tmp/ws",
        "iteration": 0,
        "max_iterations": 5,
        "terminal_reason": None,
    }
    delta = {"messages": [AIMessage(content="on it")], "iteration": 1, "terminal_reason": None}

    updated = apply_delta(state, delta)

    assert [message.content for message in updated["messages"]] == ["fix bug", "on it"]
    assert updated["iteration"] == 1
    assert updated["workspace"] == "/tmp/ws"


def test_rebuilds_state_from_run_started_and_deltas() -> None:
    events = [
        {
            "type": "run_started",
            "task": "fix the bug",
            "workspace": "/tmp/ws",
            "max_iterations": 5,
        },
        {
            "type": "state_delta",
            "node": "agent",
            "delta": {
                "messages": messages_to_dict([AIMessage(content="looking into it")]),
                "iteration": 1,
                "terminal_reason": None,
            },
        },
    ]

    state = rebuild_state(events)

    assert [type(message).__name__ for message in state["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert state["messages"][0].content == "fix the bug"
    assert state["messages"][1].content == "looking into it"
    assert state["workspace"] == "/tmp/ws"
    assert state["max_iterations"] == 5
    assert state["iteration"] == 1
    assert state["terminal_reason"] is None


def test_raises_without_run_started_event() -> None:
    with pytest.raises(ValueError, match="run_started"):
        rebuild_state([{"type": "state_delta", "node": "agent", "delta": {}}])
