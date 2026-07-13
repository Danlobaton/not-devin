from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from not_devin.guard import (
    MAX_INVALID_STRIKES,
    MAX_SIGNATURE_REPEATS,
    batch_signature,
    build_guard_node,
)
from not_devin.tools import TOOLS

CALL_READ = {
    "name": "read_file",
    "args": {"path": "README.md"},
    "id": "call_1",
    "type": "tool_call",
}


def make_state(tool_calls: list[dict], **extra: object) -> dict:
    return {
        "messages": [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=tool_calls),
        ],
        "workspace": "/tmp/ws",
        "iteration": 1,
        "max_iterations": 5,
        **extra,
    }


def test_valid_batch_passes_through() -> None:
    update = build_guard_node(TOOLS)(make_state([CALL_READ]))

    assert "messages" not in update
    assert update.get("terminal_reason") is None
    assert update["signature_repeats"] == 1
    assert update["last_tool_signature"] == batch_signature([CALL_READ])


def test_unknown_tool_rejected_with_per_call_tool_messages() -> None:
    bad = {"name": "no_such_tool", "args": {}, "id": "call_1", "type": "tool_call"}
    good_sibling = {**CALL_READ, "id": "call_2"}

    update = build_guard_node(TOOLS)(make_state([bad, good_sibling]))

    messages = update["messages"]
    assert [message.tool_call_id for message in messages] == ["call_1", "call_2"]
    assert "unknown tool" in messages[0].content
    assert "sibling" in messages[1].content
    assert update["invalid_strikes"] == 1
    assert update.get("terminal_reason") is None


def test_bad_args_rejected() -> None:
    bad = {
        "name": "read_file",
        "args": {"not_a_field": 1},
        "id": "call_1",
        "type": "tool_call",
    }

    update = build_guard_node(TOOLS)(make_state([bad]))

    assert "invalid arguments" in update["messages"][0].content
    assert update["invalid_strikes"] == 1


def test_final_invalid_strike_terminates() -> None:
    bad = {"name": "no_such_tool", "args": {}, "id": "call_1", "type": "tool_call"}

    update = build_guard_node(TOOLS)(
        make_state([bad], invalid_strikes=MAX_INVALID_STRIKES - 1)
    )

    assert update["terminal_reason"] == "invalid_tool_call"
    assert "messages" not in update


def test_final_identical_batch_terminates_before_validation() -> None:
    update = build_guard_node(TOOLS)(
        make_state(
            [CALL_READ],
            last_tool_signature=batch_signature([CALL_READ]),
            signature_repeats=MAX_SIGNATURE_REPEATS - 1,
        )
    )

    assert update["terminal_reason"] == "repeated_tool_calls"


def test_different_batch_resets_repeat_counter() -> None:
    update = build_guard_node(TOOLS)(
        make_state(
            [CALL_READ],
            last_tool_signature="a-different-signature",
            signature_repeats=2,
        )
    )

    assert update["signature_repeats"] == 1
    assert update.get("terminal_reason") is None
