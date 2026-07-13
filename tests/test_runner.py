from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import Field

from not_devin.event_log.reader import read_events
from not_devin.runner import resume_agent, run_agent


class ScriptedToolCallingModel(GenericFakeChatModel):
    bound_tools: list[BaseTool] = Field(default_factory=list)

    def bind_tools(
        self, tools: Sequence[BaseTool], *, tool_choice: str | None = None, **kwargs: Any
    ) -> "ScriptedToolCallingModel":
        self.bound_tools = list(tools)
        return self


def _scripted_tool_call_model() -> ScriptedToolCallingModel:
    return ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "README.md"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )


def test_run_agent_logs_every_event_and_returns_final_state(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("fixture content")

    run_id, final_state = run_agent(
        _scripted_tool_call_model(),
        task="inspect README",
        workspace=str(tmp_path),
        max_iterations=3,
        root=tmp_path / "runs",
    )

    events = read_events(tmp_path / "runs" / run_id / "events.jsonl")
    event_types = [event["type"] for event in events]
    assert event_types == [
        "run_started",
        "llm_start",
        "llm_end",
        "state_delta",   # agent
        "state_delta",   # guard (signature bookkeeping)
        "tool_start",
        "tool_end",
        "state_delta",   # tools
        "llm_start",
        "llm_end",
        "state_delta",   # agent
        "state_delta",   # verify (success)
        "run_finished",
    ]
    assert final_state["terminal_reason"] == "success"
    assert final_state["messages"][-1].content == "done"


def test_resume_agent_continues_from_last_durable_event(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    root = tmp_path / "runs"

    run_id, _ = run_agent(
        _scripted_tool_call_model(),
        task="inspect README",
        workspace=str(tmp_path),
        max_iterations=3,
        root=root,
    )

    # Simulate a crash after the tool result was durably logged but before
    # the agent's follow-up response was recorded: drop every event after
    # the "tools" node's state_delta.
    log_path = root / run_id / "events.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    tools_delta_index = next(i for i, line in enumerate(lines) if '"node": "tools"' in line)
    log_path.write_text("\n".join(lines[: tools_delta_index + 1]) + "\n", encoding="utf-8")

    resume_model = ScriptedToolCallingModel(messages=iter([AIMessage(content="done")]))
    final_state = resume_agent(run_id, resume_model, root=root)

    assert final_state["terminal_reason"] == "success"
    assert final_state["messages"][-1].content == "done"

    all_events = read_events(log_path)
    event_types = [event["type"] for event in all_events]
    assert event_types.count("run_started") == 1
    assert event_types[-1] == "run_finished"
