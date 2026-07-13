from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import Field

from not_devin.cli import cmd_inspect, cmd_resume, cmd_run


class ScriptedToolCallingModel(GenericFakeChatModel):
    bound_tools: list[BaseTool] = Field(default_factory=list)

    def bind_tools(
        self, tools: Sequence[BaseTool], *, tool_choice: str | None = None, **kwargs: Any
    ) -> "ScriptedToolCallingModel":
        self.bound_tools = list(tools)
        return self


def test_cmd_run_logs_and_returns_success(tmp_path: Path, capsys: Any) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    model = ScriptedToolCallingModel(messages=iter([AIMessage(content="done")]))
    root = tmp_path / "runs"

    exit_code = cmd_run(tmp_path, "inspect README", model=model, root=root)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "run_id:" in output
    assert "terminal_reason: success" in output


def test_cmd_inspect_renders_events(tmp_path: Path, capsys: Any) -> None:
    model = ScriptedToolCallingModel(messages=iter([AIMessage(content="done")]))
    root = tmp_path / "runs"
    cmd_run(tmp_path, "inspect README", model=model, root=root)
    run_id = next(root.iterdir()).name

    exit_code = cmd_inspect(run_id, root=root)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "run_started" in output
    assert "run_finished" in output


def test_cmd_inspect_missing_run_id_returns_error(tmp_path: Path, capsys: Any) -> None:
    exit_code = cmd_inspect("does-not-exist", root=tmp_path / "runs")

    assert exit_code == 1
    assert "no events found" in capsys.readouterr().out


def test_cmd_resume_continues_after_truncation(tmp_path: Path, capsys: Any) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    model = ScriptedToolCallingModel(
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
    root = tmp_path / "runs"
    cmd_run(tmp_path, "inspect README", model=model, root=root)
    run_id = next(root.iterdir()).name

    log_path = root / run_id / "events.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    tools_delta_index = next(i for i, line in enumerate(lines) if '"node": "tools"' in line)
    log_path.write_text("\n".join(lines[: tools_delta_index + 1]) + "\n", encoding="utf-8")

    resume_model = ScriptedToolCallingModel(messages=iter([AIMessage(content="done")]))
    exit_code = cmd_resume(run_id, model=resume_model, root=root)

    assert exit_code == 0
    assert "terminal_reason: success" in capsys.readouterr().out
