from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from not_devin.verify import MAX_VERIFY_OUTPUT_CHARS, build_verify_node


def make_state(workspace: Path, verify_command: str | None = None) -> dict:
    return {
        "messages": [HumanMessage(content="task")],
        "workspace": str(workspace),
        "iteration": 1,
        "max_iterations": 5,
        "verify_command": verify_command,
    }


def test_no_command_is_success(tmp_path: Path) -> None:
    assert build_verify_node()(make_state(tmp_path)) == {
        "terminal_reason": "success"
    }


def test_passing_command_is_success(tmp_path: Path) -> None:
    assert build_verify_node()(make_state(tmp_path, "exit 0")) == {
        "terminal_reason": "success"
    }


def test_failing_command_feeds_back_output(tmp_path: Path) -> None:
    update = build_verify_node()(
        make_state(tmp_path, "echo broken output; exit 3")
    )

    assert "terminal_reason" not in update
    message = update["messages"][0]
    assert isinstance(message, HumanMessage)
    assert "exit 3" in message.content
    assert "broken output" in message.content


def test_failure_output_is_tail_truncated(tmp_path: Path) -> None:
    command = (
        "python3 -c \"print('x' * 20000); print('THE_END'); raise SystemExit(1)\""
    )

    update = build_verify_node()(make_state(tmp_path, command))

    content = update["messages"][0].content
    assert "THE_END" in content
    assert len(content) < MAX_VERIFY_OUTPUT_CHARS + 200


def test_command_runs_in_workspace(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("present")

    assert build_verify_node()(make_state(tmp_path, "test -f marker.txt")) == {
        "terminal_reason": "success"
    }


def test_timeout_feeds_back(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    update = build_verify_node()(make_state(tmp_path, "sleep 999"))

    assert "terminal_reason" not in update
    assert "timed out" in update["messages"][0].content
