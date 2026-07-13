from __future__ import annotations

from not_devin.event_log.render import render_timeline


def test_renders_one_line_per_event_in_order() -> None:
    events = [
        {"seq": 0, "type": "run_started", "task": "fix bug", "workspace": "/tmp/ws"},
        {"seq": 1, "type": "tool_start", "tool_name": "read_file", "args": {"path": "a.py"}},
        {"seq": 2, "type": "tool_end", "result": "content", "latency_ms": 1.2},
        {"seq": 3, "type": "run_finished", "terminal_reason": "success", "iteration_count": 2},
    ]

    lines = render_timeline(events).splitlines()

    assert len(lines) == 4
    assert lines[0].startswith("[0] run_started")
    assert "task='fix bug'" in lines[0]
    assert lines[1].startswith("[1] tool_start")
    assert "read_file" in lines[1]
    assert lines[3].startswith("[3] run_finished")
    assert "success" in lines[3]


def test_renders_empty_log_as_empty_string() -> None:
    assert render_timeline([]) == ""


def test_run_started_line_surfaces_verify_command() -> None:
    events = [
        {
            "seq": 0,
            "type": "run_started",
            "task": "fix bug",
            "workspace": "/tmp/ws",
            "verify_command": "pytest -q",
        }
    ]

    assert "verify_command='pytest -q'" in render_timeline(events)
