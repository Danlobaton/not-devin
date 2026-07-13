from __future__ import annotations

import json
from pathlib import Path

from not_devin.event_log.writer import EventLogWriter


def test_write_appends_jsonl_with_monotonic_seq(tmp_path: Path) -> None:
    writer = EventLogWriter("run-1", root=tmp_path)

    first = writer.write("run_started", task="fix bug")
    second = writer.write("run_finished", terminal_reason="success")

    assert first["seq"] == 0
    assert second["seq"] == 1
    assert first["run_id"] == "run-1"
    assert first["type"] == "run_started"
    assert "ts" in first

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["task"] == "fix bug"
    assert json.loads(lines[1])["terminal_reason"] == "success"


def test_reopening_continues_seq_after_existing_events(tmp_path: Path) -> None:
    first_writer = EventLogWriter("run-1", root=tmp_path)
    first_writer.write("run_started", task="fix bug")

    second_writer = EventLogWriter("run-1", root=tmp_path)
    event = second_writer.write("run_finished", terminal_reason="success")

    assert event["seq"] == 1
