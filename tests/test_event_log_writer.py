from __future__ import annotations

import json
from pathlib import Path

from not_devin.event_log.reader import read_events
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


def test_reopening_after_mid_write_crash_drops_partial_line(tmp_path: Path) -> None:
    first_writer = EventLogWriter("run-1", root=tmp_path)
    first_writer.write("run_started", task="fix bug")

    # Simulate a genuine mid-write crash: a newline-less partial JSON
    # fragment left dangling at the end of the file. writer.write() always
    # writes complete lines, so we hand-corrupt the file directly, the way
    # a real crash would.
    with first_writer.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "state_delta", "node": "tool')

    second_writer = EventLogWriter("run-1", root=tmp_path)
    second_writer.write("run_finished", terminal_reason="success")

    events = read_events(second_writer.path)
    assert [event["type"] for event in events] == ["run_started", "run_finished"]
