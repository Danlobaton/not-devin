from __future__ import annotations

from pathlib import Path

import pytest

from not_devin.event_log.reader import read_events


def test_returns_empty_list_for_missing_file(tmp_path: Path) -> None:
    assert read_events(tmp_path / "missing.jsonl") == []


def test_parses_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type": "a", "seq": 0}\n{"type": "b", "seq": 1}\n')

    events = read_events(path)

    assert [event["type"] for event in events] == ["a", "b"]


def test_tolerates_truncated_last_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type": "a", "seq": 0}\n{"type": "b", "trunc')

    events = read_events(path)

    assert [event["type"] for event in events] == ["a"]


def test_raises_on_corrupt_earlier_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type": "a", "trunc\n{"type": "b", "seq": 1}\n')

    with pytest.raises(ValueError, match="corrupt event log line 0"):
        read_events(path)
