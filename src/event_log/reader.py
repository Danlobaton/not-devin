"""Reads and parses an append-only run event log."""

from __future__ import annotations

import json
from pathlib import Path


def read_events(path: Path) -> list[dict]:
    """Parse events.jsonl into a list of event dicts.

    Tolerates a truncated final line (a crash mid-write); corruption on
    an earlier line is a real error and raises.
    """
    path = Path(path)
    if not path.exists():
        return []

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events: list[dict] = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                break
            raise ValueError(f"corrupt event log line {index} in {path}") from error
    return events
