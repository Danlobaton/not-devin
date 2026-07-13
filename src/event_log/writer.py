"""Append-only writer for per-run event logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reader import read_events

DEFAULT_ROOT = Path(".not-devin/runs")


class EventLogWriter:
    """Appends structured events to one run's events.jsonl file."""

    def __init__(self, run_id: str, root: Path | str = DEFAULT_ROOT) -> None:
        self.run_id = run_id
        self._dir = Path(root) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / "events.jsonl"
        self._seq = len(read_events(self.path))

    def write(self, event_type: str, **fields: Any) -> dict:
        """Append one event and return the full event dict written."""
        event = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        self._seq += 1
        return event
