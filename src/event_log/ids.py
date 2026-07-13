"""Run identifier generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_run_id() -> str:
    """Return a unique, lexicographically sortable run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"
