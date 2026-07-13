"""Run identifier generation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_run_id() -> str:
    """Generate a unique, lexicographically sortable run identifier.

    Returns:
        A string of the form ``YYYYMMDDTHHMMSS-{8 hex chars}``, combining a
        UTC timestamp with a random suffix so identifiers sort chronologically
        and are unique across concurrent runs.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"
