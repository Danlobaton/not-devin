from __future__ import annotations

import re

from not_devin.event_log.ids import new_run_id


def test_new_run_id_is_sortable_and_unique() -> None:
    first = new_run_id()
    second = new_run_id()

    assert re.match(r"^\d{8}T\d{6}-[0-9a-f]{8}$", first)
    assert first != second
