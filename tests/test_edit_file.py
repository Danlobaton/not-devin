from pathlib import Path

import pytest

from not_devin.tools.edit_file import edit_workspace_file


def test_replaces_exactly_one_match(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("before\nold value\nafter\n", encoding="utf-8")

    result = edit_workspace_file(
        "example.py", "old value", "new value", str(tmp_path)
    )

    assert result == "Edited example.py"
    assert target.read_text(encoding="utf-8") == "before\nnew value\nafter\n"


@pytest.mark.parametrize(
    ("content", "old_text", "message"),
    [
        ("content", "", "must not be empty"),
        ("content", "missing", "not found"),
        ("same same", "same", "multiple matches"),
    ],
)
def test_rejects_invalid_replacement_without_modifying_file(
    tmp_path: Path,
    content: str,
    old_text: str,
    message: str,
) -> None:
    target = tmp_path / "example.py"
    target.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        edit_workspace_file("example.py", old_text, "replacement", str(tmp_path))

    assert target.read_text(encoding="utf-8") == content


def test_rejects_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="outside workspace"):
        edit_workspace_file("../outside.py", "old", "new", str(tmp_path))

    assert outside.read_text(encoding="utf-8") == "old"
