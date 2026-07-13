from pathlib import Path

import pytest

from not_devin.tools.read_file import read_workspace_file


def test_reads_file_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "README.md"
    target.write_text("fixture content")

    assert read_workspace_file("README.md", str(tmp_path)) == "fixture content"


def test_rejects_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")

    with pytest.raises(ValueError, match="outside workspace"):
        read_workspace_file("../secret.txt", str(tmp_path))
