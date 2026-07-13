"""Tools for editing files within the active workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ..state import AgentState


def edit_workspace_file(
    path: str,
    old_text: str,
    new_text: str,
    workspace: str,
) -> str:
    """Replace one exact text match in an existing workspace file.

    Args:
        path: File path relative to the workspace.
        old_text: Existing text that must occur exactly once.
        new_text: Replacement text, which may be empty.
        workspace: Root directory that contains the file.

    Returns:
        A short message identifying the edited file.

    Raises:
        ValueError: If ``old_text`` is empty, missing, ambiguous, or the
            resolved path is outside the workspace.
        FileNotFoundError: If the target file does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    if not old_text:
        raise ValueError("old_text must not be empty")

    root = Path(workspace).resolve()
    target = (root / path).resolve()

    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path is outside workspace: {path}") from error

    content = target.read_text(encoding="utf-8")
    matches = content.count(old_text)
    if matches == 0:
        raise ValueError(f"old_text not found in {path}")
    if matches > 1:
        raise ValueError(f"old_text has multiple matches in {path}")

    target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


@tool
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    state: Annotated[AgentState, InjectedState],
) -> str:
    """Replace exactly one text match in an existing UTF-8 file.

    Use this only after reading the file. The edit fails if ``old_text`` is
    empty, absent, or appears more than once. The path must be relative to the
    current workspace.

    Args:
        path: Existing file path relative to the current workspace.
        old_text: Exact text expected to occur once.
        new_text: Replacement text, which may be empty to delete the match.

    Returns:
        A short message identifying the edited file.
    """
    return edit_workspace_file(path, old_text, new_text, state["workspace"])
