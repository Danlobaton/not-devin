"""Tools for reading files within the active workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ..state import AgentState


def read_workspace_file(path: str, workspace: str) -> str:
    """Read a UTF-8 file after verifying it is inside the workspace.

    Args:
        path: File path relative to the workspace.
        workspace: Root directory that contains the file.

    Returns:
        The complete text content of the file.

    Raises:
        ValueError: If the resolved path is outside the workspace.
        FileNotFoundError: If the target file does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    root = Path(workspace).resolve()
    target = (root / path).resolve()

    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path is outside workspace: {path}") from error

    return target.read_text(encoding="utf-8")


@tool
def read_file(
    path: str,
    state: Annotated[AgentState, InjectedState],
) -> str:
    """Read the complete contents of a UTF-8 text file.

    Use a path relative to the current workspace. Paths outside the workspace
    are rejected.

    Args:
        path: File path relative to the current workspace.

    Returns:
        The complete text content of the file.
    """
    return read_workspace_file(path, state["workspace"])
