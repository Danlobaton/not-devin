from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ..state import AgentState


def read_workspace_file(path: str, workspace: str) -> str:
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
    """Read a UTF-8 text file relative to the current workspace."""
    return read_workspace_file(path, state["workspace"])
