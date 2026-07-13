from langchain_core.tools import BaseTool

from .read_file import read_file

TOOLS: list[BaseTool] = [read_file]

__all__ = ["TOOLS", "read_file"]
