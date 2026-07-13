from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import Field

from not_devin.graph import build_graph


class ScriptedToolCallingModel(GenericFakeChatModel):
    bound_tools: list[BaseTool] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedToolCallingModel":
        self.bound_tools = list(tools)
        return self


def test_runs_model_tool_model_cycle(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    model = ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "README.md"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="README inspected"),
            ]
        )
    )

    result = build_graph(model).invoke(
        {
            "messages": [HumanMessage(content="Inspect README.md")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 3,
        }
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert tool_messages[0].content == "fixture content"
    assert result["messages"][-1].content == "README inspected"
    assert result["iteration"] == 2
    assert result["terminal_reason"] == "success"
    assert [tool.name for tool in model.bound_tools] == ["read_file"]


def test_stops_before_tool_execution_at_iteration_limit(tmp_path: Path) -> None:
    model = ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "README.md"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
    )

    result = build_graph(model).invoke(
        {
            "messages": [HumanMessage(content="Inspect README.md")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 1,
        }
    )

    assert not any(
        isinstance(message, ToolMessage) for message in result["messages"]
    )
    assert result["iteration"] == 1
    assert result["terminal_reason"] == "iteration_limit"
