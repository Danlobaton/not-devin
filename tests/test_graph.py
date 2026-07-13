from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import Field

from not_devin.graph import build_graph
from not_devin.runtime.budget import BudgetExceeded
from not_devin.runtime.config import (
    BudgetConfig,
    ProviderConfig,
    RetryConfig,
    RuntimeConfig,
    TimeoutConfig,
)
from not_devin.runtime.invoker import (
    ProviderFailure,
    ReliableModelInvoker,
    RunDeadlineExceeded,
)


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
    assert [tool.name for tool in model.bound_tools] == ["read_file", "edit_file"]


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


def test_build_graph_accepts_explicit_tools(tmp_path: Path) -> None:
    @tool
    def sentinel() -> str:
        """Return a sentinel value."""
        return "sentinel"

    model = ScriptedToolCallingModel(
        messages=iter([AIMessage(content="done")])
    )

    build_graph(model, tools=[sentinel]).invoke(
        {
            "messages": [HumanMessage(content="Finish")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 1,
        }
    )

    assert [bound_tool.name for bound_tool in model.bound_tools] == [
        "sentinel"
    ]


def test_reliable_invoker_records_usage_in_state(tmp_path: Path) -> None:
    model = ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            ]
        )
    )
    config = RuntimeConfig(
        model=ProviderConfig(provider="openai", name="gpt"),
        timeouts=TimeoutConfig(),
        retry=RetryConfig(max_retries=0),
        budget=BudgetConfig(),
    )

    result = build_graph(
        model,
        invoker=ReliableModelInvoker(config, jitter=lambda: 0),
    ).invoke(
        {
            "messages": [HumanMessage(content="Finish")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 2,
        }
    )

    assert result["usage"]["successful_calls"] == 1
    assert result["usage"]["total_tokens"] == 15
    assert result["terminal_reason"] == "success"
    assert result["run_started_at"]
    assert result["run_deadline"]


def test_reliable_invoker_reuses_persisted_run_deadline(
    tmp_path: Path,
) -> None:
    class TimingInvoker:
        timing_created = False

        def new_run_timing(self):
            self.timing_created = True
            return ("new-start", "new-deadline")

        def invoke(self, model, messages, ledger, run_deadline=None):
            assert run_deadline == "persisted-deadline"
            return AIMessage(content="done")

    invoker = TimingInvoker()
    model = ScriptedToolCallingModel(
        messages=iter([AIMessage(content="unused")])
    )

    result = build_graph(model, invoker=invoker).invoke(
        {
            "messages": [HumanMessage(content="Finish")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 2,
            "run_started_at": "persisted-start",
            "run_deadline": "persisted-deadline",
        }
    )

    assert invoker.timing_created is False
    assert result["run_started_at"] == "persisted-start"
    assert result["run_deadline"] == "persisted-deadline"


@pytest.mark.parametrize(
    ("error", "terminal_reason"),
    [
        (RunDeadlineExceeded("deadline"), "timeout"),
        (
            ProviderFailure(retryable=False, status_code=400),
            "provider_failure",
        ),
        (
            BudgetExceeded("token_budget", "tokens"),
            "token_budget",
        ),
    ],
)
def test_reliable_invoker_errors_become_terminal_reasons(
    tmp_path: Path,
    error: Exception,
    terminal_reason: str,
) -> None:
    class RaisingInvoker:
        def new_run_timing(self):
            return (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:30:00+00:00",
            )

        def invoke(self, model, messages, ledger, run_deadline=None):
            raise error

    model = ScriptedToolCallingModel(
        messages=iter([AIMessage(content="unused")])
    )
    result = build_graph(model, invoker=RaisingInvoker()).invoke(
        {
            "messages": [HumanMessage(content="Finish")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 2,
        }
    )

    assert result["terminal_reason"] == terminal_reason


def test_guard_rejects_invalid_call_then_model_corrects(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    model = ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "no_such_tool", "args": {}, "id": "c1", "type": "tool_call"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "README.md"},
                            "id": "c2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    result = build_graph(model).invoke(
        {
            "messages": [HumanMessage(content="Inspect README.md")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 5,
        }
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert "unknown tool" in tool_messages[0].content
    assert tool_messages[1].content == "fixture content"
    assert result["invalid_strikes"] == 1
    assert result["terminal_reason"] == "success"


def test_repeated_identical_batches_terminate(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("fixture content")
    repeated_call = {
        "name": "read_file",
        "args": {"path": "README.md"},
        "id": "c1",
        "type": "tool_call",
    }
    model = ScriptedToolCallingModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{**repeated_call, "id": f"c{n}"}])
                for n in range(3)
            ]
        )
    )

    result = build_graph(model).invoke(
        {
            "messages": [HumanMessage(content="Inspect README.md")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 10,
        }
    )

    assert result["terminal_reason"] == "repeated_tool_calls"
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2  # third identical batch never executed


def test_verify_failure_feeds_back_then_success(tmp_path: Path) -> None:
    model = ScriptedToolCallingModel(
        messages=iter(
            [AIMessage(content="claiming done"), AIMessage(content="actually done")]
        )
    )

    result = build_graph(model).invoke(
        {
            "messages": [HumanMessage(content="Fix it")],
            "workspace": str(tmp_path),
            "iteration": 0,
            "max_iterations": 5,
            # Fails on first run (and creates the marker), passes on second.
            "verify_command": "test -f fixed.txt || { touch fixed.txt; exit 1; }",
        }
    )

    assert result["terminal_reason"] == "success"
    feedback = [
        message
        for message in result["messages"]
        if isinstance(message, HumanMessage)
        and "Verification failed" in message.content
    ]
    assert len(feedback) == 1
