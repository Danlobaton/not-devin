from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from not_devin.event_log.handler import EventLogHandler


class FakeWriter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event_type: str, **fields: object) -> dict:
        self.events.append((event_type, fields))
        return {"type": event_type, **fields}


def test_logs_llm_start_and_end() -> None:
    writer = FakeWriter()
    handler = EventLogHandler(writer)
    run_id = uuid4()

    handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="hello"))]])
    handler.on_llm_end(response, run_id=run_id)

    event_types = [event_type for event_type, _ in writer.events]
    assert event_types == ["llm_start", "llm_end"]
    assert writer.events[0][1]["messages"][0]["data"]["content"] == "hi"
    assert writer.events[1][1]["response"]["data"]["content"] == "hello"
    assert writer.events[1][1]["latency_ms"] >= 0


def test_logs_llm_error() -> None:
    writer = FakeWriter()
    handler = EventLogHandler(writer)
    run_id = uuid4()

    handler.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id=run_id)
    handler.on_llm_error(RuntimeError("provider down"), run_id=run_id)

    assert writer.events[-1][0] == "llm_end"
    assert writer.events[-1][1]["error"] == "provider down"


def test_logs_tool_start_and_end() -> None:
    writer = FakeWriter()
    handler = EventLogHandler(writer)
    run_id = uuid4()

    handler.on_tool_start(
        {"name": "read_file"},
        '{"path": "README.md"}',
        run_id=run_id,
        inputs={"path": "README.md"},
    )
    output = ToolMessage(content="fixture content", tool_call_id="call_1")
    handler.on_tool_end(output, run_id=run_id)

    event_types = [event_type for event_type, _ in writer.events]
    assert event_types == ["tool_start", "tool_end"]
    assert writer.events[0][1] == {"tool_name": "read_file", "args": {"path": "README.md"}}
    assert writer.events[1][1]["tool_call_id"] == "call_1"
    assert writer.events[1][1]["result"] == "fixture content"


def test_logs_tool_error() -> None:
    writer = FakeWriter()
    handler = EventLogHandler(writer)
    run_id = uuid4()

    handler.on_tool_start({"name": "read_file"}, "{}", run_id=run_id, inputs={})
    handler.on_tool_error(RuntimeError("boom"), run_id=run_id)

    assert writer.events[-1][0] == "tool_end"
    assert writer.events[-1][1]["error"] == "boom"
