"""Runs the agent graph end-to-end while logging every event to a durable log."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.tools import BaseTool

from .event_log.handler import EventLogHandler
from .event_log.ids import new_run_id
from .event_log.reader import read_events
from .event_log.replay import apply_delta, rebuild_state
from .event_log.writer import DEFAULT_ROOT, EventLogWriter
from .graph import build_graph
from .state import AgentState


def run_agent(
    model: BaseChatModel,
    task: str,
    workspace: str,
    max_iterations: int,
    tools: Sequence[BaseTool] | None = None,
    root: Path | str = DEFAULT_ROOT,
) -> tuple[str, AgentState]:
    """Run the agent on a fresh task, logging every event.

    Returns:
        A tuple of (run_id, final AgentState).
    """
    run_id = new_run_id()
    writer = EventLogWriter(run_id, root=root)
    writer.write("run_started", task=task, workspace=workspace, max_iterations=max_iterations)

    state: AgentState = {
        "messages": [HumanMessage(content=task)],
        "workspace": workspace,
        "iteration": 0,
        "max_iterations": max_iterations,
        "terminal_reason": None,
    }

    final_state = _stream_and_log(model, state, tools, writer)
    return run_id, final_state


def resume_agent(
    run_id: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    root: Path | str = DEFAULT_ROOT,
) -> AgentState:
    """Resume a run from its last durable event, appending new events onward.

    Resume always re-enters at the agent node. A crash between the agent
    step completing and the tool step running will cause the model to be
    re-invoked with an unanswered tool call still in `messages` — this is a
    known limitation of resuming without a full graph checkpointer.
    """
    writer = EventLogWriter(run_id, root=root)
    events = read_events(writer.path)
    state = rebuild_state(events)
    return _stream_and_log(model, state, tools, writer)


def _stream_and_log(
    model: BaseChatModel,
    state: AgentState,
    tools: Sequence[BaseTool] | None,
    writer: EventLogWriter,
) -> AgentState:
    graph = build_graph(model, tools=tools)
    handler = EventLogHandler(writer)

    final_state = state
    for update in graph.stream(state, config={"callbacks": [handler]}, stream_mode="updates"):
        for node, delta in update.items():
            serialized_delta = dict(delta)
            if "messages" in serialized_delta:
                serialized_delta["messages"] = messages_to_dict(serialized_delta["messages"])
            writer.write("state_delta", node=node, delta=serialized_delta)
            final_state = apply_delta(final_state, delta)

    writer.write(
        "run_finished",
        terminal_reason=final_state.get("terminal_reason"),
        iteration_count=final_state.get("iteration"),
    )
    return final_state
