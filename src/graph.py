from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from .guard import build_guard_node
from .runtime.budget import BudgetExceeded, UsageLedger
from .runtime.invoker import (
    InvokableModel,
    ProviderFailure,
    RunDeadlineExceeded,
    RuntimeModelInvoker,
)
from .state import AgentState
from .tools import TOOLS
from .verify import build_verify_node


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    invoker: RuntimeModelInvoker | None = None,
) -> CompiledStateGraph:
    active_tools = list(TOOLS if tools is None else tools)
    bound_model = model.bind_tools(active_tools)

    def agent_node(state: AgentState) -> dict:
        iteration = state.get("iteration", 0)
        if iteration >= state["max_iterations"]:
            return {"terminal_reason": "iteration_limit"}

        ledger = UsageLedger.from_state(state.get("usage"))
        runtime_updates = {}
        if invoker is None:
            response = bound_model.invoke(state["messages"])
        else:
            run_started_at = state.get("run_started_at")
            run_deadline = state.get("run_deadline")
            if run_started_at is None or run_deadline is None:
                run_started_at, run_deadline = invoker.new_run_timing()
            runtime_updates = {
                "run_started_at": run_started_at,
                "run_deadline": run_deadline,
            }
            try:
                response = invoker.invoke(
                    cast(InvokableModel, bound_model),
                    state["messages"],
                    ledger,
                    run_deadline,
                )
            except BudgetExceeded as error:
                return {
                    **runtime_updates,
                    "usage": ledger.to_state(),
                    "terminal_reason": error.terminal_reason,
                }
            except RunDeadlineExceeded:
                return {
                    **runtime_updates,
                    "usage": ledger.to_state(),
                    "terminal_reason": "timeout",
                }
            except ProviderFailure as error:
                return {
                    **runtime_updates,
                    "usage": ledger.to_state(),
                    "terminal_reason": "provider_failure",
                    "last_provider_error": str(error),
                }

        next_iteration = iteration + 1
        terminal_reason = None

        if response.tool_calls and next_iteration >= state["max_iterations"]:
            terminal_reason = "iteration_limit"

        return {
            **runtime_updates,
            "messages": [response],
            "iteration": next_iteration,
            "terminal_reason": terminal_reason,
            **({"usage": ledger.to_state()} if invoker is not None else {}),
        }

    def route_after_agent(
        state: AgentState,
    ) -> Literal["guard", "verify", "__end__"]:
        if state.get("terminal_reason") is not None:
            return END
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "guard"
        return "verify"

    def route_after_guard(
        state: AgentState,
    ) -> Literal["tools", "agent", "__end__"]:
        if state.get("terminal_reason") is not None:
            return END
        if isinstance(state["messages"][-1], ToolMessage):
            return "agent"
        return "tools"

    def route_after_verify(state: AgentState) -> Literal["agent", "__end__"]:
        if state.get("terminal_reason") is not None:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("guard", build_guard_node(active_tools))
    graph.add_node("tools", ToolNode(active_tools))
    graph.add_node("verify", build_verify_node())
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["guard", "verify", END])
    graph.add_conditional_edges("guard", route_after_guard, ["tools", "agent", END])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", route_after_verify, ["agent", END])
    return graph.compile()
