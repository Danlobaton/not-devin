from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .tools import TOOLS


def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    bound_model = model.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        iteration = state.get("iteration", 0)
        if iteration >= state["max_iterations"]:
            return {"terminal_reason": "iteration_limit"}

        response = bound_model.invoke(state["messages"])
        next_iteration = iteration + 1
        terminal_reason = None

        if not response.tool_calls:
            terminal_reason = "success"
        elif next_iteration >= state["max_iterations"]:
            terminal_reason = "iteration_limit"

        return {
            "messages": [response],
            "iteration": next_iteration,
            "terminal_reason": terminal_reason,
        }

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        if state.get("terminal_reason") is not None:
            return END

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()
