from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state import AgentState


def agent_node(state: AgentState) -> dict:
    """Model turn — bind tools, call LLM, append response.

    Intentionally empty scaffolding for interview prep.
    """
    return {"iteration": state.get("iteration", 0) + 1}


def build_graph() -> CompiledStateGraph:
    """Compile the agent loop graph.

    Planned shape:
      START → agent ⇄ tools → END
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()
