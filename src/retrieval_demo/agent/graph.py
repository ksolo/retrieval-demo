from functools import cache

from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.constants import START, END

from .agent_state import AgentState
from .nodes import retrieval_node, model_node


@cache
def get_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("model_node", model_node)

    graph.add_edge(START, "retrieval_node")
    graph.add_edge("retrieval_node", "model_node")
    graph.add_edge("model_node", END)

    return graph.compile()
