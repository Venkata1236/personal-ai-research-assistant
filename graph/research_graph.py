# graph/research_graph.py
# ─────────────────────────────────────────────────────────────────
# Builds and compiles the LangGraph StateGraph.
# The graph defines the ORDER and ROUTING of all nodes.
#
# Flow: START → plan → research → write → critic → END
# ─────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import plan_node, research_node, write_node, critic_node


def build_graph():
    """
    Construct the full research pipeline as a LangGraph graph.
    Returns a compiled graph ready to invoke.
    """

    # Initialize the graph with our shared state schema
    graph = StateGraph(AgentState)

    # ── Add Nodes ────────────────────────────────────────────────
    # Each node is a Python function from nodes.py
    graph.add_node("plan",     plan_node)
    graph.add_node("research", research_node)
    graph.add_node("write",    write_node)
    graph.add_node("critic",   critic_node)

    # ── Define Edges (execution order) ───────────────────────────
    # set_entry_point defines where the graph starts
    graph.set_entry_point("plan")

    # Sequential edges — each node flows into the next
    graph.add_edge("plan",     "research")
    graph.add_edge("research", "write")
    graph.add_edge("write",    "critic")

    # Terminal edge — critic is the last node
    graph.add_edge("critic", END)

    # ── Compile ───────────────────────────────────────────────────
    # compile() validates the graph and returns an executable object
    return graph.compile()


# Module-level graph instance — import this in crew.py and main.py
research_graph = build_graph()