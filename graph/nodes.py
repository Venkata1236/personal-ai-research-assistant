# graph/nodes.py
# ─────────────────────────────────────────────────────────────────
# Fix: CrewOutput object must be converted to str before
# passing between LangGraph nodes. CrewAI only accepts
# str, int, float, bool, dict, list as interpolation inputs.
# ─────────────────────────────────────────────────────────────────

from graph.state import AgentState
from crew import ResearchCrew


def plan_node(state: AgentState) -> dict:
    """Node 1 — Planner: breaks topic into sub-questions."""
    print(f"\n[Node: Planner] Planning for: {state['topic']}")

    crew = ResearchCrew()
    result = crew.run_task("plan_task", inputs={"topic": state["topic"]})

    # ← Convert CrewOutput → str before storing in state
    return {"plan": str(result)}


def research_node(state: AgentState) -> dict:
    """Node 2 — Researcher: web search + RAG retrieval."""
    print(f"\n[Node: Researcher] Researching: {state['topic']}")

    crew = ResearchCrew()
    result = crew.run_task(
        "research_task",
        inputs={
            "topic": state["topic"],
            "plan":  str(state.get("plan", ""))   # ← ensure str
        }
    )

    return {"research": str(result)}              # ← Convert to str


def write_node(state: AgentState) -> dict:
    """Node 3 — Writer: structures findings into report."""
    print(f"\n[Node: Writer] Writing report...")

    crew = ResearchCrew()
    result = crew.run_task(
        "write_task",
        inputs={
            "topic":    state["topic"],
            "research": str(state.get("research", ""))  # ← ensure str
        }
    )

    return {"report": str(result)}                # ← Convert to str


def critic_node(state: AgentState) -> dict:
    """Node 4 — Critic: reviews and polishes final report."""
    print(f"\n[Node: Critic] Reviewing report...")

    crew = ResearchCrew()
    result = crew.run_task(
        "critic_task",
        inputs={
            "topic":  state["topic"],
            "report": str(state.get("report", ""))      # ← ensure str
        }
    )

    return {"final_output": str(result)}          # ← Convert to str


def should_continue(state: AgentState) -> str:
    """Conditional edge — decides next node."""
    if state.get("error"):
        return "end"
    if state.get("research") and not state.get("report"):
        return "write"
    return "end"