# graph/nodes.py
# ─────────────────────────────────────────────────────────────────
# Each function here is one NODE in the LangGraph StateGraph.
# A node receives the current AgentState, does its work,
# and returns a dict of updated state fields.
# LangGraph merges the returned dict back into the state.
# ─────────────────────────────────────────────────────────────────

from graph.state import AgentState
from crew import ResearchCrew


def plan_node(state: AgentState) -> dict:
    """
    Node 1: Planner
    Input : state["topic"]
    Output: state["plan"] — list of sub-questions
    """
    print(f"\n[Node: Planner] Planning research for: {state['topic']}")

    # Kick off only the planner task from the crew
    crew = ResearchCrew()
    plan = crew.run_task("plan_task", inputs={"topic": state["topic"]})

    # Return only the fields we want to update
    return {"plan": plan}


def research_node(state: AgentState) -> dict:
    """
    Node 2: Researcher
    Input : state["topic"] + state["plan"]
    Output: state["research"] — raw findings + sources
    """
    print(f"\n[Node: Researcher] Researching: {state['topic']}")

    crew = ResearchCrew()
    research = crew.run_task(
        "research_task",
        inputs={"topic": state["topic"], "plan": state["plan"]}
    )

    return {"research": research}


def write_node(state: AgentState) -> dict:
    """
    Node 3: Writer
    Input : state["research"]
    Output: state["report"] — structured draft report
    """
    print(f"\n[Node: Writer] Writing report...")

    crew = ResearchCrew()
    report = crew.run_task(
        "write_task",
        inputs={"topic": state["topic"], "research": state["research"]}
    )

    return {"report": report}


def critic_node(state: AgentState) -> dict:
    """
    Node 4: Critic
    Input : state["report"]
    Output: state["final_output"] — reviewed, polished report
    """
    print(f"\n[Node: Critic] Reviewing report...")

    crew = ResearchCrew()
    final = crew.run_task(
        "critic_task",
        inputs={"topic": state["topic"], "report": state["report"]}
    )

    return {"final_output": final}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — LangGraph calls this to decide
    which node to go to next.

    Returns "end"   if there's an error or final output is ready
    Returns "write" if research is done but report isn't written yet
    """
    if state.get("error"):
        return "end"       # Something went wrong — stop the graph
    if state.get("research") and not state.get("report"):
        return "write"     # Research done — go write the report
    return "end"