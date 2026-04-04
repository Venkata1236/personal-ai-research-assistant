# main.py
# ─────────────────────────────────────────────────────────────────
# Entry point of the entire pipeline.
# Two modes:
#   1. Direct CrewAI mode  — runs the crew sequentially
#   2. LangGraph mode      — runs through the stateful graph
# ─────────────────────────────────────────────────────────────────

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from crew import ResearchCrew
from graph.research_graph import research_graph
from memory.conversation_memory import ResearchMemory

# Rich console for pretty terminal output
console = Console()


def run_with_crew(topic: str) -> str:
    """
    Mode 1: Direct CrewAI pipeline.
    Runs: Planner → Researcher → Writer → Critic
    Best for: simple, single-topic research.
    """
    console.print(Panel(f"[bold blue]Research Topic:[/] {topic}"))

    # Initialize memory — check if we've researched this before
    memory = ResearchMemory()
    cached = memory.get_past_research(topic)

    if cached:
        console.print("[yellow]Found cached research — using memory.[/yellow]")
        return cached

    # Run the full crew pipeline
    result = ResearchCrew().crew().kickoff(inputs={"topic": topic})

    # Save result to long-term memory for future runs
    memory.save_research(topic, str(result))

    return str(result)


def run_with_graph(topic: str) -> str:
    """
    Mode 2: LangGraph pipeline.
    Same agents but orchestrated through a stateful graph.
    Best for: complex topics needing conditional routing.
    """
    console.print(Panel(f"[bold green]LangGraph Mode:[/] {topic}"))

    # Build the initial state — only topic is known at the start
    initial_state = {
        "topic":        topic,
        "plan":         None,
        "research":     None,
        "report":       None,
        "final_output": None,
        "error":        None,
    }

    # invoke() runs the graph — returns the final state dict
    final_state = research_graph.invoke(initial_state)

    # final_output is set by the critic node (last in the graph)
    return final_state.get("final_output", "No output generated.")


if __name__ == "__main__":
    # ── Change topic here to run different research ───────────────
    TOPIC = "Agentic AI frameworks in 2025"

    # ── Choose your mode ──────────────────────────────────────────
    # result = run_with_crew(TOPIC)     # Direct CrewAI
    result = run_with_graph(TOPIC)      # LangGraph orchestrated

    # Print the final report with Markdown formatting
    console.print("\n")
    console.print(Panel(
        Markdown(result),
        title="[bold]Final Research Report[/bold]",
        border_style="green"
    ))