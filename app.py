# app.py
# ─────────────────────────────────────────────────────────────────
# Streamlit UI for the Personal AI Research Assistant.
# Two modes available from the sidebar:
#   1. CrewAI Direct   — sequential 4-agent pipeline
#   2. LangGraph Mode  — stateful graph orchestration
#
# Run locally:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import time
from pathlib import Path

# ── Page Config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Personal AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container { padding-top: 2rem; }

    /* Report output card */
    .report-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }

    /* Agent step badges */
    .step-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        margin: 4px 2px;
    }

    /* Sidebar styling */
    .sidebar-info {
        background: #e8f4f8;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 13px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/research.png", width=60)
    st.title("Research Assistant")
    st.markdown("---")

    # Mode selector
    st.subheader("Pipeline Mode")
    mode = st.radio(
        label="Choose mode",
        options=["CrewAI Direct", "LangGraph"],
        help=(
            "CrewAI Direct: Runs 4 agents sequentially.\n"
            "LangGraph: Stateful graph with conditional routing."
        )
    )

    # Mode description
    if mode == "CrewAI Direct":
        st.markdown("""
        <div class='sidebar-info'>
        <b>CrewAI Direct</b><br>
        Planner → Researcher → Writer → Critic<br>
        Best for: straightforward research topics.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='sidebar-info'>
        <b>LangGraph Mode</b><br>
        Stateful graph with conditional routing.<br>
        Best for: complex, multi-step research.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Document upload for RAG
    st.subheader("Upload Documents (RAG)")
    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Uploaded docs are added to the RAG knowledge base."
    )

    # Save uploaded files to data/raw/ for RAG ingestion
    if uploaded_files:
        raw_path = Path("data/raw")
        raw_path.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            save_path = raw_path / f.name
            with open(save_path, "wb") as out:
                out.write(f.read())
        st.success(f"Saved {len(uploaded_files)} file(s) to knowledge base.")

    st.markdown("---")
    st.caption("Built with CrewAI · LangGraph · RAG")


# ── Main Content ──────────────────────────────────────────────────
st.title("🔬 Personal AI Research Assistant")
st.markdown("*Powered by CrewAI + LangGraph + RAG — full agentic pipeline*")
st.markdown("---")

# Topic input
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        label="Research Topic",
        placeholder="e.g. Agentic AI frameworks in 2025",
        help="Enter any topic — the agents will plan, research, write and review."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Research", type="primary", use_container_width=True)


# ── Pipeline Status Display ───────────────────────────────────────
def show_pipeline_steps(mode: str):
    """Show the 4-step agent pipeline visually while running."""
    steps = ["Planner", "Researcher", "Writer", "Critic"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    cols = st.columns(4)
    placeholders = []
    for i, (col, step, color) in enumerate(zip(cols, steps, colors)):
        with col:
            ph = st.empty()
            # Start all as grey (waiting)
            ph.markdown(
                f"<div style='text-align:center; padding:12px; "
                f"border-radius:8px; background:#f0f0f0; color:#888;'>"
                f"⏳ {step}</div>",
                unsafe_allow_html=True
            )
            placeholders.append((ph, step, color))

    return placeholders


def activate_step(placeholders, index: int):
    """Light up one step as active (blue pulse)."""
    ph, step, color = placeholders[index]
    ph.markdown(
        f"<div style='text-align:center; padding:12px; "
        f"border-radius:8px; background:{color}22; "
        f"border:2px solid {color}; color:{color}; font-weight:600;'>"
        f"⚡ {step}</div>",
        unsafe_allow_html=True
    )


def complete_step(placeholders, index: int):
    """Mark a step as done (green check)."""
    ph, step, color = placeholders[index]
    ph.markdown(
        f"<div style='text-align:center; padding:12px; "
        f"border-radius:8px; background:#E8F5E9; "
        f"border:2px solid #4CAF50; color:#2E7D32; font-weight:600;'>"
        f"✅ {step}</div>",
        unsafe_allow_html=True
    )


# ── Run Button Logic ──────────────────────────────────────────────
if run_btn and topic.strip():

    # Show pipeline step tracker
    st.markdown("### Pipeline Progress")
    placeholders = show_pipeline_steps(mode)

    # Log output area
    st.markdown("### Agent Logs")
    log_area = st.empty()
    logs = []

    def log(msg: str):
        """Append a message to the live log area."""
        logs.append(msg)
        log_area.markdown(
            "\n".join(f"- {l}" for l in logs[-10:])  # Show last 10 logs
        )

    # Result placeholder
    result_placeholder = st.empty()

    try:
        # ── Import pipeline functions ─────────────────────────────
        # Imported here (not at top) so Streamlit loads fast
        # even before the heavy ML libraries are needed
        from crew import ResearchCrew
        from graph.research_graph import research_graph
        from memory.conversation_memory import ResearchMemory

        memory = ResearchMemory()

        # ── Check memory cache first ──────────────────────────────
        cached = memory.get_past_research(topic)
        if cached:
            st.info("Found this topic in memory — showing cached result.")
            result_placeholder.markdown(
                f"<div class='report-card'>{cached}</div>",
                unsafe_allow_html=True
            )
            st.stop()

        # ── Run chosen pipeline ───────────────────────────────────
        if mode == "CrewAI Direct":

            # Step 1 — Planner
            activate_step(placeholders, 0)
            log("Planner: Breaking topic into sub-questions...")
            time.sleep(0.5)   # Visual delay so user sees the step activate

            # Step 2 — Researcher
            activate_step(placeholders, 1)
            log("Researcher: Searching web + RAG knowledge base...")

            # Step 3 — Writer
            activate_step(placeholders, 2)
            log("Writer: Structuring findings into report...")

            # Step 4 — Critic
            activate_step(placeholders, 3)
            log("Critic: Reviewing and polishing final report...")

            # Kick off the full crew pipeline
            # kickoff() blocks until all 4 agents complete
            result = ResearchCrew().crew().kickoff(
                inputs={"topic": topic}
            )
            result_str = str(result)

        else:  # LangGraph mode

            activate_step(placeholders, 0)
            log("LangGraph: Initialising state graph...")

            # Build initial state — all fields except topic start as None
            initial_state = {
                "topic":        topic,
                "plan":         None,
                "research":     None,
                "report":       None,
                "final_output": None,
                "error":        None,
            }

            activate_step(placeholders, 1)
            log("LangGraph: Running plan → research → write → critic nodes...")

            # invoke() runs through all graph nodes and returns final state
            final_state = research_graph.invoke(initial_state)

            activate_step(placeholders, 2)
            log("LangGraph: Writer node complete...")

            activate_step(placeholders, 3)
            log("LangGraph: Critic node complete — finalising output...")

            result_str = final_state.get("final_output", "No output generated.")

        # ── Mark all steps complete ───────────────────────────────
        for i in range(4):
            complete_step(placeholders, i)

        log("Pipeline complete.")

        # ── Save to memory ────────────────────────────────────────
        memory.save_research(topic, result_str)

        # ── Display final report ──────────────────────────────────
        st.markdown("---")
        st.markdown("### Final Research Report")
        st.markdown(result_str)   # Renders markdown formatting

        # ── Download button ───────────────────────────────────────
        st.download_button(
            label="Download Report as Markdown",
            data=result_str,
            file_name=f"research_{topic[:30].replace(' ', '_')}.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.exception(e)   # Shows full traceback in UI for debugging

elif run_btn and not topic.strip():
    # User clicked Run without entering a topic
    st.warning("Please enter a research topic first.")


# ── Past Research Section ─────────────────────────────────────────
st.markdown("---")
st.markdown("### Past Research (Memory)")

memory_file = Path("memory/long_term_memory.json")
if memory_file.exists():
    import json
    with open(memory_file) as f:
        past = json.load(f)

    if past:
        # Show each past topic as an expandable section
        for saved_topic, saved_result in past.items():
            with st.expander(f"📄 {saved_topic}"):
                st.markdown(saved_result)
    else:
        st.caption("No past research yet. Run your first query above.")
else:
    st.caption("No past research yet. Run your first query above.")