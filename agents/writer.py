# agents/writer.py
# ─────────────────────────────────────────────────────────────────
# The Writer agent receives the Researcher's raw findings
# and produces a clean, structured, professional report.
#
# WHY a dedicated Writer?
#   Researchers are great at collecting facts but tend to dump
#   everything without structure. The Writer's sole job is to
#   organise, format, and make the content readable.
#
# The Writer has NO tools — it only reads context and writes.
# ─────────────────────────────────────────────────────────────────

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME
from config.prompts import WRITER_SYSTEM


class WriterAgent:
    """
    Wraps the CrewAI Agent + Task for the Writer role.
    No tools — pure LLM writing based on research context.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            # Higher temperature = more natural, fluent writing
            # 0.5-0.7 is the sweet spot for structured reports
            temperature=0.6
        )

    def build_agent(self, config: dict) -> Agent:
        """
        Build the Writer Agent.
        No tools needed — the agent only needs the LLM to write.
        """
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            system_template=WRITER_SYSTEM,

            tools=[],   # Writer doesn't search — it writes

            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def build_task(self, config: dict, agent: Agent, context_tasks: list) -> Task:
        """
        Build the Write Task.

        The writer receives research_task output as context,
        then produces a formatted 3-section report:
          → Introduction
          → Key Findings (bullet points)
          → Conclusion
        """
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=agent,

            # Writer receives the researcher's output as input
            context=context_tasks,

            output_file="outputs/report_draft.md"
        )