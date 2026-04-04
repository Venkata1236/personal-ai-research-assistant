# agents/critic.py
# ─────────────────────────────────────────────────────────────────
# The Critic agent is the final quality gate.
# It receives the Writer's draft and performs:
#   1. Fact-checking — are all claims supported by sources?
#   2. Completeness  — are there gaps in the research?
#   3. Clarity       — is the writing clear and well-structured?
#   4. Output        — returns the polished, final version
#
# WHY a Critic?
#   Without review, AI-generated reports can contain confident
#   but unsupported claims. The Critic enforces quality standards.
# ─────────────────────────────────────────────────────────────────

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME
from config.prompts import CRITIC_SYSTEM


class CriticAgent:
    """
    Wraps the CrewAI Agent + Task for the Critic/QA role.
    Last agent in the pipeline — produces the final output.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            # Low temperature = careful, analytical review
            # We want the critic to be precise, not creative
            temperature=0.1
        )

    def build_agent(self, config: dict) -> Agent:
        """
        Build the Critic Agent.
        Like the Writer, the Critic uses no tools —
        it reviews content already in its context window.
        """
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            system_template=CRITIC_SYSTEM,

            tools=[],   # Critic reviews — doesn't search

            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def build_task(self, config: dict, agent: Agent, context_tasks: list) -> Task:
        """
        Build the Critic Task.

        The critic receives write_task output as context,
        reviews it, fixes issues, and returns the final report.

        This is the LAST task — its output becomes the
        final result returned by main.py.
        """
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=agent,

            # Critic receives the writer's draft as input
            context=context_tasks,

            # Final output saved here — this is what gets printed
            output_file="outputs/final_report.md"
        )