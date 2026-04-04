# agents/planner.py
# ─────────────────────────────────────────────────────────────────
# The Planner agent's job is to receive the user's topic and
# break it into focused, researchable sub-questions.
#
# WHY a Planner?
#   Without a plan, the Researcher tries to answer everything at
#   once and produces shallow results. A plan forces depth.
#
# This file defines the Agent object + its Task.
# The YAML (config/agents.yaml + tasks.yaml) holds the
# role/goal/backstory/description — this file holds the logic.
# ─────────────────────────────────────────────────────────────────

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME
from config.prompts import PLANNER_SYSTEM


class PlannerAgent:
    """
    Wraps the CrewAI Agent + Task for the Planner role.
    Instantiated by crew.py and used in the @agent decorator.
    """

    def __init__(self):
        # Shared LLM — low temperature for structured, logical output
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.1   # Low = more focused, less creative
        )

    def build_agent(self, config: dict) -> Agent:
        """
        Build and return the CrewAI Agent object.

        config: dict loaded from config/agents.yaml["planner"]
                Contains: role, goal, backstory (with {topic} placeholder)
        """
        return Agent(
            # Pull role, goal, backstory from agents.yaml
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],

            # Inject system-level instructions on top of the YAML backstory
            system_template=PLANNER_SYSTEM,

            # Planner does NOT need web search — it only thinks and plans
            tools=[],

            llm=self.llm,

            # allow_delegation=False means this agent won't hand off
            # its work to another agent mid-task
            allow_delegation=False,

            verbose=True   # Print agent's chain-of-thought to terminal
        )

    def build_task(self, config: dict, agent: Agent) -> Task:
        """
        Build and return the CrewAI Task for planning.

        config: dict loaded from config/tasks.yaml["plan_task"]
                Contains: description, expected_output (with {topic} placeholder)
        agent : the Agent object returned by build_agent()
        """
        return Task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=agent,

            # output_file saves task result to disk automatically
            # Useful for debugging — check outputs/plan.md after a run
            output_file="outputs/plan.md"
        )