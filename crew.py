# crew.py
# ─────────────────────────────────────────────────────────────────
# The heart of the CrewAI setup.
# Reads agents.yaml and tasks.yaml, wires agents to tools,
# and exposes the crew for both direct use and LangGraph nodes.
# ─────────────────────────────────────────────────────────────────

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from langchain_openai import ChatOpenAI

from tools.web_search import WebSearchTool
from tools.file_reader import FileReaderTool
from tools.summarizer import SummarizerTool
from rag.retriever import retrieve
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME


@CrewBase
class ResearchCrew:
    """
    The main CrewAI crew class.
    @CrewBase decorator reads YAML files automatically.
    @agent, @task, @crew decorators register methods with CrewAI.
    """

    # Paths to YAML config files (relative to this file)
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self):
        # Shared LLM instance — all agents use the same model
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.3   # Slight creativity for writing tasks
        )

        # Instantiate tools once — reused across agents
        self.search_tool     = WebSearchTool()
        self.file_tool       = FileReaderTool()
        self.summarizer_tool = SummarizerTool()

    # ── Agent Definitions ─────────────────────────────────────────
    # config= pulls role/goal/backstory from agents.yaml
    # tools= gives the agent its capabilities

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            llm=self.llm,
            verbose=True   # Print agent thinking to console
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=self.llm,
            tools=[self.search_tool, self.file_tool, self.summarizer_tool],
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm=self.llm,
            verbose=True
        )

    @agent
    def critic(self) -> Agent:
        return Agent(
            config=self.agents_config["critic"],
            llm=self.llm,
            verbose=True
        )

    # ── Task Definitions ──────────────────────────────────────────
    # config= pulls description/expected_output/agent from tasks.yaml

    @task
    def plan_task(self) -> Task:
        return Task(config=self.tasks_config["plan_task"])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def write_task(self) -> Task:
        return Task(config=self.tasks_config["write_task"])

    @task
    def critic_task(self) -> Task:
        return Task(config=self.tasks_config["critic_task"])

    # ── Crew Assembly ─────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,       # auto-collected from @agent methods
            tasks=self.tasks,         # auto-collected from @task methods
            process=Process.sequential,  # tasks run in order: plan→research→write→critic
            verbose=True
        )

    def run_task(self, task_name: str, inputs: dict) -> str:
        """
        Run a single task by name — used by LangGraph nodes
        so each node can trigger just its own task.
        """
        task_map = {
            "plan_task":     self.plan_task(),
            "research_task": self.research_task(),
            "write_task":    self.write_task(),
            "critic_task":   self.critic_task(),
        }

        task_obj = task_map.get(task_name)
        if not task_obj:
            raise ValueError(f"Unknown task: {task_name}")

        # Run just this one task with its agent
        mini_crew = Crew(
            agents=[task_obj.agent],
            tasks=[task_obj],
            verbose=True
        )
        return mini_crew.kickoff(inputs=inputs)