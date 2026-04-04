# crew.py
# ─────────────────────────────────────────────────────────────────
# Fix: Newer CrewAI expects llm as a string "provider/model"
# NOT as a ChatOpenAI object directly
# ─────────────────────────────────────────────────────────────────

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

from tools.web_search import WebSearchTool
from tools.file_reader import FileReaderTool
from tools.summarizer import SummarizerTool
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME
import os


# Set API key in environment — CrewAI reads it from here
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@CrewBase
class ResearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    def __init__(self):
        # ── Pass LLM as string — CrewAI's expected format ─────────
        # Format: "openai/model-name"
        self.llm = f"openai/{OPENAI_MODEL_NAME}"

        # Instantiate tools once
        self.search_tool     = WebSearchTool()
        self.file_tool       = FileReaderTool()
        self.summarizer_tool = SummarizerTool()

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],
            llm=self.llm,
            verbose=True
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

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    def run_task(self, task_name: str, inputs: dict) -> str:
        """Run a single task by name — used by LangGraph nodes."""
        task_map = {
            "plan_task":     self.plan_task(),
            "research_task": self.research_task(),
            "write_task":    self.write_task(),
            "critic_task":   self.critic_task(),
        }
        task_obj = task_map.get(task_name)
        if not task_obj:
            raise ValueError(f"Unknown task: {task_name}")

        mini_crew = Crew(
            agents=[task_obj.agent],
            tasks=[task_obj],
            verbose=True
        )
        return mini_crew.kickoff(inputs=inputs)