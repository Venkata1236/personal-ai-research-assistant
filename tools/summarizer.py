# tools/summarizer.py
# ─────────────────────────────────────────────────────────────────
# Fix: BaseTool now imports from 'crewai.tools' not 'crewai_tools'
# ─────────────────────────────────────────────────────────────────

from crewai.tools import BaseTool          # ← FIXED import
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from config.settings import OPENAI_API_KEY, OPENAI_MODEL_NAME


class SummarizerTool(BaseTool):
    name: str = "Text Summarizer"
    description: str = (
        "Summarize long text into key bullet points. "
        "Input: long text string. "
        "Output: concise bullet-point summary."
    )

    def _run(self, text: str) -> str:
        llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        prompt = (
            "Summarize the following text into 5 clear bullet points. "
            "Focus on the most important facts and findings.\n\n"
            f"{text[:3000]}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content