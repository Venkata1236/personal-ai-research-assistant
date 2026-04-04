# memory/conversation_memory.py
# ─────────────────────────────────────────────────────────────────
# Fix: langchain.memory moved to langchain_community.memory
# ─────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from langchain_community.memory.kg import ConversationKGMemory  
from langchain_core.chat_history import InMemoryChatMessageHistory  # ← FIXED

MEMORY_FILE = Path("./memory/long_term_memory.json")


class ResearchMemory:
    def __init__(self):
        # Simple in-memory chat history — no external dependency
        self.short_term = InMemoryChatMessageHistory()
        self.long_term = self._load_long_term()

    def save_research(self, topic: str, result: str):
        """Save completed research to long-term memory."""
        self.long_term[topic] = result
        self._save_long_term()
        print(f"[Memory] Saved research on: {topic}")

    def get_past_research(self, topic: str) -> str:
        """Return cached result if topic was researched before."""
        return self.long_term.get(topic, None)

    def _load_long_term(self) -> dict:
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_long_term(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.long_term, f, indent=2)