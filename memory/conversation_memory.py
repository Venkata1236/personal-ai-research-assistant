# memory/conversation_memory.py
# ─────────────────────────────────────────────────────────────────
# Manages short-term (current session) and long-term (cross-run)
# conversation memory.
#
# Short-term: ConversationBufferMemory — stores last N messages
# Long-term:  saved to a local JSON file between runs
# ─────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from langchain.memory import ConversationBufferMemory


# Path where long-term memory is persisted across runs
MEMORY_FILE = Path("./memory/long_term_memory.json")


class ResearchMemory:
    def __init__(self):
        # Short-term: holds the current conversation history in RAM
        # memory_key="chat_history" is the key agents use to access it
        self.short_term = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True   # Returns Message objects, not raw strings
        )

        # Long-term: load existing memory from disk (or start fresh)
        self.long_term = self._load_long_term()

    def save_research(self, topic: str, result: str):
        """
        Save a completed research result to long-term memory.
        Called after every successful pipeline run.
        """
        self.long_term[topic] = result
        self._save_long_term()
        print(f"[Memory] Saved research on: {topic}")

    def get_past_research(self, topic: str) -> str:
        """
        Check if we've already researched this topic.
        Returns the cached result or None.
        """
        return self.long_term.get(topic, None)

    def _load_long_term(self) -> dict:
        """Load long-term memory from JSON file."""
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return {}   # First run — start with empty memory

    def _save_long_term(self):
        """Persist long-term memory to JSON file."""
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.long_term, f, indent=2)