# config/settings.py
# ─────────────────────────────────────────────────────────────────
# Central config loader.
# All files import from here — never read os.environ directly.
# ─────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# ── LLM Settings ─────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME  = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo-preview")

# ── Search Settings ───────────────────────────────────────────────
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")

# ── RAG Settings ──────────────────────────────────────────────────
VECTOR_STORE_TYPE  = os.getenv("VECTOR_STORE_TYPE", "faiss")
VECTOR_STORE_PATH  = os.getenv("VECTOR_STORE_PATH", "./rag/vector_store")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_RESULTS      = int(os.getenv("TOP_K_RESULTS", 5))