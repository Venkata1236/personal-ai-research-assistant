# 🔬 Personal AI Research Assistant

> Full agentic pipeline — CrewAI + LangGraph + RAG + Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.28.8-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4--Turbo-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-yellow)

---

## 📌 What Is This?

A fully autonomous AI research assistant. You give it a topic — it plans, searches the web, retrieves from your documents (RAG), writes a structured report, and reviews it for quality. All powered by 4 specialised AI agents working in sequence, orchestrated by either CrewAI or LangGraph.

---

## 🗺️ Pipeline Flow

```
User enters topic
        ↓
Streamlit UI (app.py)
        ↓
┌──────────────────────────────────────────┐
│         Choose Pipeline Mode             │
│                                          │
│  CrewAI Direct          LangGraph        │
│  ─────────────          ─────────        │
│  Planner                plan node        │
│     ↓                      ↓            │
│  Researcher             research node    │
│     ↓                      ↓            │
│  Writer                 write node       │
│     ↓                      ↓            │
│  Critic                 critic node      │
└──────────────────────────────────────────┘
        ↓
   Tools Used Per Agent
   ┌─────────────────────────────┐
   │ Web Search (Tavily API)     │
   │ RAG Retrieval (FAISS)       │
   │ File Reader (PDF/TXT/DOCX)  │
   │ Summarizer (LLM)            │
   └─────────────────────────────┘
        ↓
Final Report (Markdown) + Download
```

---

## 📁 Project Structure

```
personal-ai-research-assistant/
│
├── app.py                        ← Streamlit UI — pipeline + RAG upload + memory
├── main.py                       ← CLI entry point (crew or graph mode)
├── crew.py                       ← CrewAI @CrewBase wiring (reads YAML)
│
├── config/
│   ├── agents.yaml               ← Agent roles, goals, backstories
│   ├── tasks.yaml                ← Task descriptions + expected outputs
│   ├── settings.py               ← Loads .env — single config source
│   └── prompts.py                ← System prompts for all 4 agents
│
├── agents/
│   ├── planner.py                ← Breaks topic into sub-questions
│   ├── researcher.py             ← Web search + RAG + file reading
│   ├── writer.py                 ← Structures findings into report
│   └── critic.py                 ← Reviews + polishes final output
│
├── graph/
│   ├── state.py                  ← AgentState TypedDict (shared state)
│   ├── nodes.py                  ← One function per LangGraph node
│   └── research_graph.py         ← StateGraph build + compile
│
├── rag/
│   ├── ingestor.py               ← Load + chunk documents
│   ├── embedder.py               ← Embed chunks → FAISS index
│   └── retriever.py              ← Query vector store → top-k chunks
│
├── tools/
│   ├── web_search.py             ← Tavily API wrapper (BaseTool)
│   ├── file_reader.py            ← Read PDF / TXT / DOCX files
│   └── summarizer.py             ← Condense long content (LLM)
│
├── memory/
│   └── conversation_memory.py    ← Short-term + long-term JSON memory
│
├── data/
│   ├── raw/                      ← Drop documents here for RAG ingestion
│   └── processed/                ← Chunked documents (auto-generated)
│
├── outputs/                      ← Per-task markdown outputs (auto-saved)
├── tests/
│   └── test_rag.py               ← RAG pipeline unit tests
│
├── .env                          ← API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤖 The 4 Agents

| Agent | Role | Tools | Temp |
|---|---|---|---|
| **Planner** | Breaks topic into 3–5 focused sub-questions | None | 0.1 |
| **Researcher** | Finds facts via web + RAG documents | Web Search, RAG, File Reader, Summarizer | 0.2 |
| **Writer** | Structures findings into Introduction → Findings → Conclusion | None | 0.6 |
| **Critic** | Fact-checks, fills gaps, polishes final report | None | 0.1 |

---

## 🔑 Pipeline Modes

| Mode | How It Works | Best For |
|---|---|---|
| **CrewAI Direct** | `@CrewBase` reads YAML — agents run sequentially via `kickoff()` | Simple, single-topic research |
| **LangGraph** | `StateGraph` with typed state — each node is one agent function | Complex topics needing conditional routing |

---

## 🧠 Key Concepts

| Concept | What It Does |
|---|---|
| `@CrewBase` | Auto-reads `agents.yaml` + `tasks.yaml` — no manual wiring |
| `{topic}` in YAML | Runtime placeholder — filled by `kickoff(inputs={"topic": ...})` |
| `context: [task]` | Passes one agent's output as input to the next agent |
| `AgentState` | TypedDict shared across all LangGraph nodes |
| `RAG retrieve()` | Embeds query → FAISS similarity search → returns top-k chunks |
| `ResearchMemory` | Caches results to JSON — skips re-running duplicate topics |

---

## ⚙️ Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/Venkata1236/personal-ai-research-assistant.git
cd personal-ai-research-assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API keys**
```bash
# Create .env file
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL_NAME=gpt-4-turbo-preview
TAVILY_API_KEY=tvly-your-tavily-key-here
VECTOR_STORE_TYPE=faiss
VECTOR_STORE_PATH=./rag/vector_store
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

**5. Run the app**
```bash
# Streamlit UI
streamlit run app.py

# OR CLI mode
python main.py
```

---

## 🚀 Deploy to Streamlit Cloud

**1. Push to GitHub**
```bash
git add .
git commit -m "feat: ready for deployment"
git push origin main
```

**2. Go to [share.streamlit.io](https://share.streamlit.io)**
- Click **New app**
- Select your repo: `personal-ai-research-assistant`
- Set main file: `app.py`
- Click **Deploy**

**3. Add secrets in Streamlit Cloud**
```
Go to App Settings → Secrets → paste your .env contents:

OPENAI_API_KEY = "sk-..."
TAVILY_API_KEY = "tvly-..."
OPENAI_MODEL_NAME = "gpt-4-turbo-preview"
VECTOR_STORE_TYPE = "faiss"
VECTOR_STORE_PATH = "./rag/vector_store"
CHUNK_SIZE = "500"
CHUNK_OVERLAP = "50"
TOP_K_RESULTS = "5"
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 📦 Tech Stack

- **CrewAI** — Multi-agent orchestration with YAML config
- **LangGraph** — Stateful agent graph with conditional routing
- **LangChain** — LLM chains, document loaders, text splitters
- **OpenAI** — GPT-4-Turbo as the LLM backbone
- **FAISS** — Local vector store for RAG similarity search
- **Tavily** — Web search API optimised for AI agents
- **Streamlit** — Interactive UI with file upload + live progress
- **HuggingFace** — `all-MiniLM-L6-v2` embedding model (local, free)

---

## 👤 Author

**Venkata Reddy Bommavaram**
- 📧 bommavaramvenkat2003@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/venkatareddy1203)
- 🐙 [GitHub](https://github.com/Venkata1236)