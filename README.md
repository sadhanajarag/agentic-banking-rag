# NorthStar Bank Assistant (Agentic RAG)

A chat assistant that answers questions from NorthStar Bank's customer guides
(credit cards, personal loans, digital banking), and uses web search for everything else.

Built with LangGraph, LangChain, GPT-4o-mini, Chroma, BM25, a cross-encoder reranker,
Tavily web search and Streamlit.

## How it works

```
question -> contextualize (follow-ups -> standalone) -> router
   router -> retrieval_agent: classify guide -> hybrid search (BM25 + vector) -> rerank
                              -> relevance grading -> rewrite & retry (max 3) -> answer or fallback
   router -> web_search:      Tavily search -> answer with sources
-> answer saved to the conversation (SQLite, one session per UUID)
```

## Setup

```bash
uv sync                                            # install dependencies into .venv
cp .env.example .env                               # then add OPENAI_API_KEY and TAVILY_API_KEY
uv run python -m northstar_rag.ingestion.pipeline  # build the search index (once, and when PDFs change)
```

## Run the app

```bash
uv run streamlit run src/northstar_rag/ui/streamlit_app.py
```

Each chat has its own session id in the URL (`?session=...`): refreshing the page or
reopening the link brings the conversation back. "New chat" starts a fresh session.

## Development

```bash
uv run pytest -q              # tests (needs both API keys in .env)
uv run ruff check src tests   # lint / PEP 8
```

## Layout

```
src/northstar_rag/
  config.py          settings (pydantic-settings, read from .env)
  logging_config.py  logging setup
  service.py         ChatService: the app layer used by the UI (sessions, memory, validation)
  ui/                Streamlit chat app
  ingestion/         PDF -> Markdown -> chunks -> Chroma index
  retrieval/         hybrid BM25 + vector search, reranker
  llm/               one file per LLM job (route, classify, grade, rewrite, generate, ...)
  tools/             Tavily web search tool
  graph/             LangGraph state, nodes and graph builders
tests/               pytest
notebook/            exploration only
```
