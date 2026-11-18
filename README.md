# NorthStar RAG

Agentic RAG assistant over NorthStar Bank customer guides, built with LangGraph, LangChain and Pydantic.

## Setup

```bash
uv sync                    # creates/updates .venv from pyproject.toml
cp .env.example .env       # then add your OPENAI_API_KEY
uv run pytest              # smoke test
uv run northstar-rag       # CLI entry point
```

## Layout

```
src/northstar_rag/
  config.py        settings (pydantic-settings)
  schemas.py       Pydantic models
  ingestion/       PDF loading, chunking, indexing
  retrieval/       hybrid BM25 + vector retrieval, reranker
  llm/             model factory, prompts
  graph/           LangGraph state, nodes, edges, builder
  app.py           CLI
scripts/           one-off jobs (build_index.py)
tests/             pytest
notebook/          exploration only
```
"# agentic-banking-rag" 
