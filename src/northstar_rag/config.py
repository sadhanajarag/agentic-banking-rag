"""All project settings in one place.

Values are read from the .env file. If a value is not in .env,
the default written below is used. Pydantic checks every value.
"""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root folder: C:\NorthStar-rag
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project settings. Each field can be overridden by an UPPER_CASE line in .env."""

    # Tell pydantic where the .env file is
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # OpenAI
    openai_api_key: str = Field(min_length=1)           # must not be empty
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = Field(0.0, ge=0, le=2)         # 0 to 2

    # Folders and files
    data_dir: Path = PROJECT_ROOT / "data" / "Resources_gemini"
    persist_dir: Path = PROJECT_ROOT / "vector_db" / "chroma"
    collection_name: str = "northstar_docs"

    # Chunking
    chunk_size: int = Field(2000, gt=0)
    chunk_overlap: int = Field(200, ge=0)

    # Retrieval
    retrieval_k: int = Field(20, gt=0)
    bm25_weight: float = Field(0.4, ge=0, le=1)
    vector_weight: float = Field(0.6, ge=0, le=1)

    # Reranker
    # MiniLM-L6: ~9x faster on CPU than BAAI/bge-reranker-base, similar top-4 quality
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = Field(4, gt=0)
    rerank_batch_size: int = Field(8, gt=0)   # small batches = less padding = faster on CPU
    torch_threads: int = Field(6, gt=0)       # measured: 6 beats 10 here; tune per machine

    # Web search (Tavily)
    tavily_api_key: str = Field(min_length=1)
    web_search_results: int = Field(3, gt=0)

    # Chat / app
    chat_db_path: Path = PROJECT_ROOT / "storage" / "chat_memory.sqlite"  # chat memory
    history_messages: int = Field(6, gt=0)       # earlier messages used to understand follow-ups
    max_question_chars: int = Field(1000, gt=0)  # reject very long input
    log_level: str = "INFO"

    # Graph
    max_rewrites: int = Field(3, ge=0)

    # Rules that compare two values
    @model_validator(mode="after")
    def check_values(self) -> "Settings":
        """Check rules that involve more than one field (runs after the field checks)."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if round(self.bm25_weight + self.vector_weight, 2) != 1.0:
            raise ValueError("bm25_weight + vector_weight must add up to 1.0")
        if not self.data_dir.exists():
            raise ValueError(f"data_dir not found: {self.data_dir}")
        return self


# Create one settings object that the whole project imports
settings = Settings()
