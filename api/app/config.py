"""Settings read from environment variables (see .env.example). Owner: M3."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


_DEFAULT_URLS = {"groq": "https://api.groq.com/openai/v1"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str            # ollama | groq | openai (any OpenAI-compatible API)
    llm_model: str
    ollama_url: str
    hosted_api_url: str
    hosted_api_key: str
    llm_reasoning_effort: str    # optional, for reasoning models (e.g. gpt-oss): low | medium | high
    llm_timeout_s: float
    mock_llm: bool
    embed_model: str
    enable_embeddings: bool
    enable_reranker: bool
    db_path: Path
    models_dir: Path
    sample_products: Path
    retrieval_mode: str = "hybrid"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    diversity_strategy: str = "brand"
    vector_store: str = "sqlite"  # sqlite | chromadb
    chroma_dir: Path = ROOT / "data" / "chroma"


def _load_dotenv() -> None:
    """Read ROOT/.env into the environment (without overriding real env vars). No extra dependency."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache
def get_settings() -> Settings:
    _load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        hosted_api_url=os.getenv("HOSTED_API_URL") or _DEFAULT_URLS.get(provider, ""),
        hosted_api_key=os.getenv("GROQ_API_KEY") or os.getenv("HOSTED_API_KEY", ""),
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", ""),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "20")),
        mock_llm=_bool("MOCK_LLM", True),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        enable_embeddings=_bool("ENABLE_EMBEDDINGS", False),
        enable_reranker=_bool("ENABLE_RERANKER", False),
        db_path=_path("DB_PATH", "data/shop.db"),
        models_dir=_path("MODELS_DIR", "models"),
        sample_products=ROOT / "data_kit" / "sample_products.json",
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        diversity_strategy=os.getenv("DIVERSITY_STRATEGY", "brand").strip().lower(),
        vector_store=os.getenv("VECTOR_STORE", "sqlite").strip().lower(),
        chroma_dir=_path("CHROMA_DIR", "data/chroma"),
    )
