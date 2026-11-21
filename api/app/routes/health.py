"""GET /health - 200 when ready. Owner: M3."""
from __future__ import annotations

from fastapi import APIRouter

from app.ai.llm_client import LLMClient
from app.config import get_settings
from app.schemas import HealthResponse
from app.store import indexer, repository

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        n = repository.count_products()
        db_ok = True
    except Exception:
        n, db_ok = 0, False
    llm = LLMClient().ping()
    ok = db_ok and n > 0 and llm in {"ok", "mock"}
    return HealthResponse(status="ok" if ok else "degraded", db=db_ok, products=n,
                          text_index=indexer.status(), llm=llm, mock_llm=get_settings().mock_llm)
