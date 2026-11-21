"""GET /sessions/{id} and DELETE /sessions/{id}/filters/{name} - powers the UI filter chips. Owner: M3.

Only the session owner may read or change it.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.schemas import SessionState
from app.store import repository

router = APIRouter(tags=["sessions"])


def _owned(session_id: str, customer_id: str) -> SessionState:
    s = repository.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.customer_id != customer_id:
        raise HTTPException(status_code=403, detail="This session belongs to another customer.")
    return s


@router.get("/sessions/{session_id}")
def get_session(session_id: str, customer_id: str = Query(...)) -> dict[str, Any]:
    s = _owned(session_id, customer_id)
    return {"active_filters": s.filters, "last_results": s.last_results, "clarify_streak": s.clarify_streak}


@router.delete("/sessions/{session_id}/filters/{name}")
def remove_filter(session_id: str, name: str, customer_id: str = Query(...)) -> dict[str, Any]:
    s = _owned(session_id, customer_id)
    filters = {k: v for k, v in s.filters.items() if k != name}
    repository.save_session(s.model_copy(update={"filters": filters}))
    return {"active_filters": filters}
