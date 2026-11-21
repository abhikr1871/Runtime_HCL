"""GET /trace/{trace_id} - replay how a reply was made (brief R10). Owner: M3."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.observe.trace_store import get_trace

router = APIRouter(tags=["trace"])


@router.get("/trace/{trace_id}")
def trace(trace_id: str) -> dict[str, Any]:
    t = get_trace(trace_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return t
