"""POST /chat (brief contract). Owner: M3."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.pipeline.orchestrator import SessionOwnershipError, handle_chat
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_customer_id: Optional[str] = Header(default=None)) -> ChatResponse:
    # Identity comes from the request, never from the message text.
    if x_customer_id and x_customer_id != req.customer_id:
        raise HTTPException(status_code=400, detail="customer_id in body and X-Customer-Id header differ")
    try:
        return handle_chat(req)
    except SessionOwnershipError:
        raise HTTPException(status_code=403, detail="This session belongs to another customer.")
