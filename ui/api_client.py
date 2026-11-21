"""Thin HTTP client: the UI only ever talks to the API. Owner: M4."""
from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # 127.0.0.1, not localhost: avoids a ~2 s IPv6 delay on Windows
_TIMEOUT = 60


def chat(session_id: str, customer_id: str, message: str) -> dict[str, Any]:
    r = httpx.post(f"{API_URL}/chat", json={"session_id": session_id, "customer_id": customer_id, "message": message},
                   timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_session(session_id: str, customer_id: str) -> dict[str, Any] | None:
    r = httpx.get(f"{API_URL}/sessions/{session_id}", params={"customer_id": customer_id}, timeout=10)
    return r.json() if r.status_code == 200 else None


def remove_filter(session_id: str, customer_id: str, name: str) -> None:
    httpx.delete(f"{API_URL}/sessions/{session_id}/filters/{name}", params={"customer_id": customer_id}, timeout=10)


def get_trace(trace_id: str) -> dict[str, Any] | None:
    r = httpx.get(f"{API_URL}/trace/{trace_id}", timeout=10)
    return r.json() if r.status_code == 200 else None


def ingest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r = httpx.post(f"{API_URL}/ingest", json=rows, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def health() -> dict[str, Any] | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        return None
