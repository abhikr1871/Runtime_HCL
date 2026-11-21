"""Test setup: fresh SQLite file per test session, mock LLM, sample catalogue."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["MOCK_LLM"] = "true"
    os.environ["ENABLE_EMBEDDINGS"] = "false"
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")

    from app.config import get_settings
    from app.store import db
    get_settings.cache_clear()
    db.reset_conn()

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
    db.reset_conn()


def ask(client, message: str, session: str = "s1", customer: str = "C-001") -> dict:
    r = client.post("/chat", json={"session_id": session, "customer_id": customer, "message": message})
    assert r.status_code == 200, r.text
    return r.json()
