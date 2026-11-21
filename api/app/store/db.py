"""SQLite connection and setup. Owner: M3."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from app.config import get_settings

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    """One shared connection; every access goes through execute()/query() under a lock."""
    global _conn
    if _conn is None:
        path = get_settings().db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
    return _conn


def reset_conn() -> None:
    """Used by tests to point at a fresh database."""
    global _conn
    if _conn is not None:
        _conn.close()
    _conn = None


def execute(sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
    with _lock:
        cur = get_conn().execute(sql, params)
        get_conn().commit()
        return cur


def query(sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
    with _lock:
        return get_conn().execute(sql, params).fetchall()


def init_db() -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with _lock:
        get_conn().executescript(schema)
        get_conn().commit()


def seed_sample_if_empty() -> None:
    """Loads a few sample products so the app works before the data kit (M5) has run."""
    from app.schemas import Product
    from app.store import repository

    if repository.count_products() > 0:
        return
    sample = get_settings().sample_products
    if not sample.exists():
        return
    for raw in json.loads(sample.read_text(encoding="utf-8")):
        repository.upsert_product(Product(**raw))
