"""All SQL lives here so the pipeline never builds queries by hand. Owner: M3."""
from __future__ import annotations

import json
from typing import Any, Optional

from app.schemas import Product, Review, SessionState
from app.store import db

_JSON_COLS = {"attributes": "attributes_json", "sizes": "sizes_json", "colors": "colors_json", "tags": "tags_json"}


def _row_to_product(row) -> Product:
    d = dict(row)
    for field, col in _JSON_COLS.items():
        d[field] = json.loads(d.pop(col) or ("{}" if field == "attributes" else "[]"))
    d["is_active"] = bool(d["is_active"])
    d.pop("version", None)
    d.pop("updated_at", None)
    return Product(**d)


def count_products(active_only: bool = True) -> int:
    sql = "SELECT COUNT(*) AS n FROM products" + (" WHERE is_active = 1" if active_only else "")
    return db.query(sql)[0]["n"]


def get_product(product_id: str) -> Optional[Product]:
    rows = db.query("SELECT * FROM products WHERE product_id = ?", (product_id,))
    return _row_to_product(rows[0]) if rows else None


def get_products(product_ids: list[str]) -> list[Product]:
    """Products in the order of the given IDs (missing IDs are skipped)."""
    if not product_ids:
        return []
    marks = ",".join("?" * len(product_ids))
    rows = db.query(f"SELECT * FROM products WHERE product_id IN ({marks})", list(product_ids))
    by_id = {p.product_id: p for p in map(_row_to_product, rows)}
    return [by_id[i] for i in product_ids if i in by_id]


def find_products(where: str = "1=1", params: list[Any] | None = None) -> list[Product]:
    """`where` must be built from fixed column names with ? placeholders (see pipeline/filters.py)."""
    rows = db.query(f"SELECT * FROM products WHERE is_active = 1 AND ({where})", params or [])
    return [_row_to_product(r) for r in rows]


def upsert_product(p: Product) -> bool:
    """Insert or update one product. Returns True if it was created."""
    existed = get_product(p.product_id) is not None
    data = p.model_dump()
    for field, col in _JSON_COLS.items():
        data[col] = json.dumps(data.pop(field), ensure_ascii=False)
    data["is_active"] = int(data["is_active"])
    cols = list(data.keys())
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "product_id")
    db.execute(
        f"INSERT INTO products ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT(product_id) DO UPDATE SET {updates}, version = version + 1, updated_at = datetime('now')",
        [data[c] for c in cols],
    )
    db.execute("INSERT INTO ingest_log (product_id, action) VALUES (?, ?)", (p.product_id, "update" if existed else "create"))
    return not existed


def get_reviews(product_id: str) -> list[Review]:
    rows = db.query(
        "SELECT review_id, product_id, rating, text, lang, author_alias, injection_flag FROM reviews WHERE product_id = ?",
        (product_id,),
    )
    return [Review(**{**dict(r), "injection_flag": bool(r["injection_flag"])}) for r in rows]


def get_session(session_id: str) -> Optional[SessionState]:
    rows = db.query("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,))
    return SessionState.model_validate_json(rows[0]["state_json"]) if rows else None


def save_session(state: SessionState) -> None:
    db.execute(
        "INSERT INTO sessions (session_id, customer_id, state_json) VALUES (?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, updated_at = datetime('now')",
        (state.session_id, state.customer_id, state.model_dump_json()),
    )


def get_orders_for(customer_id: str) -> list[dict]:
    """History for ONE customer. Callers pass the session owner's ID, never one taken from the message."""
    rows = db.query("SELECT * FROM orders WHERE customer_id = ? ORDER BY ordered_at DESC", (customer_id,))
    return [dict(r) for r in rows]


# ---------- embeddings (Owner: M2-B) ----------
def save_embedding(product_id: str, kind: str, model: str, vector_bytes: bytes) -> None:
    """Save or update an embedding vector for a product."""
    db.execute(
        "INSERT INTO embeddings (product_id, kind, model, vector, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT(product_id, kind) DO UPDATE SET model = excluded.model, "
        "vector = excluded.vector, updated_at = datetime('now')",
        (product_id, kind, model, vector_bytes),
    )


def save_embeddings_batch(records: list[tuple[str, str, str, bytes]]) -> None:
    """Batch save embeddings in a single transaction."""
    if not records:
        return
    with db._lock:
        conn = db.get_conn()
        conn.executemany(
            "INSERT INTO embeddings (product_id, kind, model, vector, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(product_id, kind) DO UPDATE SET model = excluded.model, "
            "vector = excluded.vector, updated_at = datetime('now')",
            records,
        )
        conn.commit()


def get_embedding(product_id: str, kind: str = "text") -> Optional[bytes]:
    """Retrieve raw vector bytes for a single product."""
    rows = db.query("SELECT vector FROM embeddings WHERE product_id = ? AND kind = ?", (product_id, kind))
    return rows[0]["vector"] if rows else None


def get_all_embeddings(kind: str = "text") -> dict[str, bytes]:
    """Retrieve all embeddings of a given kind as a mapping of product_id -> vector_bytes."""
    rows = db.query("SELECT product_id, vector FROM embeddings WHERE kind = ?", (kind,))
    return {r["product_id"]: r["vector"] for r in rows}


def delete_embedding(product_id: str, kind: str = "text") -> None:
    """Delete an embedding record."""
    db.execute("DELETE FROM embeddings WHERE product_id = ? AND kind = ?", (product_id, kind))


def deactivate_product(product_id: str) -> bool:
    """Soft-delete: mark product inactive so it never appears in search."""
    cur = db.execute("UPDATE products SET is_active = 0, updated_at = datetime('now') WHERE product_id = ?", (product_id,))
    return cur.rowcount > 0


def delete_product(product_id: str) -> bool:
    """Hard-delete product and its embeddings."""
    db.execute("DELETE FROM embeddings WHERE product_id = ?", (product_id,))
    cur = db.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
    return cur.rowcount > 0

