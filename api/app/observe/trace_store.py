"""Save and fetch the replayable record of every reply (brief R10). Owner: M3."""
from __future__ import annotations

import json
from typing import Any, Optional

from app.store import db


def save_trace(trace_id: str, session_id: str, customer_id: str, endpoint: str, record: dict[str, Any]) -> None:
    db.execute(
        "INSERT INTO traces (trace_id, session_id, customer_id, endpoint, record_json) VALUES (?, ?, ?, ?, ?)",
        (trace_id, session_id, customer_id, endpoint, json.dumps(record, ensure_ascii=False, default=str)),
    )


def get_trace(trace_id: str) -> Optional[dict[str, Any]]:
    rows = db.query("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
    if not rows:
        return None
    r = dict(rows[0])
    r["record"] = json.loads(r.pop("record_json"))
    return r
