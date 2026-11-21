"""POST /ingest - add or update products while running; searchable on the next request. Owner: M2."""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import ValidationError

from app.schemas import IngestResult, Product
from app.store import indexer, repository
from app.store.normalise import weight_to_grams

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResult)
def ingest(rows: list[dict[str, Any]]) -> IngestResult:
    t0 = time.perf_counter()
    created = updated = 0
    errors: list[dict[str, Any]] = []
    changed: list[str] = []
    for i, row in enumerate(rows):
        try:
            p = Product(**row)
        except ValidationError as exc:
            errors.append({"row": i, "product_id": row.get("product_id"), "error": exc.errors(include_url=False)})
            continue
        if p.weight_g is None and p.weight_raw:
            p.weight_g = weight_to_grams(p.weight_raw)
        if repository.upsert_product(p):
            created += 1
        else:
            updated += 1
        changed.append(p.product_id)
    indexer.refresh(changed)
    # TODO(M2): accept a CSV upload (multipart) with the same columns
    return IngestResult(created=created, updated=updated, errors=errors, took_ms=int((time.perf_counter() - t0) * 1000))
