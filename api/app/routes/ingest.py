"""POST /ingest - add or update products while running; searchable on the next request. Owner: M2."""
from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import ValidationError

from app.schemas import IngestResult, Product
from app.store import indexer, repository
from app.store.normalise import normalise_product

router = APIRouter(tags=["ingest"])


def _ingest_dict_rows(rows: list[dict[str, Any]]) -> IngestResult:
    t0 = time.perf_counter()
    created = updated = 0
    errors: list[dict[str, Any]] = []
    changed: list[str] = []

    for i, row in enumerate(rows):
        cleaned_row, _ = normalise_product(row)
        try:
            p = Product(**cleaned_row)
        except ValidationError as exc:
            errors.append({"row": i, "product_id": row.get("product_id"), "error": exc.errors(include_url=False)})
            continue

        if repository.upsert_product(p):
            created += 1
        else:
            updated += 1
        changed.append(p.product_id)

    if changed:
        indexer.refresh(changed)

    return IngestResult(
        created=created,
        updated=updated,
        errors=errors,
        took_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.post("/ingest", response_model=IngestResult)
def ingest(rows: list[dict[str, Any]]) -> IngestResult:
    return _ingest_dict_rows(rows)


@router.post("/ingest/csv", response_model=IngestResult)
async def ingest_csv(file: UploadFile = File(...)) -> IngestResult:
    """Accepts CSV file upload with catalogue products."""
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))
    rows: list[dict[str, Any]] = []
    for row in reader:
        d = dict(row)
        for json_key in ("tags", "sizes", "colors", "attributes"):
            if json_key in d and d[json_key]:
                try:
                    d[json_key] = json.loads(d[json_key])
                except Exception:
                    d[json_key] = [s.strip() for s in d[json_key].split(",") if s.strip()]
        rows.append(d)
    return _ingest_dict_rows(rows)
