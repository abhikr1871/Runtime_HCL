"""POST /ingest and POST /ingest/csv - add or update products while running. Owner: M2-B.

Requirements:
- Accept JSON list of products or multipart CSV upload.
- Validate required columns/fields.
- Normalize incoming records (brand, category, size, color, weight, price) while preserving raw values.
- Handle missing and invalid fields safely.
- Prevent accidental duplicate insertion within the batch.
- Report successful (created, updated) and failed rows with precise error descriptions.
- Incrementally update indexes (BM25 + embeddings) so products are instantly searchable.
- Protect database from corrupt data.
"""
from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.schemas import IngestResult, Product
from app.store import indexer, repository
from app.store.normalise import normalize_product

router = APIRouter(tags=["ingest"])

_REQUIRED_COLUMNS = {"product_id", "name", "category"}


def _process_product_dicts(rows: list[dict[str, Any]], start_time: float) -> IngestResult:
    created = updated = 0
    errors: list[dict[str, Any]] = []
    changed: list[str] = []
    seen_in_batch: set[str] = set()

    for i, row in enumerate(rows):
        pid = row.get("product_id")
        if not pid or not str(pid).strip():
            errors.append({"row": i, "product_id": pid, "error": "Missing required field: product_id"})
            continue

        pid_str = str(pid).strip()
        if pid_str in seen_in_batch:
            errors.append({"row": i, "product_id": pid_str, "error": f"Duplicate product_id in batch: {pid_str}"})
            continue
        seen_in_batch.add(pid_str)

        # Normalise fields (preserves raw values, fixes brand casing, category, sizes, colors, weight_g)
        norm_row, _ = normalize_product(row, record_fixes=False)

        try:
            p = Product(**norm_row)
        except ValidationError as exc:
            errors.append({"row": i, "product_id": pid_str, "error": exc.errors(include_url=False)})
            continue
        except Exception as exc:
            errors.append({"row": i, "product_id": pid_str, "error": str(exc)})
            continue

        try:
            is_new = repository.upsert_product(p)
            if is_new:
                created += 1
            else:
                updated += 1
            changed.append(p.product_id)
        except Exception as exc:
            errors.append({"row": i, "product_id": pid_str, "error": f"Database error: {exc}"})

    if changed:
        indexer.update_product_index(changed)

    took_ms = int((time.perf_counter() - start_time) * 1000)
    return IngestResult(created=created, updated=updated, errors=errors, took_ms=took_ms)


@router.post("/ingest", response_model=IngestResult)
def ingest_json(rows: list[dict[str, Any]]) -> IngestResult:
    """Ingest products via JSON payload."""
    t0 = time.perf_counter()
    return _process_product_dicts(rows, t0)


def _parse_csv_cell(field: str, val: str) -> Any:
    """Parses individual CSV string values into expected Python types."""
    val = val.strip()
    if not val:
        return None

    # JSON or list fields
    if field in {"sizes", "colors", "tags"}:
        if val.startswith("[") and val.endswith("]"):
            try:
                return json.loads(val)
            except Exception:
                pass
        return [item.strip() for item in val.split(",") if item.strip()]

    if field == "attributes":
        if val.startswith("{") and val.endswith("}"):
            try:
                return json.loads(val)
            except Exception:
                pass
        return {}

    # Numeric fields
    if field in {"price_inr", "mrp_inr", "stock", "rating_count"}:
        try:
            return int(round(float(val)))
        except ValueError:
            return val

    if field in {"rating", "weight_g"}:
        try:
            return float(val)
        except ValueError:
            return val

    if field == "is_active":
        return val.lower() in {"1", "true", "yes", "t", "active"}

    return val


@router.post("/ingest/csv", response_model=IngestResult)
async def ingest_csv(file: UploadFile = File(...)) -> IngestResult:
    """Ingest products via CSV multipart upload."""
    t0 = time.perf_counter()
    content_bytes = await file.read()
    try:
        text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode("latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot decode CSV file: {exc}")

    f = io.StringIO(text)
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Empty or invalid CSV file (no headers found)")

    headers = {h.strip() for h in reader.fieldnames if h}
    missing_required = _REQUIRED_COLUMNS - headers
    if missing_required:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {sorted(missing_required)}. Found: {sorted(headers)}",
        )

    rows: list[dict[str, Any]] = []
    for r in reader:
        row_dict: dict[str, Any] = {}
        for k, v in r.items():
            if k and v is not None:
                parsed_val = _parse_csv_cell(k.strip(), v)
                if parsed_val is not None:
                    row_dict[k.strip()] = parsed_val
        if row_dict:
            rows.append(row_dict)

    return _process_product_dicts(rows, t0)
