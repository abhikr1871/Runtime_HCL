"""Check the generated data against the schema and the brief's minimums. Owner: M5.

Run:  python data_kit/validate_data.py      (exit code 1 if any check fails)
Save its output for the submission (the brief asks for the checker and its output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.schemas import Product  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def _load(name: str) -> list[dict]:
    path = PROCESSED / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main() -> int:
    problems: list[str] = []
    products = [p for f in sorted(PROCESSED.glob("*_products.json")) for p in json.loads(f.read_text(encoding="utf-8"))]
    ids = set()
    for p in products:
        try:
            Product(**p)
        except Exception as exc:
            problems.append(f"{p.get('product_id')}: invalid ({exc.__class__.__name__})")
        if p.get("product_id") in ids:
            problems.append(f"duplicate product_id {p.get('product_id')}")
        ids.add(p.get("product_id"))
    categories = {p.get("category") for p in products}
    if len(products) < 250:
        problems.append(f"only {len(products)} products (brief minimum 250)")
    if len(categories) < 5:
        problems.append(f"only {len(categories)} categories (brief minimum 5)")

    shoppers, orders = _load("shoppers.json"), _load("orders.json")
    if len(shoppers) < 25:
        problems.append(f"only {len(shoppers)} shoppers (brief minimum 25)")
    for o in orders:
        if o.get("product_id") not in ids:
            problems.append(f"order {o.get('order_id')} points to unknown product {o.get('product_id')}")
    # TODO(M5): shoppers with 3+ orders, edge cases present, mess_log counts, Hindi text share

    print(f"products={len(products)} categories={len(categories)} shoppers={len(shoppers)} orders={len(orders)}")
    for msg in problems:
        print("FAIL:", msg)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
