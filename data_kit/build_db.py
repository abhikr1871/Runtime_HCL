"""Build data/shop.db from data/processed/*.json (replaces the sample catalogue). Owner: M5.

Run:  python data_kit/build_db.py      then restart the API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.config import get_settings  # noqa: E402
from app.schemas import Product  # noqa: E402
from app.store import db, repository  # noqa: E402
from app.store.normalise import weight_to_grams  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    path = get_settings().db_path
    if path.exists():
        path.unlink()
    db.init_db()
    n = 0
    for f in sorted(PROCESSED.glob("*_products.json")):
        for raw in json.loads(f.read_text(encoding="utf-8")):
            p = Product(**raw)
            if p.weight_g is None and p.weight_raw:
                p.weight_g = weight_to_grams(p.weight_raw)
            repository.upsert_product(p)
            n += 1
    for f in sorted(PROCESSED.glob("*_reviews.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            db.execute("INSERT OR REPLACE INTO reviews (review_id, product_id, rating, text, lang, author_alias, injection_flag) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (r["review_id"], r["product_id"], r.get("rating"), r["text"], r.get("lang", "en"),
                        r.get("author_alias"), int(r.get("injection_flag", 0))))
    # TODO(M5): load shoppers.json and orders.json into their tables
    print(f"built {path} with {n} products")


if __name__ == "__main__":
    main()
