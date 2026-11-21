"""Build data/shop.db from data/processed/*.json (replaces the sample catalogue). Owner: M5.

Run:  python data_kit/build_db.py      then restart the API.
Loads: *_products.json, *_reviews.json, shoppers.json, orders.json (whatever exists).
Weights are normalised here (weight_raw -> weight_g); unparseable ones stay empty ("not listed").
"""
from __future__ import annotations

import json

from common import PROCESSED, read_json
from app.config import get_settings  # noqa: E402
from app.schemas import Product  # noqa: E402
from app.store import db, repository  # noqa: E402
from app.store.normalise import weight_to_grams  # noqa: E402


def main() -> None:
    path = get_settings().db_path
    db.reset_conn()
    if path.exists():
        path.unlink()
    db.init_db()

    n_products = normalised = 0
    for f in sorted(PROCESSED.glob("*_products.json")):
        for raw in read_json(f, []):
            p = Product(**raw)
            if p.weight_g is None and p.weight_raw:
                p.weight_g = weight_to_grams(p.weight_raw)
                normalised += p.weight_g is not None
            repository.upsert_product(p)
            n_products += 1

    n_reviews = 0
    for f in sorted(PROCESSED.glob("*_reviews.json")):
        for r in read_json(f, []):
            db.execute("INSERT OR REPLACE INTO reviews (review_id, product_id, rating, text, lang, author_alias, injection_flag) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (r["review_id"], r["product_id"], r.get("rating"), r["text"], r.get("lang", "en"),
                        r.get("author_alias"), int(r.get("injection_flag", 0))))
            n_reviews += 1

    shoppers = read_json(PROCESSED / "shoppers.json", [])
    for s in shoppers:
        db.execute("INSERT OR REPLACE INTO shoppers (customer_id, display_name, city, preferred_language, budget_min_inr, "
                   "budget_max_inr, fav_brands_json, sizes_json, interests_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (s["customer_id"], s["display_name"], s["city"], s["preferred_language"], s["budget_min_inr"],
                    s["budget_max_inr"], json.dumps(s["fav_brands"], ensure_ascii=False),
                    json.dumps(s["sizes"], ensure_ascii=False), json.dumps(s["interests"], ensure_ascii=False)))
    orders = read_json(PROCESSED / "orders.json", [])
    for o in orders:
        db.execute("INSERT OR REPLACE INTO orders (order_id, customer_id, product_id, qty, price_paid_inr, ordered_at) "
                   "VALUES (?, ?, ?, ?, ?, ?)",
                   (o["order_id"], o["customer_id"], o["product_id"], o["qty"], o["price_paid_inr"], o["ordered_at"]))

    print(f"built {path}: {n_products} products ({normalised} weights normalised to grams), {n_reviews} reviews, "
          f"{len(shoppers)} shoppers, {len(orders)} orders")


if __name__ == "__main__":
    main()
