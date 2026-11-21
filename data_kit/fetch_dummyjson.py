"""Download the DummyJSON demo catalogue and map it to our Product schema. Owner: M5.

Run:  python data_kit/fetch_dummyjson.py
Out:  data/processed/dummyjson_products.json, data/processed/dummyjson_reviews.json

Source: https://dummyjson.com/products (MIT-licensed demo data). Reviewer names/emails are DROPPED.
Prices: USD x USD_TO_INR (fixed demo rate), rounded to an Indian price point ending in 9.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
USD_TO_INR = 83


def inr_price(usd: float) -> int:
    value = int(round(usd * USD_TO_INR))
    return max(9, value - value % 10 + 9) if value >= 10 else 9


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = httpx.get("https://dummyjson.com/products", params={"limit": 0}, timeout=60).json()
    products, reviews = [], []
    for i, d in enumerate(data["products"], start=1):
        pid = f"P-{i:04d}"
        price = inr_price(d["price"])
        products.append({
            "product_id": pid, "source": "dummyjson", "name": d["title"], "category": d["category"],
            "brand": d.get("brand"), "price_inr": price,
            "mrp_inr": int(price / (1 - d.get("discountPercentage", 0) / 100)) if d.get("discountPercentage") else None,
            "stock": d.get("stock", 0), "rating": d.get("rating"), "rating_count": len(d.get("reviews", [])),
            "description": d.get("description", ""), "tags": d.get("tags", []),
            "weight_raw": str(d["weight"]) if d.get("weight") is not None else None,  # unitless in source: left ambiguous
            "image_path": None,
        })
        for j, r in enumerate(d.get("reviews", []), start=1):
            reviews.append({"review_id": f"R-{pid[2:]}-{j}", "product_id": pid, "rating": r.get("rating"),
                            "text": r.get("comment", ""), "lang": "en", "author_alias": f"shopper-{j}"})
    (OUT / "dummyjson_products.json").write_text(json.dumps(products, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "dummyjson_reviews.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(products)} products and {len(reviews)} reviews to {OUT}")


if __name__ == "__main__":
    main()
