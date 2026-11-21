"""Generate 30 synthetic shoppers and their orders. Owner: M5.

Run:   python data_kit/generate_shoppers.py            (LLM writes names, favourite brands, sizes)
       python data_kit/generate_shoppers.py --offline  (template names, no LLM)
Needs: data/processed/*_products.json (run fetch_dummyjson.py, generate_products.py, plant_mess.py first)
Out:   data/processed/shoppers.json, orders.json, edge_cases.json

Who decides what:
- CODE fixes customer_id, city, language, budget, interests, and picks every order from REAL product IDs.
- The LLM writes display_name, fav_brands (only from the allowed list) and sizes (prompt: prompts/shoppers.md).
Edge cases required by the brief (IDs recorded in edge_cases.json):
  C-028 brand-new shopper with no orders
  C-029 contradictory tastes: premium favourite brands but a tiny budget
  C-030 favourite products are now out of stock
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from common import PROCESSED, REFERENCE_DATE, batched, get_llm, load_catalogue, load_prompt, log_generation, rng, write_json

N_SHOPPERS = 30
NEW_SHOPPER, CONTRADICTORY, OUT_OF_STOCK_FAN = "C-028", "C-029", "C-030"
CITIES = ["Pune", "Lucknow", "Bengaluru", "Jaipur", "Kolkata", "Chennai", "Delhi", "Mumbai", "Indore", "Patna", "Kochi", "Bhopal"]
LANGUAGES = ["en"] * 14 + ["hinglish"] * 9 + ["hi"] * 7
BUDGETS = [(300, 1500), (1000, 4000), (3000, 10000), (8000, 40000)]
INTERESTS = {
    "trekking": ["trekking-gear"], "music": ["audio"], "running": ["running-shoes"], "ethnic wear": ["ethnic-wear"],
    "cooking": ["kitchen", "kitchen-accessories", "groceries"], "beauty": ["beauty", "skin-care", "fragrances"],
    "gadgets": ["smartphones", "laptops", "tablets", "mobile-accessories"], "home": ["furniture", "home-decoration"],
    "fashion": ["mens-shirts", "mens-shoes", "womens-dresses", "womens-shoes", "tops", "womens-bags"],
    "sports": ["sports-accessories"], "watches": ["mens-watches", "womens-watches"],
}
PREMIUM_BRANDS = ["Sony", "JBL", "Apple", "ASICS", "Columbia", "Manyavar"]
FALLBACK_NAMES = ["Asha K.", "Rohan M.", "Priyanka S.", "Imran A.", "Lakshmi R.", "Vikram T.", "Neha G.", "Arjun P.",
                  "Fatima Z.", "Karthik V.", "Sneha D.", "Harpreet S.", "Ananya B.", "Rahul J.", "Meera N.", "Sanjay Y.",
                  "Pooja L.", "Aditya C.", "Divya H.", "Manoj E.", "Kavya I.", "Siddharth O.", "Ritu F.", "Naveen U.",
                  "Shreya Q.", "Abdul W.", "Tanvi X.", "Gaurav K.", "Isha M.", "Deepak R."]


class LLMShopper(BaseModel):
    customer_id: str
    display_name: str = Field(min_length=2, max_length=40)
    fav_brands: list[str] = []
    sizes: dict[str, str] = {}


def skeletons(catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r = rng("shoppers")
    brands_by_cat: dict[str, set[str]] = {}
    for p in catalogue:
        if p.get("brand"):
            brands_by_cat.setdefault(p["category"], set()).add(p["brand"])
    langs = LANGUAGES[:]
    r.shuffle(langs)
    out = []
    for i in range(1, N_SHOPPERS + 1):
        interests = r.sample([k for k, cats in INTERESTS.items() if any(c in brands_by_cat for c in cats)], k=2)
        lo, hi = r.choice(BUDGETS)
        available = sorted({b for k in interests for c in INTERESTS[k] for b in brands_by_cat.get(c, set())})
        out.append({"customer_id": f"C-{i:03d}", "city": r.choice(CITIES), "preferred_language": langs[i - 1],
                    "budget_min_inr": lo, "budget_max_inr": hi, "interests": interests, "available_brands": available[:25]})
    c = next(s for s in out if s["customer_id"] == CONTRADICTORY)
    c["budget_min_inr"], c["budget_max_inr"] = 200, 800
    return out


def ask_llm(llm, prompt: str, batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    data = llm.complete_json("You return only valid JSON.", prompt.replace("{skeletons_json}", json.dumps(batch, ensure_ascii=False)),
                             temperature=0.8, max_tokens=2000)
    allowed = {s["customer_id"]: set(s["available_brands"]) for s in batch}
    good = {}
    for raw in (data or {}).get("shoppers", []) if isinstance(data, dict) else []:
        try:
            s = LLMShopper(**raw)
        except (ValidationError, TypeError):
            continue
        if s.customer_id in allowed:
            s.fav_brands = [b for b in s.fav_brands if b in allowed[s.customer_id]][:3]   # only real, allowed brands
            s.sizes = {k: v for k, v in s.sizes.items() if k in {"shoe", "clothing"}}
            good[s.customer_id] = s.model_dump()
    return good


def pick_orders(r, shopper: dict[str, Any], catalogue: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    cats = {c for k in shopper["interests"] for c in INTERESTS[k]}
    cap = shopper["budget_max_inr"] * 1.5
    pool = [p for p in catalogue if p["category"] in cats and p.get("price_inr") and p["price_inr"] <= cap] or \
           [p for p in catalogue if p.get("price_inr") and p["price_inr"] <= cap] or catalogue
    fav = [p for p in pool if p.get("brand") in shopper["fav_brands"]]
    return [r.choice(fav if fav and r.random() < 0.6 else pool) for _ in range(n)]


def main(offline: bool) -> None:
    catalogue = load_catalogue()
    if not catalogue:
        raise SystemExit("no products in data/processed: run fetch_dummyjson.py, generate_products.py, plant_mess.py first")
    r = rng("orders")
    skel = skeletons(catalogue)
    llm = get_llm(offline)
    from_llm: dict[str, dict[str, Any]] = {}
    if llm:
        prompt = load_prompt("shoppers.md")
        for batch in batched(skel, 5):   # small batches fit Groq's free-tier tokens-per-minute limit
            got = ask_llm(llm, prompt, batch)
            missing = [s for s in batch if s["customer_id"] not in got]
            if missing:
                got.update(ask_llm(llm, prompt, missing))
            from_llm.update(got)
            print(f"shoppers: {len(from_llm)}/{len(skel)} from LLM")
            time.sleep(2)

    shoppers = []
    for i, s in enumerate(skel):
        gen = from_llm.get(s["customer_id"]) or {
            "display_name": FALLBACK_NAMES[i % len(FALLBACK_NAMES)],
            "fav_brands": r.sample(s["available_brands"], k=min(2, len(s["available_brands"]))), "sizes": {}}
        shoppers.append({k: v for k, v in s.items() if k != "available_brands"} |
                        {"display_name": gen["display_name"], "fav_brands": gen["fav_brands"], "sizes": gen["sizes"]})

    by_id = {s["customer_id"]: s for s in shoppers}
    premium = [b for b in PREMIUM_BRANDS if any(p.get("brand") == b for p in catalogue)]
    by_id[CONTRADICTORY]["fav_brands"] = premium[:3]
    out_of_stock = [p for p in catalogue if p.get("stock") == 0 and p.get("brand")]
    if out_of_stock:
        by_id[OUT_OF_STOCK_FAN]["fav_brands"] = sorted({p["brand"] for p in out_of_stock[:2]})

    orders, n = [], 1
    for s in shoppers:
        if s["customer_id"] == NEW_SHOPPER:
            continue
        picks = pick_orders(r, s, catalogue, r.randint(3, 5))
        if s["customer_id"] == OUT_OF_STOCK_FAN and out_of_stock:
            picks = out_of_stock[:2] + picks[:2]
        for p in picks:
            day = REFERENCE_DATE - timedelta(days=r.randint(7, 200))
            orders.append({"order_id": f"O-{n:04d}", "customer_id": s["customer_id"], "product_id": p["product_id"],
                           "qty": 2 if r.random() < 0.1 else 1, "price_paid_inr": p.get("price_inr"),
                           "ordered_at": day.isoformat()})
            n += 1

    edge = {NEW_SHOPPER: "brand-new shopper, no orders",
            CONTRADICTORY: f"premium favourite brands {by_id[CONTRADICTORY]['fav_brands']} but budget ₹200-₹800",
            OUT_OF_STOCK_FAN: f"ordered out-of-stock products {[p['product_id'] for p in out_of_stock[:2]]}"}
    write_json(PROCESSED / "shoppers.json", shoppers)
    write_json(PROCESSED / "orders.json", orders)
    write_json(PROCESSED / "edge_cases.json", edge)
    log_generation("generate_shoppers.py", llm, ["shoppers.md"], shoppers=len(shoppers), orders=len(orders),
                   from_llm=len(from_llm), from_template=len(shoppers) - len(from_llm))
    print(f"saved {len(shoppers)} shoppers ({len(from_llm)} written by the LLM) and {len(orders)} orders to {PROCESSED}")
    print("edge cases:", json.dumps(edge, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="template names, no LLM calls")
    main(ap.parse_args().offline)
