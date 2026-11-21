"""Generate ~80 India-focused products. Owner: M5.

Run:   python data_kit/generate_products.py            (uses the LLM from .env, e.g. Groq)
       python data_kit/generate_products.py --offline  (templates only, no LLM)
       python data_kit/generate_products.py --fresh    (ignore earlier LLM output and redo everything)
Out:   data/raw/generated_products.json   (clean; plant_mess.py adds the deliberate mess)
       data/raw/generated_origin.json     (which products the LLM wrote vs templates)

Re-running is safe and cheap: products the LLM already wrote are kept, and only the
template-filled ones are sent to the LLM again (useful after a rate-limit or network failure).

Who decides what:
- CODE fixes every fact: product_id, category, brand, price_inr, stock, rating, sizes, colours, weight.
- The LLM writes only name, description, tags and attributes (prompt: data_kit/prompts/products.md).
- Every LLM item is validated; invalid or missing items are retried once, then filled from a template,
  so the script always produces a complete, valid catalogue.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from common import (PROCESSED, RAW, batched, get_llm, load_prompt, log_generation, price_point, read_json, rng,
                    write_json)
from app.schemas import Product  # noqa: E402  (path set up by common)

BATCH_SIZE = 4          # small batches fit Groq's free-tier limit of 8,000 tokens per minute
MAX_TOKENS = 2000       # reasoning models think first; 2,000 leaves room for ~4 listings
PAUSE_S = 2             # gentle pacing between batches
SHOE_SIZES = ["6", "7", "8", "9", "10", "11"]
CLOTH_SIZES = ["S", "M", "L", "XL", "XXL"]

# (category, product_type, variants per brand, [(brand, (min_price, max_price))], extras)
PLAN: list[tuple[str, str, int, list[tuple[str, tuple[int, int]]], dict[str, Any]]] = [
    ("trekking-gear", "trekking backpack", 2, [("Wildcraft", (1800, 3200)), ("Quechua", (1500, 2800)), ("Skybags", (1400, 2600)), ("Wildcraft", (3500, 5500))], {"weight": (700, 1400), "colors": ["black", "blue", "green", "grey"]}),
    ("trekking-gear", "trekking shoes", 2, [("Quechua", (2200, 3500)), ("Woodland", (3500, 5500)), ("Campus", (1500, 2400)), ("Columbia", (5500, 8000))], {"sizes": SHOE_SIZES, "weight": (700, 1100), "colors": ["brown", "grey", "black"]}),
    ("trekking-gear", "trekking poles", 1, [("Quechua", (800, 1500)), ("Wildcraft", (900, 1600))], {"weight": (450, 650)}),
    ("trekking-gear", "headlamp", 1, [("Wildcraft", (600, 1200)), ("Quechua", (500, 1000))], {"weight": (80, 150)}),
    ("trekking-gear", "insulated water bottle", 1, [("Milton", (500, 1100)), ("Cello", (400, 900)), ("Borosil", (700, 1400))], {"weight": (300, 550), "colors": ["silver", "black", "blue"]}),
    ("audio", "bluetooth speaker", 2, [("boAt", (999, 2499)), ("JBL", (2999, 7999)), ("Sony", (3499, 8999)), ("Zebronics", (699, 1799)), ("Mivi", (899, 1999))], {"weight": (250, 900), "colors": ["black", "blue", "red"]}),
    ("audio", "wireless earbuds", 2, [("boAt", (999, 2299)), ("Noise", (1199, 2999)), ("OnePlus", (2499, 4999)), ("Realme", (1299, 2999)), ("JBL", (2999, 6999))], {"weight": (40, 70), "colors": ["black", "white"]}),
    ("audio", "neckband earphones", 1, [("boAt", (799, 1599)), ("OnePlus", (1599, 2299)), ("Realme", (999, 1799))], {"weight": (25, 40), "colors": ["black", "blue"]}),
    ("running-shoes", "running shoes", 2, [("Nike", (3500, 8500)), ("Adidas", (3000, 7500)), ("Puma", (2500, 5500)), ("ASICS", (4500, 9500)), ("Campus", (1200, 2200)), ("Skechers", (3000, 6000))], {"sizes": SHOE_SIZES, "weight": (220, 320), "colors": ["black", "white", "blue", "grey"]}),
    ("running-shoes", "running shoes for flat feet", 1, [("ASICS", (6500, 11000)), ("Nike", (6000, 10000)), ("Brooks", (7000, 12000))], {"sizes": SHOE_SIZES, "weight": (260, 340), "colors": ["black", "blue"]}),
    ("ethnic-wear", "men's cotton kurta", 2, [("Fabindia", (1200, 2500)), ("Manyavar", (1800, 4500)), ("Max", (600, 1200))], {"sizes": CLOTH_SIZES, "colors": ["white", "navy", "yellow", "green"]}),
    ("ethnic-wear", "women's kurti", 2, [("Biba", (900, 2200)), ("W", (1000, 2500)), ("Aurelia", (700, 1600))], {"sizes": CLOTH_SIZES, "colors": ["pink", "blue", "yellow", "white"]}),
    ("kitchen", "pressure cooker", 1, [("Prestige", (1500, 3500)), ("Hawkins", (1400, 3200)), ("Pigeon", (900, 1800))], {"weight": (1800, 3200)}),
    ("kitchen", "mixer grinder", 1, [("Bajaj", (2500, 4500)), ("Philips", (3500, 6500)), ("Prestige", (2800, 5000))], {"weight": (2500, 4000)}),
    ("kitchen", "non-stick tawa", 1, [("Prestige", (700, 1400)), ("Pigeon", (400, 900))], {"weight": (800, 1400)}),
]
_LABELS = ["Lite", "Pro", "Plus", "Max", "Classic", "Sport", "Neo", "Prime"]


class LLMItem(BaseModel):
    slot_id: str
    name: str = Field(min_length=3, max_length=80)
    description: str = Field(min_length=15, max_length=600)
    tags: list[str] = Field(min_length=2, max_length=10)
    attributes: dict[str, str] = {}


def next_id_start() -> int:
    base = read_json(PROCESSED / "dummyjson_products.json", [])
    nums = [int(p["product_id"].split("-")[1]) for p in base]
    if not nums:
        print("WARNING: run fetch_dummyjson.py first; numbering generated products from P-0195")
    return (max(nums) if nums else 194) + 1


def build_slots() -> list[dict[str, Any]]:
    r = rng("products")
    slots, n = [], next_id_start()
    for category, ptype, variants, brands, extra in PLAN:
        for brand, (lo, hi) in brands:
            prices = sorted(price_point(r.uniform(lo, hi)) for _ in range(variants))
            for price in prices:
                slot: dict[str, Any] = {
                    "slot_id": f"P-{n:04d}", "category": category, "product_type": ptype, "brand": brand,
                    "price_inr": price, "stock": r.randint(4, 60), "rating": round(r.uniform(3.6, 4.8), 1),
                    "rating_count": r.randint(20, 2500),
                }
                if "sizes" in extra:
                    slot["sizes"] = sorted(r.sample(extra["sizes"], k=min(4, len(extra["sizes"]))), key=extra["sizes"].index)
                if "colors" in extra:
                    slot["colors"] = r.sample(extra["colors"], k=min(2, len(extra["colors"])))
                if "weight" in extra:
                    slot["weight_g"] = int(r.uniform(*extra["weight"]) // 10 * 10)
                slots.append(slot)
                n += 1
    return slots


def template_item(slot: dict[str, Any]) -> dict[str, Any]:
    label = _LABELS[int(slot["slot_id"][2:]) % len(_LABELS)]
    ptype = slot["product_type"]
    words = [w for w in re.findall(r"[a-z]+", ptype.lower()) if w not in {"for", "s", "men", "women"}]
    return {
        "slot_id": slot["slot_id"],
        "name": f"{slot['brand']} {ptype.title()} {label}",
        "description": f"A {ptype} from {slot['brand']} for everyday use, with a practical design that suits regular use at home, work or outdoors.",
        "tags": list(dict.fromkeys(words + [slot["category"].replace("-", " ")]))[:6],
        "attributes": {},
    }


def ask_llm(llm, prompt: str, batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    facts = [{k: v for k, v in s.items() if k in {"slot_id", "product_type", "brand", "price_inr", "sizes", "colors", "weight_g"}}
             for s in batch]
    data = llm.complete_json("You return only valid JSON.", prompt.replace("{slots_json}", json.dumps(facts, ensure_ascii=False)),
                             temperature=0.7, max_tokens=MAX_TOKENS)
    items = data.get("products", []) if isinstance(data, dict) else []
    wanted, good = {s["slot_id"] for s in batch}, {}
    for raw in items:
        try:
            item = LLMItem(**raw)
        except (ValidationError, TypeError):
            continue
        text = f"{item.name} {item.description}".lower()
        if item.slot_id in wanted and "₹" not in text and "rs." not in text:   # LLM must not invent prices
            good[item.slot_id] = item.model_dump()
    return good


def to_product(slot: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    p = Product(
        product_id=slot["slot_id"], source="generated", name=item["name"].strip(), category=slot["category"],
        subcategory=slot["product_type"], brand=slot["brand"], price_inr=slot["price_inr"], stock=slot["stock"],
        rating=slot["rating"], rating_count=slot["rating_count"], description=item["description"].strip(),
        attributes=item.get("attributes") or {}, sizes=slot.get("sizes", []), colors=slot.get("colors", []),
        tags=[t.lower().strip() for t in item["tags"]][:7],
        weight_raw=f"{slot['weight_g']} g" if slot.get("weight_g") else None,
    )
    return p.model_dump()


def previous_llm_items(slots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """LLM-written text from an earlier run, for slots whose code-owned facts are unchanged."""
    origin = read_json(RAW / "generated_origin.json", {})
    old = {p["product_id"]: p for p in read_json(RAW / "generated_products.json", [])}
    keep = {}
    for s in slots:
        p = old.get(s["slot_id"])
        if not p:
            continue
        # runs before the origin file existed: anything that is not template text came from the LLM
        was_llm = origin.get(s["slot_id"]) == "llm" if origin else \
            not p["description"].startswith(f"A {s['product_type']} from {s['brand']} for everyday use")
        if was_llm and p["brand"] == s["brand"] and p["price_inr"] == s["price_inr"]:
            keep[s["slot_id"]] = {"slot_id": s["slot_id"], "name": p["name"], "description": p["description"],
                                  "tags": p["tags"], "attributes": p.get("attributes", {})}
    return keep


def main(offline: bool, fresh: bool) -> None:
    llm = get_llm(offline)
    prompt = load_prompt("products.md")
    slots = build_slots()
    from_llm: dict[str, dict[str, Any]] = {} if fresh else previous_llm_items(slots)
    if from_llm:
        print(f"keeping {len(from_llm)} products the LLM wrote in an earlier run (use --fresh to redo them)")
    todo = [s for s in slots if s["slot_id"] not in from_llm]
    if llm and todo:
        batches = list(batched(todo, BATCH_SIZE))
        for i, batch in enumerate(batches, start=1):
            got = ask_llm(llm, prompt, batch)
            missing = [s for s in batch if s["slot_id"] not in got]
            if missing:  # one retry for the items that failed validation
                got.update(ask_llm(llm, prompt, missing))
            from_llm.update(got)
            print(f"batch {i}/{len(batches)}: {len(got)}/{len(batch)} from LLM")
            time.sleep(PAUSE_S)
    products = [to_product(s, from_llm.get(s["slot_id"]) or template_item(s)) for s in slots]
    write_json(RAW / "generated_products.json", products)
    write_json(RAW / "generated_origin.json", {s["slot_id"]: ("llm" if s["slot_id"] in from_llm else "template") for s in slots})
    n_llm = sum(1 for s in slots if s["slot_id"] in from_llm)
    log_generation("generate_products.py", llm, ["products.md"], products=len(products), from_llm=n_llm,
                   from_template=len(products) - n_llm)
    print(f"saved {len(products)} products ({n_llm} written by the LLM, {len(products) - n_llm} from templates) "
          f"to {RAW / 'generated_products.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="templates only, no LLM calls")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier LLM output and regenerate every product")
    args = ap.parse_args()
    main(args.offline, args.fresh)
