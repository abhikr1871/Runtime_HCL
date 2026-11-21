"""Check the generated data against the schema and the brief's minimums. Owner: M5.

Run:  python data_kit/validate_data.py      (exit code 1 if any check fails)
The output is saved to data/processed/validation_report.txt - paste it into DATA_NOTE.md for the submission.
"""
from __future__ import annotations

import re
from collections import Counter

from common import PROCESSED, load_catalogue, read_json
from app.schemas import Product  # noqa: E402

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ID_RE = re.compile(r"^P-\d{4}$")


def main() -> int:
    problems: list[str] = []
    info: list[str] = []

    def check(ok: bool, msg: str) -> None:
        (info if ok else problems).append(("PASS: " if ok else "FAIL: ") + msg)

    # ---------- products ----------
    products = load_catalogue()
    ids = [p.get("product_id") for p in products]
    invalid = []
    for p in products:
        try:
            Product(**p)
        except Exception:
            invalid.append(p.get("product_id"))
    check(not invalid, f"all products match the schema ({len(invalid)} invalid: {invalid[:5]})")
    dup = [i for i, c in Counter(ids).items() if c > 1]
    check(not dup, f"product IDs unique ({len(dup)} duplicates)")
    check(all(ID_RE.match(str(i)) for i in ids), "product IDs look like P-0001")
    check(len(products) >= 250, f"{len(products)} products (brief minimum 250)")
    cats = Counter(p["category"] for p in products)
    check(len(cats) >= 5, f"{len(cats)} categories (brief minimum 5)")
    check(all((p.get("price_inr") or 0) > 0 for p in products if p.get("price_inr") is not None), "prices positive")
    check(all(p.get("stock", 0) >= 0 for p in products), "stock never negative")
    check(all(p.get("rating") is None or 0 <= p["rating"] <= 5 for p in products), "ratings between 0 and 5")

    # ---------- planted mess ----------
    mess = read_json(PROCESSED / "mess_log.json", {}).get("summary", {})
    for key, minimum in [("weight_in_kg", 6), ("weight_missing_unit", 3), ("brand_missing", 5), ("out_of_stock", 8),
                         ("low_stock", 8), ("injection_review", 3), ("near_duplicate_groups", 5)]:
        check(mess.get(key, 0) >= minimum, f"mess '{key}': {mess.get(key, 0)} (want {minimum}+)")
    non_en = [p for p in products if p.get("description_lang") in {"hi", "hinglish"}]
    check(len(non_en) >= 8, f"{len(non_en)} Hindi/Hinglish descriptions (want 8+)")
    bad_hi = [p["product_id"] for p in non_en if (p["description_lang"] == "hi") != bool(DEVANAGARI.search(p["description"]))]
    check(not bad_hi, f"Hindi descriptions in Devanagari, Hinglish in Roman script ({bad_hi[:5]})")

    # ---------- reviews ----------
    id_set = set(ids)
    reviews = [r for f in sorted(PROCESSED.glob("*_reviews.json")) for r in read_json(f, [])]
    orphan = [r["review_id"] for r in reviews if r["product_id"] not in id_set]
    check(not orphan, f"every review points to a real product ({len(orphan)} orphans)")
    check(sum(1 for r in reviews if r.get("lang") in {"hi", "hinglish"}) >= 10, "10+ Hindi/Hinglish reviews")

    # ---------- shoppers and orders ----------
    shoppers = read_json(PROCESSED / "shoppers.json", [])
    orders = read_json(PROCESSED / "orders.json", [])
    edge = read_json(PROCESSED / "edge_cases.json", {})
    check(len(shoppers) >= 25, f"{len(shoppers)} shoppers (brief minimum 25)")
    langs = Counter(s["preferred_language"] for s in shoppers)
    check(len(langs) >= 2, f"shopper languages: {dict(langs)}")
    bad_orders = [o["order_id"] for o in orders if o["product_id"] not in id_set]
    check(not bad_orders, f"every order points to a real product ({len(bad_orders)} bad)")
    shopper_ids = {s["customer_id"] for s in shoppers}
    check(all(o["customer_id"] in shopper_ids for o in orders), "every order belongs to a real shopper")
    per = Counter(o["customer_id"] for o in orders)
    new_shopper = next((cid for cid, why in edge.items() if "no orders" in why), None)
    thin = [cid for cid in shopper_ids if cid != new_shopper and per[cid] < 3]
    check(not thin, f"every shopper except the new one has 3+ orders ({thin[:5]})")
    check(new_shopper is not None and per[new_shopper] == 0, f"edge case: new shopper with no orders ({new_shopper})")
    check(len(edge) >= 3, f"3 edge-case shoppers recorded: {list(edge)}")
    check(all(not re.search(r"@|\d{10}", s["display_name"]) for s in shoppers), "no emails or phone numbers in shopper data")

    lines = [f"products={len(products)} categories={len(cats)} reviews={len(reviews)} shoppers={len(shoppers)} "
             f"orders={len(orders)}", *info, *problems, "OK" if not problems else f"{len(problems)} problem(s)"]
    report = "\n".join(lines)
    print(report)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "validation_report.txt").write_text(report + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
