"""Small tie-break from the shopper's OWN history, always stated as an assumption. Owner: M3.

- Favourite brands = brands the shopper bought before + brands saved in their own profile.
- A favourite-brand product gets a small bonus (+10% of its score), so it wins only near-ties: relevance,
  the shopper's filters and their explicit wording always come first.
- When the bonus changes what is shown, the reply says so ("You've bought Nike before...").
- Called ONLY with the session owner's customer_id from the request, never an ID from the message text.
"""
from __future__ import annotations

import json

from app.schemas import Product
from app.store import db, repository

BONUS = 0.10


def _favourite_brands(customer_id: str) -> tuple[list[str], list[str]]:
    """(brands from own past orders, brands from own saved profile), most recent order first."""
    bought: list[str] = []
    for o in repository.get_orders_for(customer_id):
        p = repository.get_product(o["product_id"])
        if p and p.brand and p.brand not in bought:
            bought.append(p.brand)
    saved: list[str] = []
    rows = db.query("SELECT fav_brands_json FROM shoppers WHERE customer_id = ?", (customer_id,))
    if rows:
        try:
            saved = [b for b in json.loads(rows[0]["fav_brands_json"] or "[]") if isinstance(b, str)]
        except (ValueError, TypeError):
            saved = []
    return bought, saved


def apply(ranked: list[tuple[Product, float]], customer_id: str) -> tuple[list[tuple[Product, float]], list[str]]:
    if not ranked or not customer_id:
        return ranked, []
    try:
        bought, saved = _favourite_brands(customer_id)
    except Exception:
        return ranked, []
    favourites = {b.lower() for b in bought + saved}
    if not favourites:
        return ranked, []

    before = [p.product_id for p, _ in ranked[:3]]
    boosted = sorted(((p, s * (1 + BONUS) if (p.brand or "").lower() in favourites else s) for p, s in ranked),
                     key=lambda x: -x[1])
    shown = boosted[:3]
    promoted = [p for p, _ in shown if p.product_id not in before and (p.brand or "").lower() in favourites]
    if not promoted:
        return ranked, []          # nothing changed on screen: say nothing, keep the original order
    brand = promoted[0].brand
    source = "bought" if brand.lower() in {b.lower() for b in bought} else "saved"
    note = (f"You've bought {brand} before, so I gave {brand} a small preference."
            if source == "bought" else f"{brand} is in your saved favourite brands, so I gave it a small preference.")
    return boosted, [note]
