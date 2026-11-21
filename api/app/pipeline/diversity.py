"""Step 5c: keep the top results varied (not three near-identical items). Owner: M2.

v0: prefer a different brand for each slot when an alternative exists.
TODO(M2): MMR over embeddings if time allows.
"""
from __future__ import annotations

from app.schemas import Product


def diversify(ranked: list[tuple[Product, float]], k: int = 3) -> list[Product]:
    picked: list[Product] = []
    seen_brands: set[str] = set()
    for p, _ in ranked:
        brand = (p.brand or "").lower()
        if brand and brand in seen_brands:
            continue
        picked.append(p)
        seen_brands.add(brand)
        if len(picked) == k:
            return picked
    for p, _ in ranked:  # not enough distinct brands: fill with the best remaining
        if p not in picked:
            picked.append(p)
            if len(picked) == k:
                break
    return picked
