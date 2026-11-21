"""Step 5c: keep the top results varied (not three near-identical items). Owner: M2.

Prevents near-duplicate results:
  - Ensures brand diversity across top recommendation slots when alternatives exist
  - Avoids near-duplicate product titles / variants (e.g. 2 nearly identical HP laptop models)
  - Preserves relevance and hard constraints
  - Configurable diversity strategy
"""
from __future__ import annotations

import os
from typing import Optional

from app.config import get_settings
from app.schemas import Product


def _title_tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _is_near_duplicate(p: Product, picked: list[Product], threshold: float = 0.65) -> bool:
    """Checks if product p has high word overlap with any already picked product."""
    p_tokens = _title_tokens(p.name)
    if not p_tokens:
        return False
    for other in picked:
        # Same brand check: if same brand, use stricter name similarity check
        same_brand = bool(p.brand and other.brand and p.brand.lower() == other.brand.lower())
        other_tokens = _title_tokens(other.name)
        if not other_tokens:
            continue
        intersection = len(p_tokens & other_tokens)
        union = len(p_tokens | other_tokens)
        jaccard = intersection / union if union > 0 else 0.0

        if same_brand and jaccard >= 0.50:
            return True
        if jaccard >= threshold:
            return True
    return False


def diversify(
    ranked: list[tuple[Product, float]],
    k: int = 3,
    strategy: Optional[str] = None,
) -> list[Product]:
    """Selects top k diverse products avoiding duplicate brands and near-duplicate variants."""
    if not ranked:
        return []
    if len(ranked) <= k:
        return [p for p, _ in ranked]

    selected_strategy = (
        strategy
        or os.getenv("DIVERSITY_STRATEGY")
        or get_settings().diversity_strategy
        or "brand"
    ).lower()

    picked: list[Product] = []
    seen_brands: set[str] = set()

    # Pass 1: Select distinct brand AND non-near-duplicate products
    for p, _ in ranked:
        brand = (p.brand or "").strip().lower()
        if brand and brand in seen_brands:
            continue
        if _is_near_duplicate(p, picked):
            continue
        picked.append(p)
        if brand:
            seen_brands.add(brand)
        if len(picked) == k:
            return picked

    # Pass 2: Allow same brand if distinct variants (not near duplicates)
    for p, _ in ranked:
        if p in picked:
            continue
        if _is_near_duplicate(p, picked, threshold=0.75):
            continue
        picked.append(p)
        if len(picked) == k:
            return picked

    # Pass 3: Fill any remaining slots with highest remaining ranked products
    for p, _ in ranked:
        if p not in picked:
            picked.append(p)
            if len(picked) == k:
                break

    return picked
