"""GET /products (browse the catalogue) and GET /products/{id} (judges use it to check what we claim). Owner: M3."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas import Product, ProductDetail
from app.store import repository

router = APIRouter(tags=["products"])


@router.get("/products", response_model=list[Product])
def list_products(
    q: Optional[str] = Query(default=None, description="words to look for in name, brand, category, tags"),
    category: Optional[str] = None,
    brand: Optional[str] = None,
    max_price: Optional[int] = Query(default=None, ge=0),
    in_stock: bool = False,
    limit: int = Query(default=500, ge=1, le=1000),
) -> list[Product]:
    """Plain catalogue listing for the UI's catalogue page (no ranking, no LLM)."""
    items = repository.find_products()
    words = (q or "").lower().split()
    out = []
    for p in items:
        if category and p.category.lower() != category.lower():
            continue
        if brand and brand.lower() not in (p.brand or "").lower():
            continue
        if max_price is not None and (p.price_inr is None or p.price_inr > max_price):
            continue
        if in_stock and p.stock <= 0:
            continue
        if words:
            text = f"{p.name} {p.brand or ''} {p.category} {' '.join(p.tags)}".lower()
            if not all(w in text for w in words):
                continue
        out.append(p)
    out.sort(key=lambda p: (p.category, p.name))
    return out[:limit]


@router.get("/products/{product_id}", response_model=ProductDetail)
def get_product(product_id: str) -> ProductDetail:
    p = repository.get_product(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return ProductDetail(**p.model_dump(), reviews=repository.get_reviews(product_id))
