"""GET /products/{id} - judges use it to check what we claim. Owner: M3."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import ProductDetail
from app.store import repository

router = APIRouter(tags=["products"])


@router.get("/products/{product_id}", response_model=ProductDetail)
def get_product(product_id: str) -> ProductDetail:
    p = repository.get_product(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return ProductDetail(**p.model_dump(), reviews=repository.get_reviews(product_id))
