"""Small tie-break from the shopper's OWN order history, always stated as an assumption. Owner: M3.

v0: pass-through.
TODO(M3): repository.get_orders_for(customer_id) -> favourite brands/sizes -> small score bonus;
          add an assumption string such as "Using size 9 from your last order".
          Never call this with a customer ID taken from the message text.
"""
from __future__ import annotations

from app.schemas import Product


def apply(ranked: list[tuple[Product, float]], customer_id: str) -> tuple[list[tuple[Product, float]], list[str]]:
    return ranked, []
