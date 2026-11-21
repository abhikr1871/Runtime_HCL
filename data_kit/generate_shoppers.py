"""Generate 30 synthetic shoppers and 3+ orders each. Owner: M5.

Out: data/processed/shoppers.json, data/processed/orders.json

- LLM writes personas: display_name, city, preferred_language (en/hi/hinglish), budget band,
  favourite brands, sizes, interests. Save the exact prompt in data_kit/prompts/shoppers.md.
- CODE picks order product IDs from the real catalogue, so every order is valid.
- Deliberate edge cases (record their IDs in DATA_NOTE.md):
    * a brand-new shopper with no orders
    * a shopper with contradictory tastes (premium brands, tiny budget)
    * a shopper whose favourite products are now out of stock
- No real names, phones or emails. Customer IDs: C-001 ... C-030.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit("TODO(M5): implement generate_shoppers.py")


if __name__ == "__main__":
    main()
