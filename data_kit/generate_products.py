"""Generate ~80 India-focused products with the LLM. Owner: M5.

Out: data/processed/generated_products.json

Rules (from the design):
- The LLM writes ONLY names, descriptions, tags, attributes (and some Hindi/Hinglish text).
- Code sets product_id (P-0195 onwards), price_inr, stock and brand assignment.
- Every row is validated against app.schemas.Product (Pydantic); invalid rows are retried once, then dropped.
- Save the exact prompt in data_kit/prompts/products.md and the model name in DATA_NOTE.md.

Categories to cover: trekking gear, bluetooth speakers, earphones, running shoes (incl. flat feet),
kurtas, kitchen items. Include near-duplicates across brands at different prices.
"""
from __future__ import annotations


def main() -> None:
    # TODO(M5): call Ollama (or write by hand with a template if the LLM is slow), validate, save
    raise SystemExit("TODO(M5): implement generate_products.py")


if __name__ == "__main__":
    main()
