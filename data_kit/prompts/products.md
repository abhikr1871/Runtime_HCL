You write product listings for a fictional Indian online shop. All products are fictional; do not copy real product listings.

You will receive a JSON list of product slots. Each slot already has its FIXED facts: slot_id, product_type, brand, price_inr and optional sizes, colors and weight. Do not change, repeat or contradict these facts, and never mention any price, discount, delivery time or stock level.

For EACH slot write:
- "name": a short realistic product name that starts with the brand and fits the product type (max 60 characters). Invent model names; do not use real model numbers.
- "description": 25 to 45 words, plain English, what it is and who it suits. No prices, no emojis, no superlatives like "best ever".
- "tags": 4 to 7 lowercase search keywords (for example "trekking", "waterproof", "flat feet").
- "attributes": an object with 2 to 4 short factual-sounding features as key/value strings (for example {"capacity": "30 L", "material": "polyester"}).

Return ONLY a JSON object of this exact shape, with one entry per slot and the same slot_id values:
{"products": [{"slot_id": "...", "name": "...", "description": "...", "tags": ["..."], "attributes": {"...": "..."}}]}

Product slots:
{slots_json}
