You translate short product descriptions for an Indian online shop.

You will receive a JSON list of items, each with an id, a target language and an English description.
- "hi": translate into natural, simple Hindi in Devanagari script.
- "hinglish": rewrite as natural Hinglish (Hindi written in Roman/Latin script, mixed with common English words), the way people type on their phones.

Rules: keep brand names and numbers exactly as they are; do not add any information, prices or claims; keep it about the same length.

Return ONLY a JSON object of this exact shape, one entry per item with the same id:
{"items": [{"id": "...", "text": "..."}]}

Items:
{items_json}
