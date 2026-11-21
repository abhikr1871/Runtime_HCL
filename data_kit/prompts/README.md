# Generation prompts (submitted verbatim)

The brief asks for the exact prompts used to generate data, word for word, plus the model that produced it.
The scripts **load these files directly**, so the text here is exactly what was sent (the `{...}` placeholder is
replaced by the JSON batch shown in the script). The model, call counts and tokens of every run are recorded in
`data/processed/generation_log.json`.

| File | Used by | Model (fill in from generation_log.json) |
| --- | --- | --- |
| `products.md` | `generate_products.py` | |
| `translations.md` | `plant_mess.py` (Hindi / Hinglish descriptions) | |
| `shoppers.md` | `generate_shoppers.py` | |

If a prompt changes during the day, copy the old one to `products_v1.md` etc. before editing, and note which version produced the final data.
