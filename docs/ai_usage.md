# AI usage disclosure

_Required by the brief. Keep it honest and specific; update it before the code freeze._

| Part | AI tool used | How we verified it |
| --- | --- | --- |
| Project scaffold, API contracts, pipeline v0, tests | Claude (coding assistant) | Reviewed by the team; `pytest -q` green; each owner can explain their module |
| Data generation prompts | _model name_ | Schema validation (`validate_data.py`), manual spot checks |
| Hindi / Hinglish translations | _model name_ | Checked by a Hindi-speaking team member |
| _add rows as you go_ | | |

The runtime assistant uses a local LLM (`qwen2.5:7b-instruct` via Ollama) only to understand requests (fallback) and to word replies; all products, prices and stock come from code.
