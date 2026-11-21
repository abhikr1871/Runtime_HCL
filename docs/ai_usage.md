# AI usage disclosure

_Required by the brief. Keep it honest and specific; update it before the code freeze._

| Part | AI tool used | How we verified it |
| --- | --- | --- |
| Project scaffold, API contracts, pipeline v0, tests | Claude (coding assistant) | Reviewed by the team; `pytest -q` green; each owner can explain their module |
| Generated product text, shopper personas | Groq `openai/gpt-oss-20b` (prompts in `data_kit/prompts/`) | Pydantic validation per item, template fallback, `validate_data.py`, manual spot checks |
| Hindi / Hinglish translations | Groq `openai/gpt-oss-20b` | Script check (Devanagari vs Roman), checked by a Hindi-speaking team member |
| _add rows as you go_ | | |

The runtime assistant uses Groq `openai/gpt-oss-20b` (organiser approval: _confirm_) only to understand requests (fallback) and to word replies; all products, prices and stock come from code. With `MOCK_LLM=true` it runs with no LLM at all.
