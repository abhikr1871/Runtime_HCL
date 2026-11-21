# M1: AI lead (understanding, answer, validator, integration)

**Goal:** the assistant understands English/Hindi/Hinglish requests and every reply is correct, worded well and verified.

**You own:** `api/app/pipeline/nlu_rules.py`, `nlu_llm.py`, `clarify.py`, `relax.py`, `reasons.py`, `composer.py`, `validator.py`, `orchestrator.py`, `api/app/ai/llm_client.py`, `api/app/ai/prompts/*`

## Tasks

**0:50-2:00**
- [ ] Ollama running with `qwen2.5:7b-instruct`; set `MOCK_LLM=false` and check `/health` says `llm: ok`
- [ ] `nlu_rules.py`: typo-tolerant brands with `rapidfuzz` ("nkie" → Nike); more Hinglish price phrases ("3 hazaar tak")
- [ ] `nlu_llm.py`: retry once with the validation error before falling back to rules
- [ ] Try 20 real messages (incl. Hinglish) and fix what breaks

**2:00-3:30**
- [ ] `clarify.py`: per-category required details; ask about the detail whose values vary most among matches
- [ ] `relax.py`: counterfactual hint ("₹500 more gives 4 options")
- [ ] `reasons.py`: add matched tags/attributes ("waterproof", "for trekking")
- [ ] `composer.py`: tune the prompt so replies are short and in the shopper's language and script
- [ ] `validator.py`: check stock claims; `orchestrator.py`: one regeneration before the template fallback

**3:30-4:30 (stretch, in order)**
- [ ] Bundle builder ("trekking kit under ₹5,000"), solved in code
- [ ] Review highlights cited by review ID

## Done when
- `pytest -q` green, including new tests in `tests/test_units.py` for each rule you add
- 10 sample queries return zero invented prices (check `validator` in the trace)
- With the LLM off (`MOCK_LLM=true`) every request still gets a correct answer

## Be ready to explain
Why rules first and the LLM second; how the validator works; why the LLM has no tools; what happens when the LLM is down.
