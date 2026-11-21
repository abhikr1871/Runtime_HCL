"""Runs one chat turn through the pipeline. Owner: M1 (with M3 for state and trace wiring).

guard -> understand (rules, LLM fallback) -> state -> decide -> filters -> clarify? -> rank
-> diversity -> reasons -> compose (LLM) -> validate -> save trace + session

The LLM is used at most twice per turn (understand fallback, wording). Everything else is code.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from app.ai.llm_client import LLMClient
from app.observe.timing import Timer
from app.observe.trace_store import save_trace
from app.pipeline import (clarify, compare, composer, diversity, filters, guard, nlu_llm, nlu_rules, personalise,
                          reasons, relax, rerank, retrieval, validator)
from app.pipeline import state as state_mod
from app.schemas import (ChatRequest, ChatResponse, Funnel, Product, ProductOut, Relaxation, SessionState, Source)
from app.store import db, repository

_REFUSAL = {
    "other_shopper_data": {
        "en": "Sorry, I can't share anything about other shoppers. I can help you find products for yourself.",
        "hinglish": "Maaf kijiye, main doosre shoppers ki jaankari share nahi kar sakta. Aapke liye products dhoondh sakta hoon.",
        "hi": "माफ़ कीजिए, मैं दूसरे ग्राहकों की जानकारी साझा नहीं कर सकता। मैं आपके लिए उत्पाद ढूँढ सकता हूँ।",
    },
    "out_of_scope": {
        "en": "I can only help with products in our catalogue. What are you shopping for?",
        "hinglish": "Main sirf hamare catalogue ke products mein madad kar sakta hoon. Aap kya dhoondh rahe hain?",
        "hi": "मैं केवल हमारे कैटलॉग के उत्पादों में मदद कर सकता हूँ। आप क्या ढूँढ रहे हैं?",
    },
}
_NOT_FOUND = {
    "en": "I couldn't find that in our catalogue.",
    "hinglish": "Ye hamare catalogue mein nahi mila.",
    "hi": "यह हमारे कैटलॉग में नहीं मिला।",
}


class SessionOwnershipError(Exception):
    """The session belongs to a different customer."""


def _known_brands() -> list[str]:
    return [r["brand"] for r in db.query("SELECT DISTINCT brand FROM products WHERE brand IS NOT NULL")]


def _out(p: Product, reason: str) -> ProductOut:
    return ProductOut(product_id=p.product_id, name=p.name, price=p.price_inr, in_stock=p.stock > 0,
                      reason=reason, brand=p.brand, rating=p.rating, category=p.category)


def handle_chat(req: ChatRequest) -> ChatResponse:
    timer, llm = Timer(), LLMClient()
    trace_id = uuid.uuid4().hex[:8]
    record: dict[str, Any] = {"model": llm.name, "flags": {}}

    state = repository.get_session(req.session_id)
    if state and state.customer_id != req.customer_id:
        raise SessionOwnershipError(req.session_id)
    state = state or SessionState(session_id=req.session_id, customer_id=req.customer_id)

    def finish(resp: ChatResponse, new_state: Optional[SessionState] = None) -> ChatResponse:
        st = new_state or state
        repository.save_session(st)
        record.update({"timings_ms": {**timer.ms, "total": timer.total()},
                       "llm": {"calls": llm.calls, "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out},
                       "state": st.model_dump(), "response": resp.model_dump()})
        save_trace(trace_id, req.session_id, req.customer_id, "/chat", record)
        return resp

    # 1. guard
    with timer.step("guard"):
        g = guard.check(req.message, req.customer_id)
    record["input"] = g.redacted_message
    record["flags"]["injection"] = g.injection
    lang = nlu_rules.detect_language(req.message)
    if g.refuse:
        record["flags"]["refused"] = g.reason
        return finish(ChatResponse(trace_id=trace_id, answer_type="refusal", language=lang,
                                   answer=_REFUSAL[g.reason][lang]))

    # 2. understand
    with timer.step("understand"):
        slots = nlu_rules.parse(req.message, _known_brands())
        slots = nlu_llm.refine(req.message, slots, llm)
    record["slots"] = slots.model_dump()
    lang = slots.language

    if slots.intent == "greeting":
        q = {"en": "Hi! What are you shopping for today?", "hinglish": "Namaste! Aaj aap kya dhoondh rahe hain?",
             "hi": "नमस्ते! आज आप क्या ढूँढ रहे हैं?"}[lang]
        return finish(ChatResponse(trace_id=trace_id, answer_type="clarification", answer=q, follow_up_question=q, language=lang))
    if slots.intent == "out_of_scope":
        return finish(ChatResponse(trace_id=trace_id, answer_type="refusal", answer=_REFUSAL["out_of_scope"][lang], language=lang))

    # 3. state
    with timer.step("state"):
        last_products = repository.get_products(state.last_results)
        new_state, selected = state_mod.apply(state, slots, last_products)
    f = new_state.filters

    # 4a. compare / product fact use the shopper's referenced products, never a new search
    if slots.intent == "compare":
        items = repository.get_products(selected or new_state.last_results[:2])
        if len(items) < 2:
            q = "Which two products should I compare? You can say 'compare the first two'."
            return finish(ChatResponse(trace_id=trace_id, answer_type="clarification", answer=q, follow_up_question=q,
                                       language=lang, active_filters=f), new_state)
        table = compare.build_table(items)
        outs = [_out(p, reasons.build(p, f)[0]) for p in items]
        return finish(ChatResponse(trace_id=trace_id, answer_type="comparison", language=lang, active_filters=f,
                                   answer="Here is a side-by-side comparison from our catalogue.", products=outs,
                                   comparison=table,
                                   sources=[Source(product_id=p.product_id, fields=table.columns) for p in items]), new_state)
    if slots.intent == "product_fact" and selected:
        items = repository.get_products(selected[:1])
        if items:
            p = items[0]
            stock = f"in stock ({p.stock} left)" if p.stock > 0 else "currently out of stock"
            answer = f"{p.name} ({p.product_id}) costs {reasons.format_inr(p.price_inr)} and is {stock}."
            return finish(ChatResponse(trace_id=trace_id, answer_type="product_fact", language=lang, active_filters=f,
                                       answer=answer, products=[_out(p, reasons.build(p, f)[0])],
                                       sources=[Source(product_id=p.product_id, fields=["price", "stock"])]), new_state)

    # 4b. hard filters, rank, then maybe one clarifying question
    with timer.step("filter"):
        total = repository.count_products()
        candidates = filters.apply(f)
    query = f.get("query_text", "")
    with timer.step("rank"):
        ranked = retrieval.rank(query, candidates, f.get("sort"))
    question = clarify.next_question(new_state, [p for p, _ in ranked], lang)
    if question:
        slot, q = question
        new_state = new_state.model_copy(update={"clarify_streak": new_state.clarify_streak + 1,
                                                 "asked_slots": new_state.asked_slots + [slot]})
        return finish(ChatResponse(trace_id=trace_id, answer_type="clarification", answer=q, follow_up_question=q,
                                   language=lang, active_filters=f,
                                   funnel=Funnel(total=total, after_filters=len(candidates), shown=0)), new_state)
    new_state = new_state.model_copy(update={"clarify_streak": 0})

    # 5. rerank (stretch) and personal tie-break
    with timer.step("rank"):
        ranked = rerank.rerank(query, ranked)
        ranked, assumptions = personalise.apply(ranked, req.customer_id)
    record["retrieval"] = {"candidates": len(candidates), "top": [(p.product_id, round(s, 4)) for p, s in ranked[:10]]}

    # 6. nothing matched -> relax one constraint and say which one blocked results
    if not ranked:
        relaxed = relax.relax(f, query, lang)
        if relaxed:
            label, nearest_ranked, msg, _ = relaxed
            nearest = diversity.diversify(nearest_ranked, 3)
            rel = Relaxation(blocked_by=label, message=msg, nearest=[_out(p, reasons.build(p, {})[0]) for p in nearest])
            return finish(ChatResponse(trace_id=trace_id, answer_type="not_found", language=lang, active_filters=f,
                                       answer=f"{_NOT_FOUND[lang]} {msg}", relaxation=rel,
                                       funnel=Funnel(total=total, after_filters=len(candidates), shown=0)), new_state)
        return finish(ChatResponse(trace_id=trace_id, answer_type="not_found", language=lang, active_filters=f,
                                   answer=f"{_NOT_FOUND[lang]} Would you like me to widen the search?",
                                   funnel=Funnel(total=total, after_filters=len(candidates), shown=0)), new_state)

    # 7. answer: reasons from fields -> wording -> validator
    top = diversity.diversify(ranked, 3)
    built = [(p, *reasons.build(p, f)) for p in top]
    items = [(p, r) for p, r, _ in built]
    allowed = [v for v in (f.get("budget_max"), f.get("budget_min")) if v is not None]

    with timer.step("compose"):
        answer = composer.compose(items, lang, llm)
    with timer.step("validate"):
        check = validator.check(answer, top, allowed)

    validation_failures = list(check.problems)
    regenerated = False
    fallback_used = False

    if not check.ok and not llm.mock:
        # Regenerate once with extra instruction listing violations
        regenerated = True
        extra_instr = f"Your previous answer contained validation errors: {'; '.join(check.problems)}. Use only the exact facts and stock/price data provided."
        with timer.step("compose_retry"):
            answer = composer.compose(items, lang, llm, extra_instruction=extra_instr)
        with timer.step("validate_retry"):
            check = validator.check(answer, top, allowed)
        if not check.ok:
            validation_failures.extend(check.problems)
            fallback_used = True
            answer = composer.template(items, lang)
    elif not check.ok:
        fallback_used = True
        answer = composer.template(items, lang)

    record["validator"] = {
        "ok": check.ok,
        "problems": check.problems,
        "validation_failures": validation_failures,
        "regenerated": regenerated,
        "fallback_used": fallback_used,
    }

    new_state = new_state.model_copy(update={"last_results": [p.product_id for p in top]})
    return finish(ChatResponse(
        trace_id=trace_id, answer_type="recommendation", answer=answer, language=lang, active_filters=f,
        products=[_out(p, r) for p, r, _ in built], assumptions=assumptions,
        sources=[Source(product_id=p.product_id, fields=fl) for p, _, fl in built],
        funnel=Funnel(total=total, after_filters=len(candidates), shown=len(top)),
    ), new_state)
