"""Streamlit chat UI. Run:  streamlit run ui/app.py   Owner: M4."""
from __future__ import annotations

import uuid

import httpx
import streamlit as st

import api_client
from components.cards import render_comparison, render_products
from components.chips import render_chips
from components.why_panel import render_why

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️", layout="wide")

# ---- sidebar: who is shopping ----
with st.sidebar:
    st.header("Shopper")
    customer_id = st.text_input("Customer ID", value="C-001", help="Sent as customer_id; never read from the message.")
    if "session_id" not in st.session_state or st.button("New conversation"):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.history = []
    st.caption(f"Session: {st.session_state.session_id}")
    h = api_client.health()
    st.caption(f"API: {h['status']} · {h['products']} products · LLM {h['llm']}" if h else "API: not reachable")

st.title("AI Shopping Assistant")
st.caption("Ask in English, Hindi or Hinglish, e.g. “gift for my father who likes trekking, under ₹3,000”.")

session = api_client.get_session(st.session_state.session_id, customer_id)
if session:
    render_chips(session["active_filters"],
                 lambda name: api_client.remove_filter(st.session_state.session_id, customer_id, name))

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("resp"):
            render_products(turn["resp"].get("products", []))
            render_comparison(turn["resp"].get("comparison"))
            render_why(turn["resp"])

if message := st.chat_input("What are you looking for?"):
    st.session_state.history.append({"role": "user", "text": message})
    try:
        resp = api_client.chat(st.session_state.session_id, customer_id, message)
        st.session_state.history.append({"role": "assistant", "text": resp["answer"], "resp": resp})
    except httpx.HTTPStatusError as exc:
        st.session_state.history.append({"role": "assistant", "text": f"Error {exc.response.status_code}: {exc.response.text}"})
    except httpx.HTTPError:
        st.session_state.history.append({"role": "assistant", "text": "The API is not reachable. Is it running on port 8000?"})
    st.rerun()
