"""Replay how any answer was produced (GET /trace/{id}). Owner: M4."""
import streamlit as st

import api_client

st.title("Trace viewer")
trace_id = st.text_input("Trace ID")
if trace_id:
    t = api_client.get_trace(trace_id.strip())
    if not t:
        st.error("Trace not found")
    else:
        rec = t["record"]
        st.write(f"**Input (redacted):** {rec.get('input')}")
        st.write(f"**Model:** {rec.get('model')} · **Timings (ms):** {rec.get('timings_ms')}")
        st.write(f"**Flags:** {rec.get('flags')}")
        st.subheader("Understood slots")
        st.json(rec.get("slots", {}), expanded=False)
        st.subheader("Retrieval")
        st.json(rec.get("retrieval", {}), expanded=False)
        st.subheader("Validator")
        st.json(rec.get("validator", {}), expanded=False)
        st.subheader("Response")
        st.json(rec.get("response", {}), expanded=False)
