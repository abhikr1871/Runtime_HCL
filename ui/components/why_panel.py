"""'Why these results?' panel: the search funnel and sources, straight from the response/trace. Owner: M4."""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_why(resp: dict[str, Any]) -> None:
    funnel = resp.get("funnel")
    with st.expander("Why these results?"):
        if funnel:
            st.write(f"{funnel['total']} products → {funnel['after_filters']} after your filters → {funnel['shown']} shown")
        if resp.get("active_filters"):
            st.json(resp["active_filters"], expanded=False)
        for a in resp.get("assumptions", []):
            st.info(a)
        if resp.get("relaxation"):
            st.warning(resp["relaxation"]["message"])
        st.caption(f"Trace ID: {resp['trace_id']} (open it in the Trace Viewer page)")
