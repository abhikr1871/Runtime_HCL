"""Active-filter chips: shows what the assistant remembers; click to remove one. Owner: M4."""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st

_HIDDEN = {"query_text", "sort"}
_LABELS = {"budget_max": "under ₹{}", "budget_min": "over ₹{}", "brands_include": "only {}", "brands_exclude": "not {}",
           "size": "size {}", "color": "{}", "category": "{}"}


def render_chips(filters: dict[str, Any], on_remove: Callable[[str], None]) -> None:
    shown = {k: v for k, v in filters.items() if k not in _HIDDEN and v}
    if not shown:
        return
    st.caption("Remembered filters (click to remove)")
    cols = st.columns(len(shown))
    for col, (k, v) in zip(cols, shown.items()):
        text = _LABELS.get(k, k + ": {}").format(", ".join(v) if isinstance(v, list) else v)
        if col.button(f"{text}  ✕", key=f"chip-{k}"):
            on_remove(k)
            st.rerun()
