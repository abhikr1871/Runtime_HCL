"""Product cards. Owner: M4."""
from __future__ import annotations

from typing import Any

import streamlit as st


def _inr(v: Any) -> str:
    return "price not listed" if v is None else f"₹{v:,}"


def render_products(products: list[dict[str, Any]]) -> None:
    if not products:
        return
    cols = st.columns(min(3, len(products)))
    for col, p in zip(cols, products):
        with col.container(border=True):
            st.markdown(f"**{p['name']}**")
            st.caption(f"{p['product_id']} · {p.get('brand') or 'brand not listed'}")
            st.markdown(f"### {_inr(p.get('price'))}")
            st.write("In stock" if p.get("in_stock") else "Out of stock")
            st.caption(p.get("reason", ""))


def render_comparison(table: dict[str, Any] | None) -> None:
    if table:
        st.dataframe(table["rows"], use_container_width=True, hide_index=True)
