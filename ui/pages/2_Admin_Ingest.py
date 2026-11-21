"""Add or update products live (POST /ingest) - shows R11 working during the demo. Owner: M4."""
import json

import streamlit as st

import api_client

st.title("Admin: add or update products")
st.caption("Upload a JSON list of products with the same fields as GET /products/{id}. Changes are searchable immediately.")
uploaded = st.file_uploader("Products JSON", type=["json"])
example = [{"product_id": "P-9200", "name": "Camping Lantern", "category": "sports-accessories", "brand": "Coleman",
            "price_inr": 1599, "stock": 7, "rating": 4.4, "tags": ["camping", "trekking", "lantern"]}]
text = st.text_area("...or paste JSON", value=json.dumps(example, indent=2), height=220)

if st.button("Ingest"):
    try:
        rows = json.loads(uploaded.read()) if uploaded else json.loads(text)
        st.success(api_client.ingest(rows if isinstance(rows, list) else [rows]))
    except Exception as exc:  # show any parse/API error to the admin
        st.error(str(exc))
