"""Shows the latest evaluation report written by eval/run_eval.py. Owner: M4 (report content: M5)."""
from pathlib import Path

import streamlit as st

st.title("Evaluation")
report = Path(__file__).resolve().parents[2] / "eval" / "report.md"
if report.exists():
    st.markdown(report.read_text(encoding="utf-8"))
else:
    st.info("No report yet. Run: python eval/run_eval.py")
