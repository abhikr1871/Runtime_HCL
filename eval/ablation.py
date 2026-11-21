"""Search ablation: BM25 vs embeddings vs hybrid vs hybrid + filter-first. Owner: M5 (with M2).

For each configuration, run the relevance cases and report relevance@3 and constraint accuracy.
This is our data-backed answer to "why hybrid search?" in the Q&A.

TODO(M5/M2): expose a config switch in pipeline/retrieval.py (e.g. RETRIEVAL_MODE=bm25|dense|hybrid)
and loop over it here, writing a table into eval/report.md.
"""


def main() -> None:
    raise SystemExit("TODO: implement after retrieval supports RETRIEVAL_MODE")


if __name__ == "__main__":
    main()
