"""Download models once, before building (run in the first 30 minutes on every laptop).

Run:  python scripts/download_models.py
Also run in a terminal:  ollama pull qwen2.5:7b-instruct
"""
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    MODELS.mkdir(exist_ok=True)
    m = SentenceTransformer("intfloat/multilingual-e5-small", cache_folder=str(MODELS))
    print("e5 ready, dimension:", m.get_sentence_embedding_dimension())
    print("Now run: ollama pull qwen2.5:7b-instruct")


if __name__ == "__main__":
    main()
