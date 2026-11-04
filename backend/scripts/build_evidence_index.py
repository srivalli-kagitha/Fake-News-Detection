"""Build (or rebuild) the FAISS evidence index.

Usage (from the backend folder, with the venv active):
    python -m scripts.build_evidence_index

The backend also builds the index automatically the first time claim verification runs,
and rebuilds it whenever a .jsonl file in data/evidence changes. Use this script to do it ahead of time.
"""
import shutil

from app.config import get_settings
from app.services.evidence import INDEX_DIRNAME, EvidenceRetriever


def main() -> None:
    settings = get_settings()
    index_dir = settings.evidence_path / INDEX_DIRNAME
    if index_dir.exists():
        shutil.rmtree(index_dir)
    retriever = EvidenceRetriever()
    print(f"Index built: {retriever.index.ntotal} passages -> {index_dir}")


if __name__ == "__main__":
    main()
