"""Evidence retrieval.

1. Local evidence corpus (backend/data/evidence/*.jsonl) indexed with BGE-M3 embeddings + FAISS.
   Each line: {"text": "...", "source": "...", "url": "..."}
2. Optional: Google Fact Check Tools API (free key, set GOOGLE_FACTCHECK_API_KEY in .env).
3. Live Wikipedia and news search, in live_sources.py (no key).
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import requests

from ..config import get_settings
from .common import Lazy, get_device, logger

INDEX_DIRNAME = "index"


def _corpus_files(evidence_dir: Path) -> list[Path]:
    return sorted(p for p in evidence_dir.glob("*.jsonl") if p.is_file())


def _corpus_fingerprint(files: list[Path], embed_model: str) -> str:
    h = hashlib.sha256(embed_model.encode())
    for f in files:
        stat = f.stat()
        h.update(f"{f.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()


def _read_corpus(files: list[Path]) -> list[dict]:
    docs = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping bad JSON in %s line %d", f.name, line_no)
                    continue
                if item.get("text"):
                    docs.append(
                        {
                            "text": str(item["text"]).strip(),
                            "source": item.get("source", f.stem),
                            "url": item.get("url", ""),
                        }
                    )
    return docs


class EvidenceRetriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.settings = get_settings()
        self.encoder = SentenceTransformer(self.settings.embed_model, device=str(get_device()))
        self.dir = self.settings.evidence_path
        self.index, self.docs = self._load_or_build()

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.encoder.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)

    def _load_or_build(self):
        files = _corpus_files(self.dir)
        index_dir = self.dir / INDEX_DIRNAME
        fingerprint = _corpus_fingerprint(files, self.settings.embed_model)
        info_file = index_dir / "info.json"

        if info_file.exists():
            info = json.loads(info_file.read_text(encoding="utf-8"))
            if info.get("fingerprint") == fingerprint:
                index = faiss.read_index(str(index_dir / "faiss.index"))
                with (index_dir / "docs.jsonl").open(encoding="utf-8") as fh:
                    docs = [json.loads(line) for line in fh if line.strip()]
                logger.info("Loaded evidence index with %d passages.", len(docs))
                return index, docs

        docs = _read_corpus(files)
        logger.info("Building evidence index for %d passages (one-time)...", len(docs))
        dim = self.encoder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        if docs:
            index.add(self.encode([d["text"] for d in docs]))

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "faiss.index"))
        with (index_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        info_file.write_text(json.dumps({"fingerprint": fingerprint, "count": len(docs)}), encoding="utf-8")
        return index, docs

    def search(self, queries: list[str], top_k: int) -> list[list[dict]]:
        if not queries:
            return []
        if self.index.ntotal == 0:
            return [[] for _ in queries]
        scores, ids = self.index.search(self.encode(queries), min(top_k, self.index.ntotal))
        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for score, idx in zip(row_scores, row_ids):
                if idx < 0:
                    continue
                hits.append({**self.docs[idx], "similarity": float(score), "type": "corpus"})
            results.append(hits)
        return results


FALSE_RATING = re.compile(
    r"\b(false|fake|hoax|misleading|incorrect|wrong|not true|untrue|baseless|fabricated|manipulated|"
    r"doctored|altered|edited|scam|satire|distorted|no evidence|unproven|unsupported|debunked|"
    r"pants on fire|misinformation|disinformation|rumou?r|myth|exaggerated|out of context|"
    r"అబద్ధం|తప్పు|నకిలీ|ఫేక్|తప్పుదోవ)", re.I)
TRUE_RATING = re.compile(r"^\s*(true|correct|accurate|mostly true|real|నిజం)\b", re.I)


def rating_polarity(rating: str) -> str | None:
    """'false' or 'true' for a clear fact-checker rating, None when it is unclear."""
    if FALSE_RATING.search(rating or ""):
        return "false"
    if TRUE_RATING.search(rating or ""):
        return "true"
    return None


def search_fact_checks(claim: str, language: str, limit: int = 3) -> list[dict]:
    key = get_settings().google_factcheck_api_key.strip()
    if not key:
        return []
    try:
        return [dict(d) for d in _fact_checks(claim[:300], language, limit, key)]
    except RuntimeError:
        return []


@lru_cache(maxsize=512)
def _fact_checks(claim: str, language: str, limit: int, key: str) -> tuple:
    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": claim, "key": key, "languageCode": language, "pageSize": limit},
            timeout=8,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Fact Check API request failed: %s", exc)
        raise RuntimeError("fact check failed") from exc  # not cached, so the next check retries

    evidence = []
    for item in resp.json().get("claims", [])[:limit]:
        for review in item.get("claimReview", [])[:1]:
            publisher = review.get("publisher", {}).get("name", "A fact-checker")
            rating = review.get("textualRating", "unrated")
            evidence.append(
                {
                    "text": f'{publisher} fact-checked the claim "{item.get("text", "")}" and rated it: {rating}.',
                    "source": f"Fact check - {publisher}",
                    "url": review.get("url", ""),
                    "similarity": None,
                    "type": "fact_check",
                    "rating": rating,
                    "claim_reviewed": item.get("text", ""),
                }
            )
    return tuple(evidence)


_retriever = Lazy("Evidence retriever", EvidenceRetriever)


def get_retriever() -> EvidenceRetriever:
    return _retriever.get()


def evidence_status() -> dict:
    settings = get_settings()
    files = _corpus_files(settings.evidence_path)
    return {
        "corpus_files": [f.name for f in files],
        "loaded": _retriever.loaded,
        "passages": len(_retriever.get().docs) if _retriever.loaded else None,
        "fact_check_api": bool(settings.google_factcheck_api_key.strip()),
        "live_search": settings.live_search,
    }
