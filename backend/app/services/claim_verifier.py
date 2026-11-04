"""Contradiction-aware, claim-level evidence verification.

article -> claims -> evidence (local FAISS corpus + live Wikipedia/news + fact-check API)
        -> NLI(evidence, claim) -> Support / Contradiction / Unknown per claim -> overall assessment
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings
from .claim_extractor import extract_claims
from .common import Lazy, get_device
from .evidence import get_retriever, rating_polarity, search_fact_checks
from .live_sources import DEBUNK_HEADLINE, search_news, search_wikipedia

DECISION_THRESHOLD = 0.5
MAX_EVIDENCE = 5          # passages per claim sent to NLI (NLI is the slow step on a CPU)
SAME_CLAIM_SIMILARITY = 0.65  # a fact-check or headline must be this close to be about the same claim


class NLIModel:
    def __init__(self, model_name: str):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        names = {i: str(n).lower() for i, n in self.model.config.id2label.items()}
        self.index = {
            key: next(i for i, n in names.items() if n.startswith(key))
            for key in ("entailment", "neutral", "contradiction")
        }

    @torch.inference_mode()
    def predict(self, premises: list[str], hypotheses: list[str], batch_size: int = 8) -> list[dict]:
        results = []
        for s in range(0, len(premises), batch_size):
            enc = self.tokenizer(
                premises[s : s + batch_size],
                hypotheses[s : s + batch_size],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for row in probs:
                results.append({k: float(row[i]) for k, i in self.index.items()})
        return results


_settings = get_settings()
_nli = {
    "en": Lazy("English NLI model", lambda: NLIModel(_settings.nli_en_model)),
    "multi": Lazy("Multilingual NLI model", lambda: NLIModel(_settings.nli_multi_model)),
}


def get_nli(language: str) -> NLIModel:
    return _nli["en" if language == "en" else "multi"].get()


def _decide(evidence: list[dict]) -> tuple[str, float]:
    if not evidence:
        return "Unknown", 0.0
    best_support = max(e["nli"]["entailment"] for e in evidence)
    best_contra = max(e["nli"]["contradiction"] for e in evidence)
    if best_contra >= DECISION_THRESHOLD and best_contra >= best_support:
        return "Contradiction", best_contra
    if best_support >= DECISION_THRESHOLD:
        return "Support", best_support
    return "Unknown", 1.0 - max(best_support, best_contra)


def aggregate(claims: list[dict]) -> dict:
    counts = {"Support": 0, "Contradiction": 0, "Unknown": 0}
    total = 0.0
    for c in claims:
        counts[c["label"]] += 1
        if c["label"] == "Support":
            total += c["confidence"]
        elif c["label"] == "Contradiction":
            total -= c["confidence"]
    n = max(len(claims), 1)
    score = total / n  # -1 (all contradicted) .. +1 (all supported)

    if counts["Support"] == 0 and counts["Contradiction"] == 0:
        verdict = "Unverified"
    elif score <= -0.2:
        verdict = "Likely Misleading"
    elif score >= 0.2:
        verdict = "Likely Credible"
    else:
        verdict = "Mixed Evidence"

    return {
        "verdict": verdict,
        "credibility_score": round((score + 1) / 2, 4),  # 0..1
        "counts": counts,
        "total_claims": len(claims),
    }


def _gather_live(claims: list[str], language: str, settings) -> list[dict]:
    """Runs all network searches for all claims in parallel. Returns {source: [items]} per claim."""
    jobs = {"fact_check": search_fact_checks}
    if settings.live_search:
        jobs.update(wikipedia=search_wikipedia, news=search_news)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [{name: pool.submit(fn, c, language) for name, fn in jobs.items()} for c in claims]
        return [{name: f.result() for name, f in per_claim.items()} for per_claim in futures]


def _rank_live(retriever, claim_vec: np.ndarray, live: dict, settings, seen: set[str]) -> list[dict]:
    """Keeps the live items most similar to the claim (top LIVE_TOP_K per source, above the threshold)."""
    picked = []
    for name, items in live.items():
        items = [i for i in items if i["text"] not in seen]
        if not items:
            continue
        keys = [i.get("claim_reviewed") or i["text"] for i in items]
        sims = retriever.encode(keys) @ claim_vec
        top = settings.live_top_k if name != "fact_check" else 2
        for i, sim in sorted(zip(items, sims), key=lambda x: -x[1])[:top]:
            if sim >= settings.evidence_min_similarity:
                i["similarity"] = float(sim)
                seen.add(i["text"])
                picked.append(i)
    return picked


def _apply_ratings(nli, claim: str, evidence: list[dict]) -> None:
    """Published fact-checks and debunking headlines decide a claim they are clearly about.

    The NLI model cannot read 'X fact-checked "..." and rated it: False', or a headline like
    '... ? Don't fall for this scam', as a contradiction. So for a fact-check we ask NLI whether
    the user's claim and the reviewed claim say the same thing: same claim + rated false ->
    Contradiction, opposite claim + rated false -> Support. A news headline that is very close
    to the claim and calls it fake, false, a hoax or a scam counts as a contradiction.
    """
    checks = [e for e in evidence if e.get("type") == "fact_check" and rating_polarity(e.get("rating", ""))
              and e["similarity"] >= SAME_CLAIM_SIMILARITY]
    if checks:
        pairs = nli.predict([e["claim_reviewed"] for e in checks], [claim] * len(checks))
        for e, p in zip(checks, pairs):
            rated_false = rating_polarity(e["rating"]) == "false"
            if p["entailment"] >= DECISION_THRESHOLD or (e["similarity"] >= 0.8 and p["contradiction"] < DECISION_THRESHOLD):
                conf, agrees = max(p["entailment"], e["similarity"]), not rated_false
            elif p["contradiction"] >= DECISION_THRESHOLD:
                conf, agrees = p["contradiction"], rated_false
            else:
                continue
            e["nli"] = {"entailment": conf if agrees else 0.0, "neutral": 1.0 - conf,
                        "contradiction": 0.0 if agrees else conf}
            e["decided_by"] = "fact_check_rating"

    if DEBUNK_HEADLINE.search(claim):
        return
    for e in evidence:
        if (e.get("type") == "news" and e["similarity"] >= SAME_CLAIM_SIMILARITY + 0.05
                and DEBUNK_HEADLINE.search(e["text"]) and e["nli"]["entailment"] < DECISION_THRESHOLD):
            conf = max(e["nli"]["contradiction"], min(e["similarity"], 0.9))
            e["nli"] = {"entailment": 0.0, "neutral": 1.0 - conf, "contradiction": conf}
            e["decided_by"] = "debunking_headline"


def verify_text(text: str, language: str) -> dict:
    settings = get_settings()
    claims = extract_claims(text, language)
    retriever = get_retriever()
    corpus_hits = retriever.search(claims, settings.evidence_top_k)
    live_results = _gather_live(claims, language, settings) if claims else []
    claim_vecs = retriever.encode(claims) if claims else []
    nli = get_nli(language)

    results = []
    for claim, hits, live, vec in zip(claims, corpus_hits, live_results, claim_vecs):
        evidence = [h for h in hits if h["similarity"] >= settings.evidence_min_similarity]
        evidence += _rank_live(retriever, vec, live, settings, {e["text"] for e in evidence})
        evidence = sorted(evidence, key=lambda e: -e["similarity"])[:MAX_EVIDENCE]
        if evidence:
            scores = nli.predict([e["text"] for e in evidence], [claim] * len(evidence))
            for e, s in zip(evidence, scores):
                e["nli"] = s
            _apply_ratings(nli, claim, evidence)
            evidence.sort(key=lambda e: max(e["nli"]["entailment"], e["nli"]["contradiction"]), reverse=True)
        label, confidence = _decide(evidence)
        results.append({"claim": claim, "label": label, "confidence": round(confidence, 4), "evidence": evidence})

    return {"language": language, "claims": results, "summary": aggregate(results)}
