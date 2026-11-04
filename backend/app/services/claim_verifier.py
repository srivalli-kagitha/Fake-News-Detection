"""Contradiction-aware, claim-level evidence verification.

article -> claims -> evidence (FAISS + fact-check API) -> NLI(evidence, claim)
        -> Support / Contradiction / Unknown per claim -> overall assessment
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings
from .claim_extractor import extract_claims
from .common import Lazy, get_device
from .evidence import get_retriever, search_fact_checks

DECISION_THRESHOLD = 0.5


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


def verify_text(text: str, language: str) -> dict:
    settings = get_settings()
    claims = extract_claims(text, language)
    retriever = get_retriever()
    corpus_hits = retriever.search(claims, settings.evidence_top_k)
    nli = get_nli(language)

    results = []
    for claim, hits in zip(claims, corpus_hits):
        evidence = [h for h in hits if h["similarity"] >= settings.evidence_min_similarity]
        evidence += search_fact_checks(claim, language)
        if evidence:
            scores = nli.predict([e["text"] for e in evidence], [claim] * len(evidence))
            for e, s in zip(evidence, scores):
                e["nli"] = s
            evidence.sort(key=lambda e: max(e["nli"]["entailment"], e["nli"]["contradiction"]), reverse=True)
        label, confidence = _decide(evidence)
        results.append({"claim": claim, "label": label, "confidence": round(confidence, 4), "evidence": evidence})

    return {"language": language, "claims": results, "summary": aggregate(results)}
