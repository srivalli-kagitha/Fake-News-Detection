"""Contradiction-aware, claim-level evidence verification.

article -> claims -> evidence (local FAISS corpus + live Wikipedia/news + fact-check API)
        -> NLI(evidence, claim) -> Support / Contradiction / Unknown per claim -> overall assessment
"""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings
from .claim_extractor import extract_claims
from .common import Lazy, get_device, logger
from .evidence import get_retriever, rating_polarity, search_fact_checks
from .llm_verifier import LLMUnavailable, llm_available, llm_name, llm_verify
from .live_sources import DEBUNK_HEADLINE, search_news, search_web, search_wikipedia

DECISION_THRESHOLD = 0.5

# "..., the Indian Space Research Organisation said." / "According to PTI, ..." : evidence states the fact,
# not who reported it, so NLI would answer "neutral". The fact is checked without the attribution.
REPORT_VERBS = r"(?:said|says|stated|states|announced|reported|reports|confirmed|claimed|claims|added|told reporters|informed)"
ATTRIBUTION = [
    re.compile(r",\s+(?:the\s+|a\s+)?[^,]{2,70}?\s+" + REPORT_VERBS + r"(?:\s+on\s+\w+|\s+today|\s+yesterday)?\s*([.!]?)\s*$", re.I),
    re.compile(r",?\s+(?:according to|as per)\s+[^,]{2,70}?\s*([.!]?)\s*$", re.I),
]
LEADING_ATTRIBUTION = [
    re.compile(r"^(?:according to|as per)\s+[^,]{2,70},\s*", re.I),
    re.compile(r"^[^,]{2,60}?\s+" + REPORT_VERBS + r"\s+that\s+", re.I),
]


def strip_attribution(claim: str) -> str:
    text = claim.strip()
    for pattern in LEADING_ATTRIBUTION:
        text = pattern.sub("", text, count=1)
    for pattern in ATTRIBUTION:
        text = pattern.sub(lambda m: m.group(1) or ".", text, count=1)
    text = text.strip()
    if len(text.split()) < 3:  # nothing left to check: keep the original
        return claim
    return text[0].upper() + text[1:]
MAX_EVIDENCE = 5          # passages per claim sent to NLI (NLI is the slow step on a CPU)
SAME_CLAIM_SIMILARITY = 0.65  # a fact-check or headline must be this close to be about the same claim
HEADLINE_SUPPORT_SIMILARITY = 0.75  # a headline is short and often about something else (the "Charminar Express"
                                    # train is not the Charminar), so it may only support a near-identical claim
MULTI_MIN_SIMILARITY = 0.50
# "Taj Mahal Palace" (a hotel in Mumbai) or "Charminar Express" (a train) is not the Taj Mahal or the Charminar:
# evidence where a name from the claim is followed by one of these words is about something else.
OTHER_THING = re.compile(
    r"^(?:palace|hotel|express|station|airport|stadium|hospital|restaurant|road|street|nagar|colony|bridge|mall|"
    r"club|foundation|award|trophy|film|movie|song|school|college|university|metro|cafe|tower|towers|resort|"
    r"apartments?|complex|market|bazaar|beach|lake|park|cinema|theatre|theater|company|limited|ltd|group|"
    r"museum|temple|street)\b", re.I)  # the multilingual NLI model accepts loose matches, so Telugu evidence must be closer


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


class MiniCheckModel:
    """MiniCheck (Tang et al., EMNLP 2024): is the claim supported by this document? -> probability.

    Unlike NLI, which reads one passage at a time, it reads all the evidence for a claim together,
    so a fact spread over two sentences ("Tokyo 2020 Olympics" + "held on 7 August 2021") can count.
    It has no "contradicted" output and is English only, so contradictions still come from NLI."""

    def __init__(self, model_name: str):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def support(self, document: str, claim: str) -> float:
        enc = self.tokenizer(document, claim, truncation="only_first", max_length=512,
                             return_tensors="pt").to(self.device)
        return float(torch.softmax(self.model(**enc).logits.float(), dim=-1)[0, 1])


_settings = get_settings()
_minicheck = Lazy("MiniCheck support model", lambda: MiniCheckModel(_settings.minicheck_model))
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


def _about_other_thing(claim: str, text: str) -> bool:
    """True when the evidence names a different thing that starts with a name from the claim."""
    claim_words = re.findall(r"[\w-]+", claim.lower())
    lower = text.lower()
    for n in (2, 1):
        for i in range(len(claim_words) - n + 1):
            name = " ".join(claim_words[i : i + n])
            if len(name) < 4:
                continue
            for m in re.finditer(r"\b" + re.escape(name) + r"\b\s+([\w-]+)", lower):
                nxt = m.group(1)
                if nxt not in claim_words and OTHER_THING.match(nxt):
                    return True
    return False


def _evidence_document(evidence: list[dict]) -> str:
    """All evidence for one claim as one document, most relevant first, without repeating a lead sentence."""
    pieces = []
    for e in evidence:
        lead = e.get("nli_text", e["text"])[: -len(e["text"])].strip() if e.get("nli_text") else ""
        for part in (lead, e["text"]):
            if part and part not in pieces:
                pieces.append(part)
    return " ".join(pieces)


def _decide_minicheck(evidence: list[dict], support: float) -> tuple[str, float]:
    """Support from MiniCheck (or a fact-checker's 'true' rating), contradiction from NLI and the
    rating/headline rules. Contradiction is tested first, as in _decide."""
    if not evidence:
        return "Unknown", 0.0
    support = max([support] + [e["nli"]["entailment"] for e in evidence if e.get("decided_by")])
    best_contra = max(e["nli"]["contradiction"] for e in evidence)
    if best_contra >= DECISION_THRESHOLD and best_contra >= support:
        return "Contradiction", best_contra
    if support >= DECISION_THRESHOLD:
        return "Support", support
    return "Unknown", 1.0 - max(support, best_contra)


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
        if settings.web_search and language == "en":  # Telugu web snippets were too noisy for the Telugu NLI model
            jobs["web"] = search_web
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
    facts = [strip_attribution(c) for c in claims]  # what is checked; the claim text is shown as written
    if claims and llm_available():
        try:
            results = llm_verify(claims, facts, language)
            return {"language": language, "claims": results, "summary": aggregate(results),
                    "method": "llm", "model": llm_name()}
        except LLMUnavailable as exc:
            logger.warning("LLM fact checker unavailable, using the NLI pipeline: %s", exc)
    retriever = get_retriever()
    corpus_hits = retriever.search(facts, settings.evidence_top_k)
    live_results = _gather_live(facts, language, settings) if claims else []
    claim_vecs = retriever.encode(facts) if claims else []
    nli = get_nli(language)

    results = []
    for claim, fact, hits, live, vec in zip(claims, facts, corpus_hits, live_results, claim_vecs):
        min_sim = settings.evidence_min_similarity if language == "en" else max(settings.evidence_min_similarity, MULTI_MIN_SIMILARITY)
        evidence = [h for h in hits if h["similarity"] >= min_sim]
        evidence += [e for e in _rank_live(retriever, vec, live, settings, {e["text"] for e in evidence}) if e["similarity"] >= min_sim]
        evidence = sorted(evidence, key=lambda e: -e["similarity"])[:MAX_EVIDENCE]
        if evidence:
            scores = nli.predict([e.get("nli_text", e["text"]) for e in evidence], [fact] * len(evidence))
            for e, s in zip(evidence, scores):
                e["nli"] = s
            _apply_ratings(nli, fact, evidence)
            for e in evidence:
                if _about_other_thing(fact, e["text"]):
                    e["nli"] = {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
                    e["about_other_thing"] = True
                elif e.get("type") == "news" and not e.get("decided_by"):
                    # Headlines are short and often old plans ("set for August launch"): on their own they
                    # never contradict (only via the debunk rule), and support only a near-identical claim.
                    e["nli"] = {**e["nli"], "contradiction": 0.0}
                    if e["similarity"] < HEADLINE_SUPPORT_SIMILARITY:
                        e["nli"]["entailment"] = 0.0
            evidence.sort(key=lambda e: max(e["nli"]["entailment"], e["nli"]["contradiction"]), reverse=True)
        minicheck = None
        if evidence and language == "en" and settings.support_verifier == "minicheck":
            minicheck = _minicheck.get().support(
                _evidence_document([e for e in evidence if not e.get("about_other_thing")]), fact)
            label, confidence = _decide_minicheck(evidence, minicheck)
        else:
            label, confidence = _decide(evidence)
        results.append({"claim": claim, "checked_as": fact, "minicheck_support": minicheck, "label": label, "confidence": round(confidence, 4), "evidence": evidence})

    return {"language": language, "claims": results, "summary": aggregate(results), "method": "nli", "model": None}
