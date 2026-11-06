"""LLM fact checker: plan searches, gather evidence, decide with cited evidence.

Follows recent open fact-checking systems (ClaimCheck 2025, FIRE 2024, AVeriTeC 2025 entries):
  1. Query planning - the LLM writes search queries for each claim (English too for Telugu claims).
  2. Evidence      - local corpus, Wikipedia, DuckDuckGo web, Google News and fact-checks are searched
                     with those queries; the passages closest to the claim (BGE-M3) are kept.
  3. Verdict       - the LLM reads the numbered evidence and answers SUPPORTED / REFUTED /
                     NOT_ENOUGH_EVIDENCE, with a confidence, the evidence numbers it relied on and a
                     one-sentence reason. If it decided from general knowledge alone, it says so.

Providers (LLM_PROVIDER in .env), both through the OpenAI-compatible chat API:
  groq   - free key from console.groq.com (GROQ_API_KEY), fast, open-weight models
  ollama - a local model (e.g. qwen3:4b) through Ollama, free and offline but slow on a CPU
  none   - switch the LLM off; the NLI pipeline in claim_verifier.py is used instead
If the LLM call fails for any reason, claim_verifier falls back to the NLI pipeline.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date
from functools import lru_cache

import requests

from ..config import get_settings
from .common import logger
from .evidence import get_retriever, search_fact_checks
from .live_sources import SearchFailed, _news, _web, _wikipedia, _wikipedia_titles, keywords

LABELS = {"SUPPORTED": "Support", "REFUTED": "Contradiction", "NOT_ENOUGH_EVIDENCE": "Unknown"}
EVIDENCE_PER_CLAIM = 10
SEARCH_DEADLINE = 20  # seconds for all searches of one claim


class LLMUnavailable(Exception):
    pass


def llm_available() -> bool:
    s = get_settings()
    if s.llm_provider == "groq":
        return bool(s.groq_api_key.strip())
    return s.llm_provider == "ollama"


def llm_name() -> str | None:
    s = get_settings()
    if not llm_available():
        return None
    return f"{s.llm_provider}:{s.llm_model if s.llm_provider == 'groq' else s.ollama_model}"


def _retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


@lru_cache(maxsize=512)
def _chat(system: str, user: str) -> dict:
    """One chat call that must return a JSON object. Cached, so a repeated check costs no tokens."""
    s = get_settings()
    if s.llm_provider == "groq":
        url, model = "https://api.groq.com/openai/v1/chat/completions", s.llm_model
        headers = {"Authorization": f"Bearer {s.groq_api_key.strip()}"}
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    elif s.llm_provider == "ollama":
        url, model, headers, extra = f"{s.ollama_url.rstrip('/')}/v1/chat/completions", s.ollama_model, {}, {}
    else:
        raise LLMUnavailable("LLM_PROVIDER is none")
    body = {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **extra}
    try:
        for attempt in range(4):
            resp = requests.post(url, json=body, headers=headers, timeout=s.llm_timeout)
            wait = _retry_after(resp)
            if resp.status_code == 429 and wait is not None and wait <= 20 and attempt < 3:
                time.sleep(wait + 0.5)  # per-minute token limit: wait the few seconds Groq asks for
                continue
            break
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")[:200]
        raise LLMUnavailable(f"{type(exc).__name__}: {exc} {detail}") from exc
    match = re.search(r"\{.*\}", text, re.S)  # tolerate text around the JSON
    try:
        return json.loads(match.group(0) if match else text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"LLM did not return JSON: {text[:200]}") from exc


PLAN_SYSTEM = """You plan web searches for a fact-checker.
For each numbered claim, write 2 or 3 short search queries (3 to 8 words) that would find reliable evidence
to confirm or refute it: the key fact, the entity's official details, a date, a place, a number.
Write the queries in English. For a claim in Telugu or another language, translate the key terms to English.
Reply with JSON only: {"queries": {"1": ["...", "..."], "2": ["..."]}}"""


def plan_queries(claims: list[str]) -> list[list[str]]:
    user = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    try:
        out = _chat(PLAN_SYSTEM, user).get("queries", {})
    except LLMUnavailable:
        raise
    plans = []
    for i, claim in enumerate(claims):
        qs = out.get(str(i + 1)) or out.get(i + 1) or []
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:3]
        plans.append(qs or [keywords(claim)])
    return plans


def _search_all(claim: str, queries: list[str], language: str) -> list[dict]:
    """Every source, every query. Failures of single sources are ignored."""
    items: list[dict] = []
    jobs = []
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        jobs.append(pool.submit(search_fact_checks, claim, language))
        for q in queries:
            jobs.append(pool.submit(lambda q=q: list(_wikipedia(q, "en", 2, 15)[0])))
            jobs.append(pool.submit(lambda q=q: list(_web(q, "in-en", 6))))
            jobs.append(pool.submit(lambda q=q: list(_news(q, "en", 8))))
        if language == "te":
            def telugu():
                found, english = _wikipedia(keywords(claim), "te", 2, 12)
                return list(found) + (list(_wikipedia_titles(tuple(english[:1]), "en", 12)) if english else [])
            jobs.append(pool.submit(telugu))
        # A search engine that stops answering must not hold up the check: every source gets a hard
        # deadline and whatever has not answered by then is skipped.
        deadline = time.monotonic() + SEARCH_DEADLINE
        for job in jobs:
            try:
                items += [dict(d) for d in job.result(timeout=max(0.1, deadline - time.monotonic()))]
            except FutureTimeout:
                logger.warning("A search source took longer than %ss and was skipped.", SEARCH_DEADLINE)
            except Exception as exc:  # one bad source must not stop the check
                logger.debug("search source failed: %s", exc)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return items


def _select(retriever, claim: str, corpus_hits: list[dict], items: list[dict], queries: list[str]) -> list[dict]:
    """The most relevant passages, without duplicates; published fact-checks always kept.

    Relevance is the best match with the claim OR one of the LLM's search queries: the passage that
    refutes "Rahul Gandhi became PM" is about Narendra Modi, so it matches the query
    "Prime Minister of India 2024" far better than the claim itself."""
    seen, pool = set(), []
    for d in corpus_hits + items:
        key = d["text"][:200]
        if key not in seen and len(d["text"]) >= 20:
            seen.add(key)
            pool.append(d)
    if not pool:
        return []
    keys = [d.get("claim_reviewed") or d["text"] for d in pool]
    targets = retriever.encode([claim] + queries)
    sims = retriever.encode(keys) @ targets.T  # passages x (claim + queries)
    for d, row in zip(pool, sims):
        d["similarity"] = float(row[0])          # shown to the user: closeness to the claim
        d["_rank"] = float(row.max())            # used for choosing the passages
    checks = sorted([d for d in pool if d.get("type") == "fact_check"], key=lambda d: -d["_rank"])[:2]
    rest = sorted([d for d in pool if d.get("type") != "fact_check"], key=lambda d: -d["_rank"])
    chosen = [d for d in checks + rest[: EVIDENCE_PER_CLAIM - len(checks)] if d["_rank"] >= 0.3]
    for d in chosen:
        d.pop("_rank", None)
    return chosen


VERDICT_SYSTEM = """You are a careful fact-checker. Decide whether the CLAIM is SUPPORTED, REFUTED or
NOT_ENOUGH_EVIDENCE, using the numbered EVIDENCE first and well-established general knowledge second.
Rules:
- A claim with any wrong key detail (person, place, date, number, title) is REFUTED.
- Watch for evidence about a different thing with a similar name (the Taj Mahal Palace hotel is not the
  Taj Mahal; the Charminar Express train is not the Charminar). Ignore such evidence.
- Old news about plans can differ from what happened; prefer encyclopedic and later sources.
- A fact-check that rated the same claim False or Misleading is strong evidence for REFUTED.
- Use NOT_ENOUGH_EVIDENCE for predictions, local rumours or anything you cannot settle.
- Events after your training data: rely only on the evidence.
- "cited": the evidence numbers your decision rests on. If you decided from general knowledge without
  any supporting evidence, give [] and set "knowledge_only": true.
- "reason": one plain English sentence a student can read aloud.
Today's date is {today}. Reply with JSON only:
{{"verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_EVIDENCE", "confidence": 0.0-1.0, "cited": [1, 2],
"knowledge_only": false, "reason": "..."}}"""


def _evidence_block(evidence: list[dict]) -> str:
    lines = []
    for i, e in enumerate(evidence, 1):
        text = e.get("nli_text") or e["text"]
        lines.append(f"[{i}] ({e.get('source', 'source')}) {text[:700]}")
    return "\n".join(lines) or "(no evidence found)"


def decide(claim: str, evidence: list[dict]) -> dict:
    system = VERDICT_SYSTEM.format(today=date.today().isoformat())
    out = _chat(system, f"CLAIM: {claim}\n\nEVIDENCE:\n{_evidence_block(evidence)}")
    verdict = str(out.get("verdict", "")).upper().replace(" ", "_")
    label = LABELS.get(verdict, "Unknown")
    try:
        confidence = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    cited = [int(c) for c in out.get("cited", []) if str(c).isdigit() and 1 <= int(c) <= len(evidence)]
    knowledge_only = bool(out.get("knowledge_only")) or (label != "Unknown" and not cited)
    if knowledge_only and label != "Unknown":
        confidence = min(confidence, 0.8)  # no source backs it: never fully confident
    return {"label": label, "confidence": round(confidence, 4), "cited": cited,
            "knowledge_only": knowledge_only and label != "Unknown",
            "reason": str(out.get("reason", "")).strip()}


def llm_verify(claims: list[str], facts: list[str], language: str) -> list[dict]:
    """Returns one result per claim in the same shape as the NLI pipeline, plus reason/cited fields."""
    settings = get_settings()
    retriever = get_retriever()
    plans = plan_queries(facts)
    corpus = retriever.search(facts, settings.evidence_top_k)
    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda a: _search_all(*a), [(f, q, language) for f, q in zip(facts, plans)]))
    results = []
    for claim, fact, queries, hits, items in zip(claims, facts, plans, corpus, found):
        evidence = _select(retriever, fact, [dict(h) for h in hits if h["similarity"] >= 0.3], items, queries)
        d = decide(fact, evidence)
        for i, e in enumerate(evidence, 1):
            e["cited"] = i in d["cited"]
            e.pop("nli", None)
        evidence.sort(key=lambda e: (not e["cited"], -e["similarity"]))
        results.append({"claim": claim, "checked_as": fact, "label": d["label"], "confidence": d["confidence"],
                        "reason": d["reason"], "knowledge_only": d["knowledge_only"], "queries": queries,
                        "decided_by": "llm", "evidence": evidence})
    return results
