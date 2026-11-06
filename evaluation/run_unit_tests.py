"""Unit tests of the rule-based parts (Chapter 7). Run from backend/: ../evaluation/... with the backend venv."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.language import detect_language
from app.services.claim_extractor import extract_claims
from app.services.claim_verifier import _decide, aggregate
from app.routers.text import combine

ev = lambda e, c: [{"nli": {"entailment": e, "neutral": max(0.0, 1 - e - c), "contradiction": c}}]
ART = ("India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023. "
       "Is this the greatest achievement ever? Share this before it gets deleted! "
       "ISRO is headquartered in Bengaluru. Telangana was formed on 2 June 2014.")
rows = []
def t(cid, fn, inp, expected, got):
    ok = got == expected
    rows.append(dict(id=cid, function=fn, input=inp, expected=str(expected), actual=str(got), status="Pass" if ok else "Fail"))
    print(cid, "PASS" if ok else "FAIL", got)

t("UT01", "detect_language", "Hyderabad is the capital of Telangana.", "en", detect_language("Hyderabad is the capital of Telangana."))
t("UT02", "detect_language", "హైదరాబాద్ తెలంగాణ రాజధాని.", "te", detect_language("హైదరాబాద్ తెలంగాణ రాజధాని."))
t("UT03", "detect_language", "5జీ టవర్లు 2020 నుంచి ఉన్నాయి (Telugu with digits and 5G)", "te", detect_language("5జీ టవర్లు 2020 నుంచి ఉన్నాయి"))
cl = extract_claims(ART, "en")
t("UT04", "extract_claims", "Article with a question and a call to action", False, any("Share this" in c for c in cl))
t("UT05", "extract_claims", "Same article", False, any(c.strip().endswith("?") for c in cl))
t("UT06", "extract_claims", "Same article (3 factual sentences)", 3, len(cl))
t("UT07", "extract_claims", "Single headline with no factual cues", 1, len(extract_claims("Wow, unbelievable stuff", "en")))
t("UT08", "_decide", "entailment 0.9, contradiction 0.05", ("Support", 0.9), _decide(ev(0.9, 0.05)))
t("UT09", "_decide", "entailment 0.6, contradiction 0.7", ("Contradiction", 0.7), _decide(ev(0.6, 0.7)))
lab, conf = _decide(ev(0.3, 0.2))
t("UT10", "_decide", "entailment 0.3, contradiction 0.2", ("Unknown", 0.7), (lab, round(conf, 4)))
t("UT11", "_decide", "no evidence", ("Unknown", 0.0), _decide([]))
a = aggregate([{"label": "Support", "confidence": 0.9}, {"label": "Contradiction", "confidence": 0.8}])
t("UT12", "aggregate", "Support 0.9 + Contradiction 0.8", ("Mixed Evidence", 0.525), (a["verdict"], a["credibility_score"]))
a = aggregate([{"label": "Contradiction", "confidence": 0.95}, {"label": "Unknown", "confidence": 0.5}])
t("UT13", "aggregate", "Contradiction 0.95 + Unknown", ("Likely Misleading", 0.2625), (a["verdict"], a["credibility_score"]))
t("UT14", "aggregate", "only Unknown claims", "Unverified", aggregate([{"label": "Unknown", "confidence": 0.6}])["verdict"])
o = combine({"fake_probability": 0.8}, {"summary": {"verdict": "Mixed Evidence", "credibility_score": 0.4}})
t("UT15", "combine", "fake probability 0.8, credibility 0.4 (evidence weight 0.7)", ("Fake", 0.66), (o["verdict"], round(o["fake_risk"], 4)))
o = combine({"fake_probability": 0.3}, {"summary": {"verdict": "Unverified", "credibility_score": 0.5}})
t("UT16", "combine", "fake probability 0.3, claims Unverified", ("Unverified", None, 0.3), (o["verdict"], o["fake_risk"], o["style_fake_probability"]))

out = Path(__file__).parent / "results"; out.mkdir(exist_ok=True)
json.dump(rows, open(out / "unit_tests.json", "w"), ensure_ascii=False, indent=1)
print(sum(r["status"] == "Pass" for r in rows), "/", len(rows))
