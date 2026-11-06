"""Does live evidence (Wikipedia, Google News, fact-check API) help claim verification?

Runs every claim twice through the real verify_text():
  local - local corpus only (LIVE_SEARCH off, no fact-check key), as in the main evaluation
  live  - local corpus + live Wikipedia and Google News + Google Fact Check API
Sets:
  curated - evaluation/data/curated_claims.csv (40 claims written for the 40-passage corpus)
  open    - evaluation/data/open_claims.csv (36 claims on topics the corpus does not cover)

Usage (from the repo root): backend/.venv/bin/python evaluation/eval_live_evidence.py
Writes evaluation/results/live_evidence.json and live_evidence_predictions.csv.
Needs internet. Live results depend on what the web returns on the day of the run.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.services.claim_verifier import verify_text  # noqa: E402

OUT = ROOT / "evaluation" / "results"
LABELS = ["Support", "Contradiction", "Unknown"]


def run(df: pd.DataFrame, mode: str) -> list[dict]:
    settings = get_settings()
    key = settings.google_factcheck_api_key
    settings.live_search = mode == "live"
    if mode == "local":
        settings.google_factcheck_api_key = ""
    rows = []
    try:
        for r in df.itertuples():
            t = time.time()
            res = verify_text(r.claim, r.language)
            c = res["claims"][0] if res["claims"] else {"label": "Unknown", "evidence": []}
            top = c["evidence"][0] if c["evidence"] else {}
            rows.append({"claim": r.claim, "language": r.language, "label": r.label, "mode": mode,
                         "pred": c["label"], "evidence_count": len(c["evidence"]),
                         "top_source": top.get("source", ""), "top_type": top.get("type", ""),
                         "decided_by": top.get("decided_by", ""), "seconds": round(time.time() - t, 1)})
            print(f"{mode:5s} {r.label:13s} -> {c['label']:13s} {rows[-1]['seconds']:5.1f}s  {r.claim[:60]}", flush=True)
    finally:
        settings.google_factcheck_api_key = key
        settings.live_search = True
    return rows


def summary(df: pd.DataFrame) -> dict:
    out = {"n": len(df), "correct": int((df.label == df.pred).sum()),
           "accuracy": round(float((df.label == df.pred).mean()), 4) if len(df) else None,
           "unknown_predictions": int((df.pred == "Unknown").sum()),
           "with_evidence": int((df.evidence_count > 0).sum()),
           "mean_seconds": round(float(df.seconds.mean()), 1) if len(df) else None,
           "confusion": {t: {p: int(((df.label == t) & (df.pred == p)).sum()) for p in LABELS} for t in LABELS}}
    return out


def main():
    sets = {"curated": pd.read_csv(ROOT / "evaluation" / "data" / "curated_claims.csv"),
            "open": pd.read_csv(ROOT / "evaluation" / "data" / "open_claims.csv")}
    verify_text("Warm-up sentence for loading the models.", "en")
    verify_text("ఇది మోడల్స్ లోడ్ చేయడానికి ఒక వాక్యం.", "te")
    all_rows = []
    for name, df in sets.items():
        for mode in ("local", "live"):
            for row in run(df, mode):
                all_rows.append({"set": name, **row})
    pred = pd.DataFrame(all_rows)
    pred.to_csv(OUT / "live_evidence_predictions.csv", index=False, encoding="utf-8")

    results = {"date": date.today().isoformat()}
    for name in sets:
        for mode in ("local", "live"):
            sub = pred[(pred.set == name) & (pred["mode"] == mode)]
            entry = {"all": summary(sub)}
            if name == "curated":
                entry["support_or_contradiction_only"] = summary(sub[sub.label != "Unknown"])
                entry["unknown_labelled"] = summary(sub[sub.label == "Unknown"])
            for lang in ("en", "te"):
                entry[lang] = summary(sub[sub.language == lang])
            results[f"{name}_{mode}"] = entry
    live_open = pred[(pred.set == "open") & (pred["mode"] == "live")]
    results["open_live_decided_by"] = {k: int(v) for k, v in live_open.decided_by.replace("", "nli").value_counts().items()}
    results["open_live_top_evidence_type"] = {k: int(v) for k, v in live_open.top_type.replace("", "none").value_counts().items()}
    (OUT / "live_evidence.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in results.items():
        if isinstance(v, dict) and "all" in v:
            print(f"{k:14s} {v['all']['correct']}/{v['all']['n']}  unknown={v['all']['unknown_predictions']}  {v['all']['mean_seconds']}s/claim")


if __name__ == "__main__":
    main()
