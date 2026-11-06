"""Compares three versions of claim checking on the same claims, in one process:

  nli          - Wikipedia + news + fact-checks, NLI decides (no web search)
  web+nli      - the same plus DuckDuckGo web search, NLI decides
  web+minicheck- the same evidence, MiniCheck decides support for English claims

Live search results are cached in memory, so every version sees the same evidence.
Claims: evaluation/data/sentence_cases.csv (41, overall verdict Real/Fake/Unverified)
        evaluation/data/open_claims.csv (36, claim label Support/Contradiction/Unknown)
Usage (repo root, internet needed): backend/.venv/bin/python evaluation/compare_verifiers.py [variants] [sets]
  e.g. ... compare_verifiers.py web+minicheck sentence   (one version, the 41 sentence cases only)
Writes evaluation/results/verifier_comparison.json and verifier_comparison.csv.
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
from app.routers.text import combine  # noqa: E402
from app.services.claim_verifier import verify_text  # noqa: E402
from app.services.text_classifier import get_text_classifier  # noqa: E402

OUT = ROOT / "evaluation" / "results"
VARIANTS = {"nli": (False, "nli"), "web+nli": (True, "nli"), "web+minicheck": (True, "minicheck")}
SUMMARY_TO_LABEL = {"Likely Credible": "Support", "Likely Misleading": "Contradiction",
                    "Unverified": "Unknown", "Mixed Evidence": "Mixed"}


def main():
    settings = get_settings()
    sentences = pd.read_csv(ROOT / "evaluation" / "data" / "sentence_cases.csv")
    open_claims = pd.read_csv(ROOT / "evaluation" / "data" / "open_claims.csv")
    pick_variants = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    pick_sets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sentence", "open"]
    items = [("sentence", r.id, r.category, r.language, r.text, r.expected) for r in sentences.itertuples()]
    items += [("open", f"OC{i + 1:02d}", "Open claim", r.language, r.claim, r.label)
              for i, r in enumerate(open_claims.itertuples())]
    items = [i for i in items if i[0] in pick_sets]
    shard = sys.argv[3] if len(sys.argv) > 3 else ""  # "0/2" -> every second item from 0, to run in parallel
    if shard:
        k, n = map(int, shard.split("/"))
        items = items[k::n]
    style = {}
    for _, _, _, lang, text, _ in items:
        if (lang, text) not in style:
            style[(lang, text)] = get_text_classifier(lang).predict(text)

    rows = []
    for name, (web, verifier) in VARIANTS.items():
        if name not in pick_variants:
            continue
        settings.web_search, settings.support_verifier = web, verifier
        for kind, cid, cat, lang, text, expected in items:
            t = time.time()
            v = verify_text(text, lang)
            if kind == "sentence":
                got = combine(style[(lang, text)], v)["verdict"]
                result = "Pass" if got == expected else "Unverified" if got == "Unverified" else "Fail"
            else:
                got = SUMMARY_TO_LABEL.get(v["summary"]["verdict"], v["summary"]["verdict"])
                result = "Pass" if got == expected else "Unverified" if got == "Unknown" else "Fail"
            rows.append({"variant": name, "set": kind, "id": cid, "category": cat, "language": lang,
                         "text": text, "expected": expected, "got": got, "result": result,
                         "claims": "|".join(c["label"] for c in v["claims"]), "seconds": round(time.time() - t, 1)})
            print(f"{name:14s} {cid} {result:10s} expected {expected:13s} got {got:13s} "
                  f"{rows[-1]['seconds']:5.1f}s | {text[:55]}", flush=True)
    settings.web_search, settings.support_verifier = True, "minicheck"

    df = pd.DataFrame(rows)
    tag = f"_{shard.replace('/', 'of')}" if shard else ""
    df.to_csv(OUT / f"verifier_comparison{tag}.csv", index=False, encoding="utf-8")
    summary = {"date": date.today().isoformat()}
    for name in pick_variants:
        entry = {}
        for kind in pick_sets:
            sub = df[(df.variant == name) & (df.set == kind)]
            entry[kind] = {r: int((sub.result == r).sum()) for r in ("Pass", "Unverified", "Fail")}
            entry[kind]["total"] = len(sub)
            entry[kind]["mean_seconds"] = round(float(sub.seconds.mean()), 1)
        summary[name] = entry
    (OUT / f"verifier_comparison{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
