"""Single-sentence test cases: does the overall verdict match the known truth?

Each case in evaluation/data/sentence_cases.csv is sent to the running app
(POST /api/text/analyze with claim checking on), exactly as the web page does.
Start the app first (start.bat / python start.py), then from the repo root:
    backend/.venv/bin/python evaluation/run_sentence_tests.py      (Windows: backend\\.venv\\Scripts\\python ...)
Writes evaluation/results/sentence_tests.json and sentence_tests.md.
Needs internet: live evidence changes with what Wikipedia and Google News return that day.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "results"
API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/api"


def main():
    cases = pd.read_csv(ROOT / "evaluation" / "data" / "sentence_cases.csv")
    rows = []
    for c in cases.itertuples():
        t = time.time()
        r = requests.post(f"{API}/text/analyze", json={"text": c.text, "language": c.language,
                                                       "explain": False, "verify_claims": True}, timeout=900)
        r.raise_for_status()
        d = r.json()
        claims = d["verification"]["claims"] if d.get("verification") else []
        top = claims[0]["evidence"][0] if claims and claims[0]["evidence"] else {}
        row = {"id": c.id, "category": c.category, "language": c.language, "text": c.text,
               "expected": c.expected, "verdict": d["overall"]["verdict"], "fake_risk": d["overall"]["fake_risk"],
               "style_fake_probability": round(d["classification"]["fake_probability"], 4),
               "claims": [x["label"] for x in claims],
               "method": (d.get("verification") or {}).get("method"), "reason": claims[0].get("reason", "") if claims else "", "top_evidence": top.get("text", "")[:160],
               "top_source": top.get("source", ""), "seconds": round(time.time() - t, 1)}
        # "Unverified" = no evidence for or against: the app says it cannot tell, which is not a wrong answer
        row["result"] = ("Pass" if row["verdict"] == row["expected"]
                         else "Unverified" if row["verdict"] == "Unverified" else "Fail")
        rows.append(row)
        print(f"{row['id']} {row['result']:4s} expected {row['expected']:4s} got {row['verdict']:4s} "
              f"{row['method']} risk={row['fake_risk'] if row['fake_risk'] is None else round(row['fake_risk'], 2)} claims={row['claims']} {row['seconds']:.0f}s | {c.text[:60]}", flush=True)
        hid = d.get("id")
        if hid:
            requests.delete(f"{API}/history/{hid}", timeout=30)

    by_cat = defaultdict(lambda: {"Pass": 0, "Unverified": 0, "Fail": 0, "total": 0})
    for r in rows:
        by_cat[r["category"]][r["result"]] += 1
        by_cat[r["category"]]["total"] += 1
    passed = sum(r["result"] == "Pass" for r in rows)
    unverified = sum(r["result"] == "Unverified" for r in rows)
    failed = sum(r["result"] == "Fail" for r in rows)
    result = {"date": date.today().isoformat(), "passed": passed, "unverified": unverified, "failed": failed,
              "total": len(rows), "by_category": dict(by_cat),
              "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1), "cases": rows}
    (OUT / "sentence_tests.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Single-sentence tests ({result['date']}): {passed} passed, {unverified} unverified, "
             f"{failed} failed (of {len(rows)})", "",
             "Unverified = no evidence found, the app says it cannot tell (not a wrong answer).", "",
             "| ID | Category | Sentence | Expected | Got | Claims | Result |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {r['text']} | {r['expected']} | {r['verdict']} "
                     f"({'-' if r['fake_risk'] is None else format(r['fake_risk'], '.2f')}) | {', '.join(r['claims'])} | {r['result']} |")
    (OUT / "sentence_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed} passed, {unverified} unverified, {failed} failed (of {len(rows)})")
    for k, v in result["by_category"].items():
        print(f"  {k:24s} pass {v['Pass']}  unverified {v['Unverified']}  fail {v['Fail']}  (of {v['total']})")


if __name__ == "__main__":
    main()
