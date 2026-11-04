"""Run the functional test cases of Chapter 7 against the running backend and record actual results.

Usage: python evaluation/run_test_cases.py <folder with real_*.jpg / fake_*.jpg images>
Writes evaluation/results/test_cases.json and test_cases.md
"""
import io, json, sys, time, zipfile
from pathlib import Path
import requests

API = "http://127.0.0.1:8000/api"
IMG = Path(sys.argv[1])
OUT = Path(__file__).parent / "results"
SAMPLES = {
    "enFake": "BREAKING: Scientists confirm that 5G mobile towers are spreading the coronavirus across India. The government has secretly ordered all towers to be shut down by Monday. Share this before it gets deleted!",
    "enReal": "India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023, the Indian Space Research Organisation said. ISRO, which is headquartered in Bengaluru, said the rover would study the lunar surface for one lunar day.",
    "te": "తెలంగాణ రాష్ట్రం 2 జూన్ 2014న ఏర్పడింది. హైదరాబాద్ తెలంగాణ రాజధాని. 5జీ టవర్ల వల్ల కరోనా వైరస్ వ్యాపిస్తుందని సోషల్ మీడియాలో ప్రచారం జరుగుతోంది.",
    "claims": "India's Chandrayaan-3 lander touched down near the Moon's south pole on 23 August 2023. The new 2000 rupee notes contain a GPS nano chip that lets the government track them. UNESCO has declared Jana Gana Mana the best national anthem in the world. Hyderabad is the capital of Telangana.",
}
cases = []


def case(cid, module, desc, inp, expected, fn):
    t = time.time()
    try:
        actual, ok = fn()
    except Exception as e:  # record, never stop
        actual, ok = f"error: {e}", False
    dt = time.time() - t
    cases.append(dict(id=cid, module=module, description=desc, input=inp, expected=expected,
                      actual=actual, status="Pass" if ok else "Fail", seconds=round(dt, 1)))
    print(f"{cid} {'PASS' if ok else 'FAIL'} ({dt:.1f}s) {actual}", flush=True)


def post_text(text, **kw):
    return requests.post(f"{API}/text/analyze", json={"text": text, "language": "auto", "explain": False, "verify_claims": True, **kw}, timeout=900)


def img(name):
    return open(next(IMG.glob(name)), "rb").read()


# ---------- system
def tc_health():
    r = requests.get(f"{API}/health", timeout=30).json()
    m = r.get("models", r)
    s = json.dumps(m)
    return f"HTTP 200; models reported: {', '.join(k for k in (m if isinstance(m, dict) else {}))}", "text_en" in s and "image" in s
case("TC01", "System", "Health check reports model status", "GET /api/health", "HTTP 200 with status of every model", tc_health)

# ---------- validation
def tc_short():
    r = post_text("hello"); return f"HTTP {r.status_code}", r.status_code == 422
case("TC02", "Validation", "Reject text shorter than 10 characters", "\"hello\"", "HTTP 422, request rejected", tc_short)

def tc_long():
    r = post_text("word " * 4100); return f"HTTP {r.status_code}", r.status_code == 422
case("TC03", "Validation", "Reject text longer than 20,000 characters", "20,500 characters", "HTTP 422, request rejected", tc_long)

# ---------- text classification
def verdict_case(key, want_lang, want):
    def f():
        r = post_text(SAMPLES[key]).json()
        c, o = r["classification"], r["overall"]
        return (f"language={r['language']}, classifier={c['label']} ({c['confidence']:.1%}), overall={o['verdict']} (risk {o['fake_risk']:.2f})",
                r["language"] == want_lang and o["verdict"] == want)
    return f
case("TC04", "Text check", "English rumour is flagged fake", "5G towers spread coronavirus (sample)", "language en; overall verdict Fake", verdict_case("enFake", "en", "Fake"))
case("TC05", "Text check", "English news report is judged real", "Chandrayaan-3 landing report (sample)", "language en; overall verdict Real", verdict_case("enReal", "en", "Real"))

def tc_te():
    r = post_text(SAMPLES["te"]).json()
    return f"language={r['language']}, classifier={r['classification']['label']} ({r['classification']['confidence']:.1%}), claims={r['verification']['summary']['counts']}", r["language"] == "te" and r["classification"] is not None
case("TC06", "Text check", "Telugu text is detected and classified", "Telugu sample (Telangana, 5G rumour)", "language te; Telugu model gives a verdict", tc_te)

def tc_override():
    r = post_text(SAMPLES["enReal"], language="en", verify_claims=False).json()
    return f"language={r['language']}, verification={'skipped' if r.get('verification') is None else 'ran'}", r["language"] == "en" and r.get("verification") is None
case("TC07", "Text check", "Claim checking can be switched off", "verify_claims=false", "verdict only, no claim results", tc_override)

def tc_explain():
    r = requests.post(f"{API}/text/explain", json={"text": SAMPLES["enFake"], "language": "en"}, timeout=1800).json()
    top = r.get("top_words") or r.get("top") or []
    words = [w.get("word", w.get("token")) if isinstance(w, dict) else w[0] for w in top][:5]
    return f"{len(r.get('tokens', []))} tokens weighted; top words: {', '.join(map(str, words))}", len(r.get("tokens", [])) > 0
case("TC08", "Explanation", "LIME returns word weights", "5G rumour sample", "a weight for each word, top words listed", tc_explain)

# ---------- claims
def tc_claims():
    r = requests.post(f"{API}/claims/verify", json={"text": SAMPLES["claims"], "language": "auto"}, timeout=900).json()
    labels = [c["label"] for c in r["claims"]]
    return f"labels={labels}; summary={r['summary']['verdict']}", labels == ["Support", "Contradiction", "Contradiction", "Support"]
case("TC09", "Claim check", "Four claims labelled against evidence", "Chandrayaan, GPS chip, UNESCO anthem, Hyderabad", "Support, Contradiction, Contradiction, Support", tc_claims)

def tc_unknown():
    r = requests.post(f"{API}/claims/verify", json={"text": "The Polavaram irrigation project will supply water to 4,000 villages by 2027.", "language": "en"}, timeout=900).json()
    labels = [c["label"] for c in r["claims"]]
    return f"labels={labels}; summary={r['summary']['verdict']}", labels == ["Unknown"] and r["summary"]["verdict"] == "Unverified"
case("TC10", "Claim check", "Claim with no evidence is Unknown", "Polavaram claim (not in corpus)", "Unknown; article Unverified", tc_unknown)

def tc_te_claim():
    r = requests.post(f"{API}/claims/verify", json={"text": "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.", "language": "auto"}, timeout=900).json()
    c = r["claims"][0]
    return f"label={c['label']} ({c['confidence']:.0%}); evidence: {c['evidence'][0]['text'][:60] if c['evidence'] else 'none'}", c["label"] == "Support"
case("TC11", "Claim check", "Telugu claim matched with English evidence", "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని.", "Support, using the English Hyderabad passage", tc_te_claim)

# ---------- image
def img_case(name, want):
    def f():
        r = requests.post(f"{API}/image/analyze", files={"file": (name, img(name), "image/jpeg")}, data={"explain": "true"}, timeout=900).json()
        has = bool(r.get("explanation", {}).get("overlay"))
        return f"label={r['label']} ({r['confidence']:.1%}); heatmap={'yes' if has else 'no'}", r["label"].startswith(want) and has
    return f
case("TC12", "Image check", "Real photograph is judged real", "CIFAKE real image", "Real, with Grad-CAM heatmap", img_case("real_0.jpg", "Real"))
case("TC13", "Image check", "Generated image is judged AI-generated", "CIFAKE generated image", "AI-Generated / Manipulated, with Grad-CAM heatmap", img_case("fake_0.jpg", "AI-Generated"))

def tc_badimg():
    r = requests.post(f"{API}/image/analyze", files={"file": ("notes.png", b"this is not an image", "image/png")}, timeout=60)
    return f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 400
case("TC14", "Image check", "Non-image file is rejected", "text file renamed .png", "HTTP 400 with a message", tc_badimg)

def tc_bigimg():
    r = requests.post(f"{API}/image/analyze", files={"file": ("big.jpg", b"\0" * (16 * 1024 * 1024), "image/jpeg")}, timeout=120)
    return f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 413
case("TC15", "Image check", "Image over 15 MB is rejected", "16 MB file", "HTTP 413 with a message", tc_bigimg)

# ---------- batch
job = {}
def tc_batch_text():
    csv = "text\n" + "\n".join('"' + SAMPLES[k].replace('"', '""') + '"' for k in ("enFake", "enReal", "te"))
    r = requests.post(f"{API}/batch/text", files={"file": ("news.csv", csv.encode(), "text/csv")}, data={"language": "auto", "verify_claims": "false"}, timeout=1800).json()
    job["id"] = r["job_id"]
    labels = [x.get("prediction") for x in r["results"]]
    return f"{len(r['results'])} rows; labels={labels}; counts={r.get('counts')}", len(r["results"]) == 3
case("TC16", "Batch", "CSV of three texts is checked", "CSV with text column, 3 rows", "3 results with labels and a job id", tc_batch_text)

def tc_batch_nocol():
    r = requests.post(f"{API}/batch/text", files={"file": ("bad.csv", b"body\nsomething here to check\n", "text/csv")}, timeout=60)
    return f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 400
case("TC17", "Batch", "CSV without a text column is rejected", "CSV with column 'body'", "HTTP 400 naming accepted columns", tc_batch_nocol)

def tc_batch_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in ["real_1.jpg", "real_2.jpg", "fake_1.jpg", "fake_2.jpg"]:
            z.writestr(n, img(n))
    r = requests.post(f"{API}/batch/images", files=[("files", ("images.zip", buf.getvalue(), "application/zip"))], timeout=900).json()
    labels = [x.get("prediction") for x in r["results"]]
    return f"{len(r['results'])} images; labels={labels}", len(r["results"]) == 4
case("TC18", "Batch", "ZIP of four images is checked", "ZIP: 2 real, 2 generated", "4 results with labels", tc_batch_zip)

def tc_download():
    r = requests.get(f"{API}/batch/{job['id']}/download", timeout=60)
    lines = r.text.strip().splitlines()
    return f"HTTP {r.status_code}; {len(lines)} lines; header: {lines[0][:60]}", r.status_code == 200 and len(lines) >= 4
case("TC19", "Batch", "Batch results download as CSV", "job id from TC16", "CSV file with header and 3 rows", tc_download)

def tc_badjob():
    r = requests.get(f"{API}/batch/..%2Fetc/download", timeout=30)
    return f"HTTP {r.status_code}", r.status_code in (400, 404)
case("TC20", "Batch", "Invalid job id is refused", "job id '../etc'", "HTTP 400 or 404, no file served", tc_badjob)

# ---------- history
def tc_hist():
    r = requests.get(f"{API}/history", params={"limit": 50}, timeout=30).json()
    items = r if isinstance(r, list) else r.get("items", [])
    kinds = sorted({i["kind"] for i in items})
    return f"{len(items)} records; kinds={kinds}", len(items) > 0
case("TC21", "History", "Checks are saved in history", "GET /api/history", "records of the checks above", tc_hist)

def tc_hist_filter():
    r = requests.get(f"{API}/history", params={"kind": "image"}, timeout=30).json()
    items = r if isinstance(r, list) else r.get("items", [])
    return f"{len(items)} records, all kind=image: {all(i['kind'] == 'image' for i in items)}", len(items) > 0 and all(i["kind"] == "image" for i in items)
case("TC22", "History", "History filter by type", "kind=image", "only image checks returned", tc_hist_filter)

def tc_hist_404():
    r = requests.delete(f"{API}/history/99999999", timeout=30)
    return f"HTTP {r.status_code}: {r.json().get('detail')}", r.status_code == 404
case("TC23", "History", "Deleting a missing record", "DELETE /api/history/99999999", "HTTP 404 Record not found", tc_hist_404)

OUT.mkdir(exist_ok=True)
json.dump(cases, open(OUT / "test_cases.json", "w"), ensure_ascii=False, indent=1)
with open(OUT / "test_cases.md", "w") as f:
    f.write("| ID | Module | Test | Expected | Actual | Status | Time (s) |\n|---|---|---|---|---|---|---|\n")
    for c in cases:
        f.write(f"| {c['id']} | {c['module']} | {c['description']} | {c['expected']} | {c['actual']} | {c['status']} | {c['seconds']} |\n")
print(sum(c["status"] == "Pass" for c in cases), "/", len(cases), "passed")
