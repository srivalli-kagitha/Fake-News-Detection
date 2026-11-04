"""Evaluate TruthLens claim verification (no training needed: the models are pre-trained).

Benchmarks
  1. FEVER (dev, gold evidence)        claim + Wikipedia evidence -> Support / Contradiction / Unknown
  2. Climate-FEVER (test)              real-world claims, out-of-domain for the NLI model
  3. IndicXNLI Telugu (test)           Telugu premise/hypothesis pairs: shows why a multilingual NLI model is used
  4. Curated end-to-end set            full pipeline (BGE-M3 + FAISS search over backend/data/evidence, then NLI)

Each benchmark compares the TruthLens decision rule (contradiction-aware NLI) with a
similarity-only baseline (a claim is "supported" when similar evidence exists), and sweeps
the decision threshold.

Run from the project folder (uses the backend's virtual environment):
    backend/.venv/bin/python evaluation/eval_claims.py            (Linux/macOS)
    backend\\.venv\\Scripts\\python evaluation\\eval_claims.py      (Windows)
Options: --fever 300 --climate 200 --telugu 1000 (samples per class / pairs), --only fever,climate,...
Results: evaluation/results/claim_eval.md, metrics.json, *.png
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services import claim_verifier  # noqa: E402
from app.services.claim_verifier import NLIModel  # noqa: E402
from app.services.evidence import get_retriever  # noqa: E402

LABELS = ["Support", "Contradiction", "Unknown"]
OUT = ROOT / "evaluation" / "results"
SEED = 42


# ------------------------------------------------------------------ decision rules
def decide(nli_scores: list[dict], threshold: float) -> str:
    """TruthLens rule (same as the app): strongest contradiction vs strongest support."""
    if not nli_scores:
        return "Unknown"
    sup = max(s["entailment"] for s in nli_scores)
    con = max(s["contradiction"] for s in nli_scores)
    if con >= threshold and con >= sup:
        return "Contradiction"
    if sup >= threshold:
        return "Support"
    return "Unknown"


def decide_similarity(similarities: list[float], threshold: float) -> str:
    """Baseline: evidence that is similar enough counts as support. Cannot detect contradiction."""
    return "Support" if similarities and max(similarities) >= threshold else "Unknown"


def metrics(gold: list[str], pred: list[str]) -> dict:
    p, r, f, n = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0), 4),
        "per_class": {
            lab: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4), "support": int(n[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion": confusion_matrix(gold, pred, labels=LABELS).tolist(),
    }


def plot_confusion(cm: list[list[int]], title: str, path: Path, labels=LABELS):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=200)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sweep(items: list[dict], gold: list[str]) -> list[dict]:
    rows = []
    for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
        pred = [decide(it["nli"], t) for it in items]
        rows.append({"threshold": float(t), "accuracy": round(accuracy_score(gold, pred), 4),
                     "macro_f1": round(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0), 4)})
    return rows


def score_claims(items: list[dict], nli: NLIModel, retriever, desc: str) -> None:
    """Add NLI scores and claim-evidence similarities to items: {claim, evidence: [str]}."""
    premises, hypotheses, owner = [], [], []
    for i, it in enumerate(items):
        for ev in it["evidence"]:
            premises.append(ev)
            hypotheses.append(it["claim"])
            owner.append(i)
    t0 = time.time()
    print(f"[{desc}] {len(items)} claims, {len(premises)} claim-evidence pairs: running NLI...", flush=True)
    scores = []
    for s in range(0, len(premises), 64):
        scores += nli.predict(premises[s : s + 64], hypotheses[s : s + 64], batch_size=16)
        print(f"  {min(s + 64, len(premises))}/{len(premises)}  ({time.time() - t0:.0f}s)", flush=True)
    claim_emb = retriever.encode([it["claim"] for it in items])
    ev_emb = retriever.encode(premises) if premises else np.zeros((0, claim_emb.shape[1]), np.float32)
    for it in items:
        it["nli"], it["sim"] = [], []
    for k, (i, sc) in enumerate(zip(owner, scores)):
        items[i]["nli"].append(sc)
        items[i]["sim"].append(float(ev_emb[k] @ claim_emb[i]))


def evaluate(name: str, items: list[dict], threshold: float, sim_threshold: float) -> dict:
    gold = [it["label"] for it in items]
    ours = [decide(it["nli"], threshold) for it in items]
    base = [decide_similarity(it["sim"], sim_threshold) for it in items]
    res = {
        "truthlens": metrics(gold, ours),
        "similarity_baseline": metrics(gold, base),
        "threshold_sweep": sweep(items, gold),
        "gold_distribution": dict(Counter(gold)),
    }
    slug = name.lower().replace(" ", "_").replace("-", "_")
    plot_confusion(res["truthlens"]["confusion"], f"{name}: TruthLens", OUT / f"cm_{slug}.png")
    return res


# ------------------------------------------------------------------ datasets
def load_fever(per_class: int) -> list[dict]:
    path = hf_hub_download("copenlu/fever_gold_evidence", "valid.jsonl", repo_type="dataset")
    mapping = {"SUPPORTS": "Support", "REFUTES": "Contradiction", "NOT ENOUGH INFO": "Unknown"}
    by_label = {v: [] for v in mapping.values()}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ev = [e[2] for e in r.get("evidence", []) if len(e) > 2 and e[2].strip()][:5]
            lab = mapping.get(r["label"])
            if lab and ev:
                by_label[lab].append({"claim": r["claim"], "evidence": ev, "label": lab})
    rng = random.Random(SEED)
    return [x for lab in by_label for x in rng.sample(by_label[lab], min(per_class, len(by_label[lab])))]


def load_climate(per_class: int) -> list[dict]:
    path = hf_hub_download("climate_fever", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    mapping = {0: "Support", 1: "Contradiction", 2: "Unknown"}  # 3 = DISPUTED, left out
    df = df[df.claim_label.isin(mapping)]
    items = [{"claim": r.claim, "evidence": [e["evidence"] for e in r.evidences], "label": mapping[r.claim_label]}
             for r in df.itertuples()]
    rng = random.Random(SEED)
    out = []
    for lab in LABELS:
        group = [x for x in items if x["label"] == lab]
        out += rng.sample(group, min(per_class, len(group)))
    return out


def load_indicxnli_te(n: int) -> list[dict]:
    path = hf_hub_download("Divyanshu/IndicXNLI", "forward/test/xnli_te.json", repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))["test"]
    rng = random.Random(SEED)
    return rng.sample(data, min(n, len(data)))


# ------------------------------------------------------------------ benchmarks
def run_nli_language_test(pairs: list[dict], models: dict) -> dict:
    """3-class NLI accuracy on Telugu pairs for each NLI model (label 0/1/2 = entail/neutral/contradict)."""
    names = ["entailment", "neutral", "contradiction"]
    gold = [names[p["label"]] for p in pairs]
    out = {}
    for key, model in models.items():
        t0 = time.time()
        print(f"[IndicXNLI-te] {key}: {len(pairs)} pairs...", flush=True)
        scores = model.predict([p["premise"] for p in pairs], [p["hypothesis"] for p in pairs], batch_size=16)
        pred = [max(names, key=lambda k: s[k]) for s in scores]
        out[key] = {"accuracy": round(accuracy_score(gold, pred), 4),
                    "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
                    "seconds": round(time.time() - t0)}
        plot_confusion(confusion_matrix(gold, pred, labels=names).tolist(), f"IndicXNLI Telugu: {key}",
                       OUT / f"cm_indicxnli_te_{key}.png", labels=["Entail", "Neutral", "Contradict"])
    return out


def run_end_to_end(path: Path, nli_models: dict, retriever, settings, threshold: float) -> dict:
    df = pd.read_csv(path)
    hits = retriever.search(df.claim.tolist(), settings.evidence_top_k)
    rows = []
    for (_, r), h in zip(df.iterrows(), hits):
        ev = [x for x in h if x["similarity"] >= settings.evidence_min_similarity]
        nli = nli_models["en" if r.language == "en" else "multi"]
        scores = nli.predict([x["text"] for x in ev], [r.claim] * len(ev)) if ev else []
        rows.append({"claim": r.claim, "language": r.language, "label": r.label,
                     "pred": decide(scores, threshold),
                     "baseline": decide_similarity([x["similarity"] for x in ev], settings.evidence_min_similarity),
                     "top_evidence": ev[0]["text"] if ev else ""})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "end_to_end_predictions.csv", index=False, encoding="utf-8")
    res = {"all": metrics(out.label.tolist(), out.pred.tolist()),
           "similarity_baseline": metrics(out.label.tolist(), out.baseline.tolist())}
    for lang in ("en", "te"):
        sub = out[out.language == lang]
        res[lang] = metrics(sub.label.tolist(), sub.pred.tolist())
    plot_confusion(res["all"]["confusion"], "Curated set (end-to-end)", OUT / "cm_end_to_end.png")
    return res


# ------------------------------------------------------------------ report
def fmt_row(name, m):
    pc = m["per_class"]
    return (f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {pc['Support']['f1']:.4f} "
            f"| {pc['Contradiction']['f1']:.4f} | {pc['Unknown']['f1']:.4f} | {m['n']} |")


def report(results: dict, args) -> str:
    lines = ["# Claim verification evaluation", "",
             f"Decision threshold: {args.threshold}. Similarity-baseline threshold: {args.sim_threshold}.", ""]
    head = ["| Method | Accuracy | Macro-F1 | F1 Support | F1 Contradiction | F1 Unknown | Claims |",
            "|---|---|---|---|---|---|---|"]
    for key, title in (("fever", "FEVER (dev, gold evidence)"), ("climate", "Climate-FEVER (out-of-domain)")):
        if key in results:
            r = results[key]
            lines += [f"## {title}", ""] + head + [
                fmt_row("Similarity only (baseline)", r["similarity_baseline"]),
                fmt_row("TruthLens: contradiction-aware NLI", r["truthlens"]), ""]
            best = max(r["threshold_sweep"], key=lambda x: x["macro_f1"])
            lines += [f"Best threshold by macro-F1: {best['threshold']} (macro-F1 {best['macro_f1']:.4f}).", ""]
    if "telugu" in results:
        lines += ["## IndicXNLI Telugu (3-class NLI)", "", "| NLI model | Accuracy | Macro-F1 |", "|---|---|---|"]
        names = {"mdeberta_multilingual": "mDeBERTa-v3-base, multilingual (used for Telugu)",
                 "deberta_large_english": "DeBERTa-v3-large, English only"}
        for k, v in results["telugu"].items():
            lines.append(f"| {names.get(k, k)} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} |")
        lines.append("")
    if "end_to_end" in results:
        r = results["end_to_end"]
        lines += ["## Curated set, full pipeline (retrieval + NLI)", ""] + head + [
            fmt_row("Similarity only (baseline)", r["similarity_baseline"]),
            fmt_row("TruthLens (all)", r["all"]),
            fmt_row("TruthLens (English)", r["en"]),
            fmt_row("TruthLens (Telugu)", r["te"]), ""]
    return "\n".join(lines)


def plot_sweeps(results: dict):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=200)
    for key, title in (("fever", "FEVER"), ("climate", "Climate-FEVER")):
        if key in results:
            s = results[key]["threshold_sweep"]
            ax.plot([x["threshold"] for x in s], [x["macro_f1"] for x in s], marker="o", ms=3, label=title)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sweep.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fever", type=int, default=300, help="claims per class")
    p.add_argument("--climate", type=int, default=150, help="claims per class")
    p.add_argument("--telugu", type=int, default=1000, help="Telugu NLI pairs")
    p.add_argument("--threshold", type=float, default=claim_verifier.DECISION_THRESHOLD)
    p.add_argument("--sim-threshold", type=float, default=get_settings().evidence_min_similarity)
    p.add_argument("--only", default="fever,climate,telugu,end_to_end")
    args = p.parse_args()
    only = set(args.only.split(","))
    OUT.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    print("Loading models (first run downloads them)...", flush=True)
    retriever = get_retriever()
    nli = {"en": NLIModel(settings.nli_en_model), "multi": NLIModel(settings.nli_multi_model)}

    results_file = OUT / "metrics.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    if "fever" in only:
        items = load_fever(args.fever)
        score_claims(items, nli["en"], retriever, "FEVER")
        results["fever"] = evaluate("FEVER", items, args.threshold, args.sim_threshold)
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))  # keep progress
    if "climate" in only:
        items = load_climate(args.climate)
        score_claims(items, nli["en"], retriever, "Climate-FEVER")
        results["climate"] = evaluate("Climate-FEVER", items, args.threshold, args.sim_threshold)
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))  # keep progress
    if "telugu" in only:
        pairs = load_indicxnli_te(args.telugu)
        results["telugu"] = run_nli_language_test(
            pairs, {"mdeberta_multilingual": nli["multi"], "deberta_large_english": nli["en"]})
        results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if "end_to_end" in only:
        results["end_to_end"] = run_end_to_end(ROOT / "evaluation" / "data" / "curated_claims.csv",
                                               nli, retriever, settings, args.threshold)
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    plot_sweeps(results)
    md = report(results, args)
    (OUT / "claim_eval.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
