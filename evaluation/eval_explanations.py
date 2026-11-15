"""Compares word-explanation methods for the text classifier: speed and faithfulness.

Faithfulness = deletion test (Atanasova et al., EMNLP 2020): remove the 3 words a method says push
hardest towards the predicted class and measure how much the probability of that class drops.
Higher is better. Methods: LIME (200 samples), Gradient x Input, Integrated Gradients (16 steps),
occlusion (each content word removed once; the app's default).

Usage (repo root): backend/.venv/bin/python evaluation/eval_explanations.py
Writes evaluation/results/explanations.json.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.text_classifier import get_text_classifier  # noqa: E402
from app.services.text_explainer import _result, lime_explain, occlusion_explain  # noqa: E402

TEXTS = [
    ("en", "The Taj Mahal is located in Mumbai and was built in 1990 by the British government."),
    ("en", "BREAKING: Scientists confirm that drinking hot water with lemon every morning cures cancer within "
           "two weeks, according to a viral WhatsApp message. Share before it is deleted!"),
    ("en", "The Reserve Bank of India kept the repo rate unchanged at 6.5 per cent on Friday, Governor "
           "Shaktikanta Das said at the monetary policy briefing in Mumbai."),
    ("en", "Chandrayaan-3 landed near the south pole of the Moon on 23 August 2023, making India the fourth "
           "country to land on the Moon, ISRO said."),
    ("te", "5జీ టవర్ల వల్ల కరోనా వైరస్ వ్యాపిస్తుంది అని వాట్సాప్ లో వైరల్ అవుతున్న సందేశం పూర్తిగా అబద్ధం."),
    ("te", "తెలంగాణ రాష్ట్రం 2 జూన్ 2014న ఏర్పడింది, హైదరాబాద్ దాని రాజధాని."),
]


def _words(text, offsets, scores):
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    weights, w = {}, 0
    for (s, e), sc in zip(offsets, scores):
        if e <= s:
            continue
        while w < len(spans) and spans[w][1] <= s:
            w += 1
        if w < len(spans):
            weights[spans[w][2]] = weights.get(spans[w][2], 0.0) + sc
    return weights


def _embed(clf, text):
    enc = clf.tokenizer(text, truncation=True, max_length=clf.max_length, return_offsets_mapping=True,
                        return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    for p in clf.model.parameters():
        p.requires_grad_(False)
    return enc, offsets, clf.model.get_input_embeddings()


def gradient_x_input(clf, text):
    enc, offsets, emb_layer = _embed(clf, text)
    with torch.enable_grad():
        x = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
        logits = clf.model(inputs_embeds=x, attention_mask=enc["attention_mask"]).logits
        torch.log_softmax(logits.float(), -1)[0, clf.fake_index].backward()
    return _result(text, _words(text, offsets, (x.grad * x).sum(-1)[0].tolist()), "Gradient x Input", 12)


def integrated_gradients(clf, text, steps=16):
    enc, offsets, emb_layer = _embed(clf, text)
    ids = enc["input_ids"]
    x = emb_layer(ids).detach()
    base_ids = ids.clone()
    special = torch.tensor([o[1] <= o[0] for o in offsets])
    base_ids[0, ~special] = clf.tokenizer.pad_token_id
    b = emb_layer(base_ids).detach()
    alphas = torch.linspace(1 / steps, 1, steps).view(-1, 1, 1)
    with torch.enable_grad():
        path = (b + alphas * (x - b)).requires_grad_(True)
        logits = clf.model(inputs_embeds=path, attention_mask=enc["attention_mask"].expand(steps, -1)).logits
        torch.log_softmax(logits.float(), -1)[:, clf.fake_index].sum().backward()
    scores = ((x - b) * path.grad.mean(0, keepdim=True)).sum(-1)[0].tolist()
    return _result(text, _words(text, offsets, scores), "Integrated Gradients", 12)


METHODS = {"LIME (200 samples)": lime_explain, "Gradient x Input": gradient_x_input,
           "Integrated Gradients (16 steps)": integrated_gradients, "Occlusion (app default)": occlusion_explain}


def main():
    rows, totals = [], {m: {"seconds": 0.0, "drop": 0.0} for m in METHODS}
    for lang, text in TEXTS:
        clf = get_text_classifier(lang)
        clf.predict(text)
        p0 = clf.predict_proba([text])[0]
        k = int(p0.argmax())
        sign = 1 if k == clf.fake_index else -1
        for name, fn in METHODS.items():
            t = time.time()
            ex = fn(clf, text)
            dt = time.time() - t
            top = [w["word"] for w in sorted(ex["top_words"], key=lambda w: -sign * w["weight"])[:3]]
            reduced = " ".join(x for x in text.split() if x not in top)
            drop = float(p0[k] - clf.predict_proba([reduced])[0][k])
            totals[name]["seconds"] += dt
            totals[name]["drop"] += drop
            rows.append({"language": lang, "text": text[:60], "method": name, "seconds": round(dt, 2),
                         "drop": round(drop, 4), "top3": [str(w) for w in top]})
            print(f"{lang} {name:32s} {dt:6.1f}s drop {drop:+.3f} {top}", flush=True)
    n = len(TEXTS)
    summary = {m: {"total_seconds": round(v["seconds"], 1), "mean_seconds": round(v["seconds"] / n, 2),
                   "mean_drop": round(v["drop"] / n, 4)} for m, v in totals.items()}
    out = {"threads": torch.get_num_threads(), "texts": n, "summary": summary, "rows": rows}
    (ROOT / "evaluation" / "results" / "explanations.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                                        encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
