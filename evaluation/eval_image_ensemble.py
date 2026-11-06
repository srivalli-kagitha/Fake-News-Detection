"""Compares our CLIP (CIFAKE) image detector, the modern detector and their average.

Test images (downloaded once into evaluation/data/image_ensemble/, not committed):
  ai/   24 Midjourney v6 images (HF datasets brivangl/midjourney-v6-llava, Photoroom/midjourney-v6-recap)
        + 6 CIFAKE AI images + 1 AI portrait supplied by the user (ai/user_portrait.jpg, optional)
  real/ 8 Wikipedia lead photos + 16 Flickr30k test photos + 6 CIFAKE real images

Usage (from the repo root): backend/.venv/bin/python evaluation/eval_image_ensemble.py
Writes evaluation/results/image_ensemble.json.
"""
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.image_detector import get_image_detector  # noqa: E402

DATA = ROOT / "evaluation" / "data" / "image_ensemble"
OUT = ROOT / "evaluation" / "results"
UA = {"User-Agent": "TruthLensAI-student-project/1.0 (https://github.com/srivalli-kagitha/Fake-News-Detection)"}
WIKI = ["Narendra_Modi", "Virat_Kohli", "Charminar", "Taj_Mahal", "Hyderabad", "Sachin_Tendulkar",
        "Golden_Gate_Bridge", "Cat"]


def rows_api(dataset: str, config: str, split: str, offset: int, length: int) -> list[str]:
    url = "https://datasets-server.huggingface.co/rows"
    r = requests.get(url, params={"dataset": dataset, "config": config, "split": split,
                                  "offset": offset, "length": length}, timeout=60)
    r.raise_for_status()
    urls = []
    for row in r.json()["rows"]:
        for v in row["row"].values():
            if isinstance(v, dict) and "src" in v:
                urls.append(v["src"])
    return urls


def save(url: str, path: Path):
    if not path.exists():
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        time.sleep(0.5)


def download():
    (DATA / "ai").mkdir(parents=True, exist_ok=True)
    (DATA / "real").mkdir(parents=True, exist_ok=True)
    if not list((DATA / "ai").glob("mj_*.jpg")):
        urls = rows_api("brivangl/midjourney-v6-llava", "default", "train", 0, 12)
        urls += rows_api("Photoroom/midjourney-v6-recap", "default", "train", 0, 12)
        for i, u in enumerate(urls, 1):
            save(u, DATA / "ai" / f"mj_{i}.jpg")
    if not list((DATA / "real").glob("flickr_*.jpg")):
        for i, u in enumerate(rows_api("nlphuji/flickr30k", "TEST", "test", 100, 16), 1):
            save(u, DATA / "real" / f"flickr_{i}.jpg")
    for title in WIKI:
        path = DATA / "real" / f"wiki_{title}.jpg"
        if not path.exists():
            info = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", headers=UA, timeout=30).json()
            if info.get("thumbnail", {}).get("source"):
                save(info["thumbnail"]["source"], path)


def main():
    download()
    detector = get_image_detector()
    groups = {
        "Midjourney v6 (AI)": sorted((DATA / "ai").glob("mj_*.jpg")),
        "CIFAKE (AI)": sorted((DATA / "ai").glob("cifake_*.jpg")),
        "User portrait (AI)": sorted((DATA / "ai").glob("user_*.jpg")),
        "Wikipedia photos (real)": sorted((DATA / "real").glob("wiki_*.jpg")),
        "Flickr30k photos (real)": sorted((DATA / "real").glob("flickr_*.jpg")),
        "CIFAKE (real)": sorted((DATA / "real").glob("cifake_*.jpg")),
    }
    rules = {"ours": lambda o, m: o, "modern": lambda o, m: m, "average": lambda o, m: (o + m) / 2}
    results = {"groups": {}, "totals": {}}
    totals = {k: {"ai": 0, "real": 0} for k in rules}
    n_ai = n_real = 0
    for name, files in groups.items():
        is_ai = "(AI)" in name
        correct = {k: 0 for k in rules}
        for f in files:
            r = detector.predict(Image.open(f))
            o, m = r["detectors"]["clip_cifake"], r["detectors"]["modern"]
            for k, rule in rules.items():
                correct[k] += (rule(o, m) >= 0.5) == is_ai
        results["groups"][name] = {"n": len(files), **correct}
        for k in rules:
            totals[k]["ai" if is_ai else "real"] += correct[k]
        if is_ai:
            n_ai += len(files)
        else:
            n_real += len(files)
        print(f"{name:26s} n={len(files):2d}  " + "  ".join(f"{k} {correct[k]}" for k in rules))
    for k in rules:
        t = totals[k]
        results["totals"][k] = {"ai_correct": t["ai"], "ai_n": n_ai, "real_correct": t["real"], "real_n": n_real,
                                "correct": t["ai"] + t["real"], "n": n_ai + n_real}
    (OUT / "image_ensemble.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["totals"], indent=1))


if __name__ == "__main__":
    main()
