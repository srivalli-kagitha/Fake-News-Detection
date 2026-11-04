"""Grow the evidence corpus with free Wikipedia article summaries.

Usage (from the backend folder, with the venv active):
    python -m scripts.add_wikipedia_evidence "Chandrayaan-3" "COVID-19 vaccine" "Telangana"
    python -m scripts.add_wikipedia_evidence --file topics.txt          # one title per line
    python -m scripts.add_wikipedia_evidence --lang te "తెలంగాణ"        # Telugu Wikipedia

Each summary is split into sentences and appended to data/evidence/wikipedia_<lang>.jsonl.
"""
import argparse
import json
import re
import urllib.parse

import requests

from app.config import get_settings

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
HEADERS = {"User-Agent": "TruthLensAI-student-project/1.0 (educational use)"}


def fetch_summary(title: str, lang: str) -> tuple[str, str] | None:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None
    data = resp.json()
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return data.get("extract", ""), page_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("titles", nargs="*")
    parser.add_argument("--file", help="text file with one Wikipedia title per line")
    parser.add_argument("--lang", default="en", help="Wikipedia language code (en, te, ...)")
    args = parser.parse_args()

    titles = list(args.titles)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            titles += [line.strip() for line in fh if line.strip()]
    if not titles:
        parser.error("give at least one title")

    out_file = get_settings().evidence_path / f"wikipedia_{args.lang}.jsonl"
    added = 0
    with out_file.open("a", encoding="utf-8") as out:
        for title in titles:
            result = fetch_summary(title, args.lang)
            if not result or not result[0]:
                print(f"  not found: {title}")
                continue
            extract, page_url = result
            for sentence in SENTENCE_END.split(extract):
                if len(sentence.split()) >= 4:
                    out.write(json.dumps({"text": sentence.strip(), "source": f"Wikipedia ({args.lang})",
                                          "url": page_url}, ensure_ascii=False) + "\n")
                    added += 1
            print(f"  added: {title}")
    print(f"Added {added} passages to {out_file}. The index rebuilds automatically on next use.")


if __name__ == "__main__":
    main()
