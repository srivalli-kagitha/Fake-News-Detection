"""Build a Telugu fake-news dataset by machine-translating a balanced English sample.

Why: the public Telugu fake-news dataset compares *translated fact-check reports* (fake) with *native
health articles* (real), so a model learns the writing style, not fake vs real. Translating fake AND real
English news with the same model gives both classes the same style, so the model must learn the content.

    python translate_dataset.py --input data/en.csv --out data/te.csv --rows 10000

Model: facebook/nllb-200-distilled-1.3B (free, no login, CC BY-NC 4.0: fine for academic use).
Each article is shortened to its headline + first sentences (--max-words) and translated sentence by
sentence. Progress is saved every few batches, so after a disconnect just run the same command again.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="English CSV with text,label (from prepare_dataset.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=10000, help="balanced number of articles to translate")
    p.add_argument("--max-words", type=int, default=120, help="words kept per article before translating")
    p.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--save-every", type=int, default=20, help="save progress every N batches")
    return p.parse_args()


def shorten(text: str, max_words: int) -> list[str]:
    """Headline + leading sentences, at most max_words words, as a list of sentences."""
    sentences, words = [], 0
    for s in SENTENCE_END.split(str(text).strip()):
        s = s.strip()
        if not s:
            continue
        n = len(s.split())
        if n > 60:  # very long "sentence" (often no punctuation) - cut it
            s, n = " ".join(s.split()[:60]), 60
        if sentences and words + n > max_words:
            break
        sentences.append(s)
        words += n
    return sentences


def main():
    args = parse_args()
    out = Path(args.out)
    partial = out.with_suffix(".partial.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.input)[["text", "label"]].dropna()
    per_class = args.rows // 2
    df = pd.concat([g.sample(n=min(per_class, len(g)), random_state=42) for _, g in df.groupby("label")])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["doc_id"] = range(len(df))

    # one row per sentence
    segments = [(doc, i, s) for doc, text in zip(df.doc_id, df.text) for i, s in enumerate(shorten(text, args.max_words))]
    seg = pd.DataFrame(segments, columns=["doc_id", "sent_id", "en"])

    done = pd.read_csv(partial) if partial.exists() else pd.DataFrame(columns=["doc_id", "sent_id", "te"])
    done_keys = set(zip(done.doc_id, done.sent_id))
    todo = seg[[k not in done_keys for k in zip(seg.doc_id, seg.sent_id)]]
    print(f"{len(df)} articles → {len(seg)} sentences ({len(todo)} left to translate) on {device}")

    if len(todo):
        tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device).eval()
        tel = tok.convert_tokens_to_ids("tel_Telu")

        todo = todo.iloc[todo.en.str.len().argsort()]  # similar lengths together = faster batches
        results, batches = [], 0
        for start in tqdm(range(0, len(todo), args.batch_size), desc="Translating"):
            batch = todo.iloc[start : start + args.batch_size]
            enc = tok(list(batch.en), return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=tel, max_new_tokens=256, num_beams=1)
            te = tok.batch_decode(gen, skip_special_tokens=True)
            results.append(pd.DataFrame({"doc_id": batch.doc_id.values, "sent_id": batch.sent_id.values, "te": te}))
            batches += 1
            if batches % args.save_every == 0:
                done = pd.concat([done, *results], ignore_index=True)
                results = []
                done.to_csv(partial, index=False)
        done = pd.concat([done, *results], ignore_index=True)
        done.to_csv(partial, index=False)

    done = done.sort_values(["doc_id", "sent_id"])
    telugu = done.groupby("doc_id")["te"].apply(lambda s: " ".join(map(str, s)))
    result = df.set_index("doc_id").join(telugu.rename("te"), how="inner")
    result = result[result.te.str.len() > 20]
    out.parent.mkdir(parents=True, exist_ok=True)
    result[["te", "label", "text"]].rename(columns={"te": "text", "text": "english_source"}).to_csv(
        out, index=False, encoding="utf-8"
    )
    counts = result.label.value_counts().to_dict()
    print(f"Saved {len(result)} Telugu articles to {out}  (Real={counts.get(0, 0)}, Fake={counts.get(1, 0)})")
    print("Example:", result.te.iloc[0][:200])


if __name__ == "__main__":
    main()
