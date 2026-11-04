"""Convert a fake-news dataset into the standard format used by train_text_classifier.py.

Output CSV columns:  text,label     (label: 0 = Real, 1 = Fake)

Examples
--------
ISOT (two files, Fake.csv and True.csv):
    python prepare_dataset.py isot --fake Fake.csv --real True.csv --out data/en.csv

Any single CSV (WELFake, Telugu datasets, your own data). Say which value means FAKE:
    python prepare_dataset.py generic --input WELFake_Dataset.csv --text-cols title text \
        --label-col label --fake-value 1 --remove-patterns welfake --out data/en.csv
    (WELFake: the actual file has 1 = fake, 0 = real - the opposite of its documentation.
     Always check a few rows yourself: Reuters / NYT articles must end up as label 0 = Real.)

Folders of .txt files (e.g. the Zenodo Telugu dataset, CC BY 4.0, DOI 10.5281/zenodo.11408513):
    python prepare_dataset.py folders --fake-dir data/te/Telugu_fake_news --real-dir data/te/Telugu_real_news --out data/te.csv

Split into train / test (stratified, 80/20):
    python prepare_dataset.py split --input data/en.csv --test-size 0.2
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Source giveaways that let a model "cheat" (it learns the publisher, not whether the news is fake).
PATTERN_PRESETS = {
    "welfake": [
        r"^[^\n]{0,120}?\(Reuters\)\s*[-\u2013\u2014]?\s*",   # "WASHINGTON (Reuters) - " dateline
        r"\(?\bReuters\b\)?",
        r"\s[-\u2013\u2014]\s(The New York Times|Breitbart|The Onion)\b",
        r"Featured (image|photo)[^.\n]{0,120}",
        r"\b(Getty Images|21st Century Wire|21WIRE)\b",
        r"\bBreitbart( News)?\b",
        r"Follow (him|her|them|us|me)?\s*on Twitter[^.\n]{0,80}",
    ],
}


def remove_patterns(series: pd.Series, patterns: list[str]) -> pd.Series:
    for pat in patterns:
        for regex in PATTERN_PRESETS.get(pat, [pat]):
            series = series.str.replace(regex, " ", regex=True, flags=re.IGNORECASE | re.MULTILINE)
    return series


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= 20]
    df = df.drop_duplicates(subset="text")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def join_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].fillna("").astype(str).agg(". ".join, axis=1)


def report(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(out, index=False, encoding="utf-8")
    counts = df["label"].value_counts().to_dict()
    print(f"Saved {len(df)} rows to {out}  (Real={counts.get(0, 0)}, Fake={counts.get(1, 0)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    isot = sub.add_parser("isot")
    isot.add_argument("--fake", required=True)
    isot.add_argument("--real", required=True)
    isot.add_argument("--out", required=True)

    gen = sub.add_parser("generic")
    gen.add_argument("--input", required=True)
    gen.add_argument("--text-cols", nargs="+", required=True)
    gen.add_argument("--label-col", required=True)
    gen.add_argument("--fake-value", required=True, help="value in label column that means FAKE")
    gen.add_argument("--out", required=True)
    gen.add_argument("--remove-patterns", nargs="*", default=[],
                     help="regexes (or preset name 'welfake') to delete from every text")
    gen.add_argument("--max-rows", type=int, default=0, help="keep a balanced random sample (0 = all)")

    folders = sub.add_parser("folders")
    folders.add_argument("--fake-dir", required=True)
    folders.add_argument("--real-dir", required=True)
    folders.add_argument("--out", required=True)
    folders.add_argument("--remove-words", nargs="*", default=[],
                         help="words to delete from every text (stops the model cheating on giveaway words)")

    split = sub.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    if args.cmd == "isot":
        fake, real = pd.read_csv(args.fake), pd.read_csv(args.real)
        fake["label"], real["label"] = 1, 0
        df = pd.concat([fake, real], ignore_index=True)
        df["text"] = join_cols(df, ["title", "text"])
        report(clean(df), Path(args.out))

    elif args.cmd == "generic":
        df = pd.read_csv(args.input)
        df["text"] = join_cols(df, args.text_cols)
        df["label"] = (df[args.label_col].astype(str).str.strip() == str(args.fake_value)).astype(int)
        if args.remove_patterns:
            df["text"] = remove_patterns(df["text"], args.remove_patterns)
            print(f"Removed source giveaways: {', '.join(args.remove_patterns)}")
        df = clean(df)
        if args.max_rows and len(df) > args.max_rows:
            per = args.max_rows // 2
            df = pd.concat([g.sample(n=min(per, len(g)), random_state=42) for _, g in df.groupby("label")])
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        report(df, Path(args.out))

    elif args.cmd == "folders":
        rows = []
        for label, folder in ((1, args.fake_dir), (0, args.real_dir)):
            files = sorted(Path(folder).rglob("*.txt"))
            if not files:
                raise SystemExit(f"No .txt files found in {folder}")
            for f in files:
                text = f.read_text(encoding="utf-8", errors="replace").replace("|", " ")
                rows.append({"text": text, "label": label})
        df = pd.DataFrame(rows)
        if args.remove_words:
            pattern = "|".join(re.escape(w) for w in args.remove_words)
            df["text"] = df["text"].str.replace(pattern, " ", regex=True)
            print(f"Removed giveaway words: {', '.join(args.remove_words)}")
        report(clean(df), Path(args.out))

    elif args.cmd == "split":
        path = Path(args.input)
        df = pd.read_csv(path)
        train, test = train_test_split(df, test_size=args.test_size, stratify=df["label"], random_state=42)
        train.to_csv(path.with_name(path.stem + "_train.csv"), index=False)
        test.to_csv(path.with_name(path.stem + "_test.csv"), index=False)
        print(f"train: {len(train)} rows, test: {len(test)} rows")


if __name__ == "__main__":
    main()
