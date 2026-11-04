"""Dataset bias check: how well does a simple word-count model (TF-IDF + logistic regression) do?

If a model that does not understand language at all scores ~100%, the dataset has a shortcut
(publisher names, writing style, topic) and a high transformer score means little. Report this
table in the paper's dataset section.

    python bias_check.py --data data/en.csv data/te.csv --out outputs/bias_check.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def check(path: str, max_rows: int) -> str:
    df = pd.read_csv(path)[["text", "label"]].dropna()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0)
    if len(df) < 20 or df.label.nunique() < 2:
        return f"### {Path(path).name}\n\n- Skipped: only {len(df)} rows (need at least 20 with both labels).\n"
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3 if len(tr) >= 500 else 1, max_features=50000)
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(tr.text), tr.label)
    pred = clf.predict(vec.transform(te.text))
    weights, names = clf.coef_[0], np.array(vec.get_feature_names_out())
    fake = ", ".join(names[np.argsort(-weights)[:12]])
    real = ", ".join(names[np.argsort(weights)[:12]])
    return (f"### {Path(path).name}\n\n"
            f"- Rows used: {len(df)}\n"
            f"- TF-IDF + Logistic Regression: accuracy **{accuracy_score(te.label, pred):.4f}**, "
            f"F1 **{f1_score(te.label, pred):.4f}**\n"
            f"- Strongest *fake* words: {fake}\n"
            f"- Strongest *real* words: {real}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-rows", type=int, default=30000)
    args = p.parse_args()
    report = "## Dataset bias check (simple word model)\n\n" + "\n".join(check(d, args.max_rows) for d in args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
