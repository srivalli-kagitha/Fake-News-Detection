"""Fine-tune a transformer for Fake / Real news classification.

Main English model (proposed):
    python train_text_classifier.py --model microsoft/deberta-v3-large \
        --train data/en_train.csv --test data/en_test.csv --out outputs/text_en

English baseline (for the comparison table in the report / paper):
    python train_text_classifier.py --model bert-base-uncased \
        --train data/en_train.csv --test data/en_test.csv --out outputs/bert_baseline --lr 2e-5 --batch-size 16

Telugu model:
    python train_text_classifier.py --model ai4bharat/IndicBERTv2-MLM-only \
        --train data/te_train.csv --test data/te_test.csv --out outputs/text_te --lr 3e-5 --batch-size 16

Input CSVs: columns `text,label` with label 0 = Real, 1 = Fake (see prepare_dataset.py).
Outputs in --out: the model (copy this folder to backend/models/...), metrics.json,
classification_report.txt, confusion_matrix.png, training_curve.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = {0: "Real", 1: "Fake"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/deberta-v3-large")
    p.add_argument("--train", required=True)
    p.add_argument("--test", help="test CSV; if omitted, 15%% of --train is held out")
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-train-rows", type=int, default=0, help="limit rows for a quick run (0 = all)")
    p.add_argument("--max-test-rows", type=int, default=0, help="limit test rows for a quick run (0 = all)")
    p.add_argument("--no-fp16", action="store_true", help="disable mixed precision (use if loss becomes NaN)")
    p.add_argument("--checkpoint-dir", default="", help="where temporary checkpoints go (default: <out>/checkpoints)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_df = load(args.train)
    if args.max_train_rows:
        train_df = train_df.sample(n=min(args.max_train_rows, len(train_df)), random_state=args.seed)
    if args.test:
        test_df = load(args.test)
    else:
        train_df, test_df = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=args.seed)
    if args.max_test_rows and len(test_df) > args.max_test_rows:
        test_df, _ = train_test_split(test_df, train_size=args.max_test_rows, stratify=test_df["label"],
                                      random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=args.seed)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    def to_ds(df):
        return Dataset.from_pandas(df.reset_index(drop=True)).map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir or str(out / "checkpoints"),
        save_only_model=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # ---------- final evaluation on the held-out test set ----------
    pred = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(pred.predictions, dtype=torch.float32), dim=-1).numpy()
    y_true, y_pred = pred.label_ids, probs.argmax(-1)
    metrics = compute_metrics((pred.predictions, y_true))
    metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    metrics.update({"model": args.model, "test_rows": int(len(y_true)), "max_length": args.max_length})
    print(json.dumps(metrics, indent=2))

    # ---------- save model + report artefacts ----------
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    )

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Real", "Fake"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix - {args.model.split('/')[-1]}")
    plt.savefig(out / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    history = [h for h in trainer.state.log_history if "loss" in h and "epoch" in h]
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    if history:
        plt.plot([h["epoch"] for h in history], [h["loss"] for h in history], label="train loss")
        plt.plot([h["epoch"] for h in evals], [h["eval_loss"] for h in evals], "o-", label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(out / "training_curve.png", dpi=200, bbox_inches="tight")
        plt.close()

    import shutil

    shutil.rmtree(args.checkpoint_dir or (out / "checkpoints"), ignore_errors=True)
    print(f"\nDone. Copy the folder '{out}' to backend/models/ (e.g. backend/models/text_en).")


if __name__ == "__main__":
    main()
