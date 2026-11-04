"""Train the Real vs AI-generated image detector: frozen CLIP ViT-L/14 + linear head.

Dataset folder layout - either with train/test folders (Kaggle CIFAKE):
    data/cifake/train/REAL/*.jpg     data/cifake/train/FAKE/*.jpg
    data/cifake/test/REAL/*.jpg      data/cifake/test/FAKE/*.jpg
or just one folder per class (Hugging Face CIFAKE) - then 20% is held out as the test set automatically:
    data/cifake/real/*.jpg           data/cifake/fake/*.jpg
Folder names are matched case-insensitively (REAL / real).

    python train_image_detector.py --data data/cifake --out outputs/image_detector

Outputs in --out: config.json + head.pt (copy the folder to backend/models/image_detector),
metrics.json, classification_report.txt, confusion_matrix.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

LABELS = ["Real", "AI-Generated"]  # index 0 = real, 1 = fake
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing train/ and test/, or real/ and fake/")
    p.add_argument("--out", required=True)
    p.add_argument("--clip", default="openai/clip-vit-large-patch14")
    p.add_argument("--real-dir", default="REAL")
    p.add_argument("--fake-dir", default="FAKE")
    p.add_argument("--max-per-class", type=int, default=0, help="limit images per class (0 = all)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-fraction", type=float, default=0.2, help="test share when there is no test/ folder")
    return p.parse_args()


def find_dir(parent: Path, name: str) -> Path | None:
    for child in parent.iterdir() if parent.is_dir() else []:
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def class_files(split_dir: Path, real: str, fake: str, limit: int) -> dict[int, list[Path]]:
    result = {}
    for label, sub in ((0, real), (1, fake)):
        folder = find_dir(split_dir, sub)
        files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXTS) if folder else []
        if not files:
            raise SystemExit(f"No images found in {split_dir / sub}")
        if limit:
            rng = np.random.default_rng(42)
            files = sorted(rng.choice(files, size=min(limit, len(files)), replace=False))
        result[label] = list(files)
    return result


def load_splits(data: Path, real: str, fake: str, limit: int, test_fraction: float):
    train_dir, test_dir = find_dir(data, "train"), find_dir(data, "test")
    if train_dir and test_dir:
        to_items = lambda d: [(f, l) for l, fs in d.items() for f in fs]
        test_limit = max(1, int(limit * test_fraction / (1 - test_fraction))) if limit else 0
        return (to_items(class_files(train_dir, real, fake, limit)),
                to_items(class_files(test_dir, real, fake, test_limit)))
    # no train/test folders: stratified random split
    total = int(limit / (1 - test_fraction)) if limit else 0
    train, test = [], []
    rng = np.random.default_rng(42)
    for label, files in class_files(data, real, fake, total).items():
        files = list(rng.permutation(files))
        n_test = max(1, int(len(files) * test_fraction))
        test += [(f, label) for f in files[:n_test]]
        train += [(f, label) for f in files[n_test:]]
    print(f"No train/test folders - held out {test_fraction:.0%} as the test set.")
    return train, test


@torch.no_grad()
def embed(items, clip, processor, device, batch_size):
    feats, labels = [], []
    for i in tqdm(range(0, len(items), batch_size), desc="CLIP features"):
        batch = items[i : i + batch_size]
        images = [Image.open(f).convert("RGB") for f, _ in batch]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            emb = clip(pixel_values=pixels).image_embeds.float()
        feats.append(nn.functional.normalize(emb, dim=-1).cpu())
        labels += [l for _, l in batch]
    return torch.cat(feats), torch.tensor(labels)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    processor = CLIPImageProcessor.from_pretrained(args.clip)
    clip = CLIPVisionModelWithProjection.from_pretrained(args.clip).to(device).eval()

    data = Path(args.data)
    train_items, test_items = load_splits(data, args.real_dir, args.fake_dir, args.max_per_class, args.test_fraction)
    print(f"train images: {len(train_items)}  test images: {len(test_items)}")

    x_train, y_train = embed(train_items, clip, processor, device, args.batch_size)
    x_test, y_test = embed(test_items, clip, processor, device, args.batch_size)

    # hold out 10% of train for validation
    perm = torch.randperm(len(x_train))
    n_val = max(1, len(perm) // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    head = nn.Linear(x_train.shape[1], len(LABELS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = x_train[tr_idx].to(device), y_train[tr_idx].to(device)
    xv, yv = x_train[val_idx].to(device), y_train[val_idx].to(device)

    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(order), 256):
            idx = order[i : i + 256]
            opt.zero_grad()
            loss = loss_fn(head(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(xv).argmax(-1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        print(f"epoch {epoch + 1:02d}  loss {loss.item():.4f}  val_acc {val_acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(x_test.to(device)), -1).cpu().numpy()
    y_true, y_pred = y_test.numpy(), probs.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs[:, 1])), 4),
        "model": args.clip,
        "test_images": int(len(y_true)),
    }
    print(json.dumps(metrics, indent=2))

    torch.save(best_state, out / "head.pt")
    (out / "config.json").write_text(json.dumps({"clip_model": args.clip, "labels": LABELS}, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    )
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABELS).plot(cmap="Blues")
    plt.title("Confusion matrix - CLIP ViT-L/14 detector")
    plt.savefig(out / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nDone. Copy the folder '{out}' to backend/models/image_detector")


if __name__ == "__main__":
    main()
