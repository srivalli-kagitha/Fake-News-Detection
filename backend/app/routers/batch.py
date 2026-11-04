"""Batch processing: a CSV of news texts, or a ZIP / several image files."""
import io
import uuid
import zipfile
from collections import Counter
from pathlib import PurePosixPath

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import get_settings
from ..database import save_check
from ..services.claim_verifier import verify_text
from ..services.image_detector import get_image_detector
from ..services.language import resolve_language
from ..services.text_classifier import get_text_classifier
from .image import read_image

router = APIRouter(tags=["batch"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_COLUMN_NAMES = ("text", "news", "content", "article", "title", "headline")


def _read_csv(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise HTTPException(400, f"Could not read CSV: {exc}")
    raise HTTPException(400, "Could not decode the CSV file. Save it as UTF-8.")


def _save_output(df: pd.DataFrame, prefix: str) -> str:
    job_id = f"{prefix}_{uuid.uuid4().hex[:10]}"
    df.to_csv(get_settings().outputs_dir / f"{job_id}.csv", index=False, encoding="utf-8-sig")
    return job_id


@router.post("/batch/text")
def batch_text(file: UploadFile = File(...), language: str = Form("auto"), verify_claims: bool = Form(False)):
    settings = get_settings()
    df = _read_csv(file.file.read())
    column = next((c for c in df.columns if str(c).strip().lower() in TEXT_COLUMN_NAMES), None)
    if column is None:
        raise HTTPException(400, f"CSV needs a column named one of: {', '.join(TEXT_COLUMN_NAMES)}.")

    df = df[df[column].notna()].head(settings.max_batch_rows).reset_index(drop=True)
    texts = df[column].astype(str).tolist()
    if not texts:
        raise HTTPException(400, "The CSV has no text rows.")
    if "language" in df.columns:
        languages = [resolve_language(t, str(l)) for t, l in zip(texts, df["language"])]
    else:
        languages = [resolve_language(t, language) for t in texts]

    predictions: list[dict | None] = [None] * len(texts)
    for lang in sorted(set(languages)):
        idx = [i for i, l in enumerate(languages) if l == lang]
        outputs = get_text_classifier(lang).predict_many([texts[i] for i in idx])
        for i, out in zip(idx, outputs):
            predictions[i] = out

    rows = []
    for i, (text, lang, pred) in enumerate(zip(texts, languages, predictions)):
        row = {
            "row": i + 1,
            "text": text,
            "language": lang,
            "prediction": pred["label"],
            "confidence": round(pred["confidence"], 4),
            "fake_probability": round(pred["fake_probability"], 4),
        }
        if verify_claims:
            summary = verify_text(text, lang)["summary"]
            row["claim_verdict"] = summary["verdict"]
            row["credibility_score"] = summary["credibility_score"]
        rows.append(row)

    out = pd.DataFrame(rows)
    job_id = _save_output(out, "text")
    counts = dict(Counter(out["prediction"]))
    save_check(
        kind="batch_text",
        input_preview=f"{file.filename} ({len(rows)} rows)",
        verdict=", ".join(f"{k}: {v}" for k, v in counts.items()),
        details={"job_id": job_id, "counts": counts},
    )
    return {"job_id": job_id, "total": len(rows), "counts": counts, "results": rows}


@router.post("/batch/images")
def batch_images(files: list[UploadFile] = File(...)):
    settings = get_settings()
    images: list[tuple[str, bytes]] = []
    for upload in files:
        data = upload.file.read()
        name = upload.filename or "upload"
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        path = PurePosixPath(info.filename)
                        if (
                            not info.is_dir()
                            and path.suffix.lower() in IMAGE_EXTENSIONS
                            and not path.name.startswith(".")
                            and "__MACOSX" not in path.parts
                        ):
                            images.append((info.filename, zf.read(info)))
            except zipfile.BadZipFile:
                raise HTTPException(400, f"{name} is not a valid ZIP file.")
        else:
            images.append((name, data))

    images = images[: settings.max_batch_rows]
    if not images:
        raise HTTPException(400, "No images found. Upload .jpg/.png/.webp files or a ZIP containing them.")

    detector = get_image_detector()
    rows = []
    for i, (name, data) in enumerate(images):
        try:
            pred = detector.predict(read_image(data, name))
            rows.append(
                {
                    "row": i + 1,
                    "filename": name,
                    "prediction": pred["label"],
                    "confidence": round(pred["confidence"], 4),
                    "fake_probability": round(pred["fake_probability"], 4),
                }
            )
        except HTTPException as exc:
            rows.append({"row": i + 1, "filename": name, "prediction": "Error", "confidence": None,
                         "fake_probability": None, "error": exc.detail})

    out = pd.DataFrame(rows)
    job_id = _save_output(out, "images")
    counts = dict(Counter(out["prediction"]))
    save_check(
        kind="batch_image",
        input_preview=f"{len(rows)} images",
        verdict=", ".join(f"{k}: {v}" for k, v in counts.items()),
        details={"job_id": job_id, "counts": counts},
    )
    return {"job_id": job_id, "total": len(rows), "counts": counts, "results": rows}


@router.get("/batch/{job_id}/download")
def download(job_id: str):
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "Invalid job id.")
    path = get_settings().outputs_dir / f"{job_id}.csv"
    if not path.exists():
        raise HTTPException(404, "Result file not found.")
    return FileResponse(path, media_type="text/csv", filename=f"truthlens_{job_id}.csv")
