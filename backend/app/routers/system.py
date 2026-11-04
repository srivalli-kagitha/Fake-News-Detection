import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, select

import torch

from ..config import get_settings, is_local_reference, resolve_local
from ..database import CheckRecord, SessionLocal
from ..services.common import get_device
from ..services.evidence import evidence_status
from ..services.image_detector import image_model_status
from ..services.text_classifier import text_model_status

router = APIRouter(tags=["system"])


def _metrics(model_ref: str) -> dict | None:
    """metrics.json written by the training scripts, if the model folder has one."""
    if not is_local_reference(model_ref):
        return None
    path = resolve_local(model_ref) / "metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


@router.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "text_en": text_model_status("en"),
            "text_te": text_model_status("te"),
            "image": image_model_status(),
            "nli_en": settings.nli_en_model,
            "nli_multi": settings.nli_multi_model,
            "embeddings": settings.embed_model,
        },
        "metrics": {
            "text_en": _metrics(settings.text_en_model),
            "text_te": _metrics(settings.text_te_model),
            "image": _metrics(settings.image_model),
        },
        "evidence": {
            "corpus_files": evidence_status()["corpus_files"],
            "fact_check_api": bool(settings.google_factcheck_api_key.strip()),
        },
    }


@router.get("/history")
def list_history(limit: int = 50, kind: str | None = None):
    limit = max(1, min(limit, 500))
    with SessionLocal() as db:
        query = select(CheckRecord).order_by(CheckRecord.id.desc()).limit(limit)
        if kind:
            query = query.where(CheckRecord.kind == kind)
        return [r.to_dict() for r in db.scalars(query)]


@router.delete("/history/{record_id}")
def delete_history_item(record_id: int):
    with SessionLocal() as db:
        record = db.get(CheckRecord, record_id)
        if record is None:
            raise HTTPException(404, "Record not found.")
        db.delete(record)
        db.commit()
    return {"deleted": record_id}


@router.delete("/history")
def clear_history():
    with SessionLocal() as db:
        db.execute(delete(CheckRecord))
        db.commit()
    return {"cleared": True}
