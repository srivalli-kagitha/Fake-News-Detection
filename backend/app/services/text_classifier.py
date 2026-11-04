"""Fake / Real news classification with fine-tuned transformers.

English : DeBERTa-v3-large  (training/train_text_classifier.py -> backend/models/text_en)
Telugu  : IndicBERT v2 / mmBERT (training/train_text_classifier.py -> backend/models/text_te)
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings, is_local_reference, resolve_local
from .common import Lazy, ModelNotReadyError, get_device

DEFAULT_LABELS = {0: "Real", 1: "Fake"}


class TextClassifier:
    def __init__(self, model_ref: str, language: str):
        settings = get_settings()
        self.language = language
        self.max_length = settings.max_text_length
        self.device = get_device()

        if is_local_reference(model_ref):
            path = resolve_local(model_ref)
            if not (path / "config.json").exists():
                raise ModelNotReadyError(
                    f"The {language.upper()} text model is not trained yet. "
                    f"Train it with training/train_text_classifier.py and copy the output folder to "
                    f"backend/{model_ref} (see training/README.md)."
                )
            model_ref = str(path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ref)
        self.model.to(self.device).eval()

        id2label = self.model.config.id2label or {}
        labels = {}
        for idx in range(self.model.config.num_labels):
            name = str(id2label.get(idx, DEFAULT_LABELS.get(idx, f"LABEL_{idx}")))
            if name.upper().startswith("LABEL_"):
                name = DEFAULT_LABELS.get(idx, name)
            labels[idx] = name.capitalize()
        self.labels = [labels[i] for i in range(len(labels))]
        self.fake_index = next((i for i, n in enumerate(self.labels) if n.lower() == "fake"), 1)

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict(self, text: str) -> dict:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[dict]:
        probs = self.predict_proba(texts)
        results = []
        for row in probs:
            idx = int(row.argmax())
            results.append(
                {
                    "label": self.labels[idx],
                    "confidence": float(row[idx]),
                    "probabilities": {self.labels[i]: float(p) for i, p in enumerate(row)},
                    "fake_probability": float(row[self.fake_index]),
                }
            )
        return results


_settings = get_settings()
_classifiers = {
    "en": Lazy("English text classifier", lambda: TextClassifier(_settings.text_en_model, "en")),
    "te": Lazy("Telugu text classifier", lambda: TextClassifier(_settings.text_te_model, "te")),
}


def get_text_classifier(language: str) -> TextClassifier:
    return _classifiers[language].get()


def text_model_status(language: str) -> str:
    lazy = _classifiers[language]
    if lazy.loaded:
        return "loaded"
    ref = _settings.text_en_model if language == "en" else _settings.text_te_model
    if is_local_reference(ref):
        return "ready" if (resolve_local(ref) / "config.json").exists() else "not_trained"
    return "ready"
