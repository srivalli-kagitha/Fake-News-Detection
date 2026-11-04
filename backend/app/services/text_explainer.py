"""LIME explanations for the text classifier.

Weights are expressed towards the *Fake* class:
  positive weight  -> the word pushes the prediction towards Fake
  negative weight  -> the word pushes the prediction towards Real
"""
import re

from lime.lime_text import LimeTextExplainer

from ..config import get_settings
from .text_classifier import TextClassifier

# Split on whitespace only, so Telugu words (with vowel signs) stay intact.
SPLIT = r"\s+"


def explain_text(classifier: TextClassifier, text: str, num_features: int = 12) -> dict:
    settings = get_settings()
    explainer = LimeTextExplainer(
        class_names=classifier.labels,
        split_expression=SPLIT,
        bow=True,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        text,
        lambda texts: classifier.predict_proba(list(texts), batch_size=16),
        labels=(classifier.fake_index,),
        num_features=num_features,
        num_samples=settings.lime_num_samples,
    )
    weights = dict(explanation.as_list(label=classifier.fake_index))

    tokens = []
    for piece in re.split(r"(\s+)", text):
        if not piece:
            continue
        tokens.append({"text": piece, "weight": float(weights.get(piece, 0.0))})

    top_words = sorted(
        ({"word": w, "weight": float(v)} for w, v in weights.items()),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )
    return {"method": "LIME", "towards": "Fake", "tokens": tokens, "top_words": top_words}
