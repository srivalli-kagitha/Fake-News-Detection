"""Word-level explanations for the text classifier.

Default: occlusion (EXPLAIN_METHOD=occlusion). Each distinct word is removed once and the change in
the Fake probability is that word's weight; one model run per word, batched. Small function words
(the, in, by ...) are not tested. Words after the
classifier's token limit are never read by the model, so they are skipped (weight 0).
Measured on our models (4 test texts, English and Telugu): occlusion was 8x faster than LIME with
200 samples and the most faithful of LIME, Gradient x Input and Integrated Gradients (largest drop
in the verdict probability when its top 3 words are removed, the deletion test of
Atanasova et al., EMNLP 2020). LIME is still available with EXPLAIN_METHOD=lime.

Weights are expressed towards the *Fake* class:
  positive weight  -> the word pushes the prediction towards Fake
  negative weight  -> the word pushes the prediction towards Real
"""
import re

from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config import get_settings
from .text_classifier import TextClassifier

# Split on whitespace only, so Telugu words (with vowel signs) stay intact.
SPLIT = r"\s+"


def explain_text(classifier: TextClassifier, text: str, num_features: int = 12) -> dict:
    if get_settings().explain_method == "lime":
        return lime_explain(classifier, text, num_features)
    return occlusion_explain(classifier, text, num_features)


def _result(text: str, weights: dict, method: str, num_features: int) -> dict:
    tokens = []
    for piece in re.split(r"(\s+)", text):
        if not piece:
            continue
        tokens.append({"text": piece, "weight": float(weights.get(piece, 0.0))})
    top_words = sorted(
        ({"word": w, "weight": float(v)} for w, v in weights.items() if v),
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )[:num_features]
    return {"method": method, "towards": "Fake", "tokens": tokens, "top_words": top_words}


def _content_word(word: str) -> bool:
    """Skip words like 'the', 'in', 'by': they say nothing to the reader and cost a model run each."""
    core = re.sub(r"[^\w]", "", word.lower())
    return bool(core) and core not in ENGLISH_STOP_WORDS


def occlusion_explain(classifier: TextClassifier, text: str, num_features: int = 12) -> dict:
    words = text.split()
    enc = classifier.tokenizer(text, truncation=True, max_length=classifier.max_length, return_offsets_mapping=True)
    read_up_to = max((e for _, e in enc["offset_mapping"]), default=len(text))  # end of the text the model reads
    seen, pos = [], 0
    for w in words:
        start = text.index(w, pos)
        pos = start + len(w)
        if start < read_up_to and w not in seen and _content_word(w):
            seen.append(w)
    variants = [" ".join(x for x in words if x != w) for w in seen]
    probs = classifier.predict_proba([text] + variants, batch_size=16)[:, classifier.fake_index]
    weights = {w: float(probs[0] - probs[i + 1]) for i, w in enumerate(seen)}
    return _result(text, weights, "Occlusion", num_features)


def lime_explain(classifier: TextClassifier, text: str, num_features: int = 12) -> dict:
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
    return _result(text, weights, "LIME", num_features)
