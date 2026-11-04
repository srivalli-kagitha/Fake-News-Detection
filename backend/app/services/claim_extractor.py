"""Extracts check-worthy factual claims (sentences) from a news article."""
import re

from ..config import get_settings
from .common import Lazy, logger

OPINION_STARTS = (
    "i think", "i feel", "i believe", "in my opinion", "maybe", "perhaps", "we think",
    "click", "subscribe", "follow us", "share this", "read more",
)
SENTENCE_END = re.compile(r"(?<=[.!?।॥|])\s+|\n+")
HAS_NUMBER = re.compile(r"\d")
CAPITALISED = re.compile(r"\b[A-Z][a-z]+")
FACT_WORDS = re.compile(
    r"\b(is|are|was|were|has|have|had|will|announced|said|reported|confirmed|declared|launched|"
    r"killed|died|won|lost|banned|approved|increased|decreased|percent|crore|lakh|million|billion)\b",
    re.IGNORECASE,
)


def _load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception as exc:  # spaCy is optional - fall back to regex splitting
        logger.warning("spaCy model not available (%s); using regex sentence splitting.", exc)
        return None


_nlp = Lazy("spaCy English pipeline", _load_spacy)


def _split_sentences(text: str, language: str) -> list[tuple[str, int]]:
    """Returns (sentence, named_entity_count)."""
    nlp = _nlp.get() if language == "en" else None
    if nlp is not None:
        doc = nlp(text)
        return [(s.text.strip(), len(s.ents)) for s in doc.sents if s.text.strip()]
    return [(s.strip(), 0) for s in SENTENCE_END.split(text) if s and s.strip()]


def _score(sentence: str, entities: int, language: str) -> float:
    words = sentence.split()
    if len(words) < (3 if language == "te" else 5) or len(words) > 80:
        return -1
    lower = sentence.lower()
    if sentence.endswith("?") or lower.startswith(OPINION_STARTS):
        return -1
    score = 1.0 + 0.8 * min(entities, 3)
    if HAS_NUMBER.search(sentence):
        score += 1.0
    if language == "en":
        score += 0.3 * min(len(CAPITALISED.findall(sentence)), 4)
        if FACT_WORDS.search(sentence):
            score += 0.7
    return score


def extract_claims(text: str, language: str, max_claims: int | None = None) -> list[str]:
    max_claims = max_claims or get_settings().max_claims
    candidates = []
    for position, (sentence, entities) in enumerate(_split_sentences(text, language)):
        score = _score(sentence, entities, language)
        if score > 0:
            candidates.append((score, position, sentence))

    if not candidates and text.strip():
        # Short input (e.g. a single headline) - treat the whole text as one claim.
        return [text.strip()[:500]]

    best = sorted(candidates, key=lambda c: c[0], reverse=True)[:max_claims]
    seen, claims = set(), []
    for _, _, sentence in sorted(best, key=lambda c: c[1]):  # keep article order
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            claims.append(sentence)
    return claims
