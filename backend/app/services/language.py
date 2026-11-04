"""Very small language detector: Telugu script vs everything else (English)."""
import re

TELUGU_CHARS = re.compile(r"[ఀ-౿]")
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)

SUPPORTED = {"en": "English", "te": "Telugu"}


def detect_language(text: str) -> str:
    telugu = len(TELUGU_CHARS.findall(text))
    letters = len(LETTERS.findall(text)) or 1
    return "te" if telugu / letters > 0.3 else "en"


def resolve_language(text: str, requested: str | None) -> str:
    if requested and requested.lower() in SUPPORTED:
        return requested.lower()
    return detect_language(text)
