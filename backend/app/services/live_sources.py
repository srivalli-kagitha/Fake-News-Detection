"""Live evidence from the internet, fetched per claim (no API key needed).

1. Wikipedia search (English or Telugu): the intro paragraphs of the best-matching articles,
   split into sentences.
2. Google News RSS: recent news headlines from Indian and international outlets.

The candidates are ranked by BGE-M3 similarity to the claim in claim_verifier, exactly like
local corpus passages. If the machine is offline every function returns [] and the system
falls back to the local corpus. Results are cached per claim for the life of the server.
"""
import re
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests

from .common import logger

HEADERS = {"User-Agent": "TruthLensAI-student-project/1.0 (https://github.com/srivalli-kagitha/Fake-News-Detection)"}
WIKI_SLOTS = threading.Semaphore(2)  # Wikipedia answers 429 when too many requests arrive at once


class SearchFailed(Exception):
    """Raised inside cached functions so that a failed search is not cached as 'no results'."""
TIMEOUT = 8
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
# A "sentence" ending like this is an abbreviation (U.S., Dr., Mr., St.), not a sentence end.
ABBREVIATION_END = re.compile(r"(\b(?:[A-Z]\.)+|\b(?:Dr|Mr|Mrs|Ms|St|Jr|Sr|Prof|Gen|Lt|Col|Rev|No|vs|Inc|Ltd|Co|approx))\.?$")


def split_sentences(text: str) -> list[str]:
    out = []
    for part in SENTENCE_END.split(text):
        if out and ABBREVIATION_END.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out
WORD = re.compile(r"[\w\u0C00-\u0C7F]+", re.UNICODE)  # Telugu vowel signs are not \w

STOPWORDS = set(
    """a an the and or but if then than that this these those is are was were be been being am
    has have had do does did will would shall should can could may might must of in on at to for
    from by with about as into over after before under again further once here there when where
    why how all any both each few more most other some such no nor not only own same so too very
    just also it its they them their he she his her we our you your i me my said says say new
    according reported report reports officials official confirmed claims claim says monday
    tuesday wednesday thursday friday saturday sunday today yesterday now share forward""".split()
)


def keywords(claim: str, limit: int = 8) -> str:
    """Content words of the claim, used as a search query (search engines dislike full sentences)."""
    words, seen = [], set()
    for w in WORD.findall(claim):
        lw = w.lower()
        if "_" in lw or lw in STOPWORDS or lw in seen or (len(lw) < 3 and not lw.isdigit() and not w.isupper()):
            continue
        seen.add(lw)
        words.append(w)
    return " ".join(words[:limit])


def _get(url: str, params: dict, retries: int = 2) -> requests.Response:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                logger.warning("Live search failed (%s): %s", url.split("/")[2], exc)
    raise SearchFailed(url)


@lru_cache(maxsize=512)
def _wikipedia(query: str, lang: str, pages: int) -> tuple:
    """One request: search for the query and return the intro of the best-matching pages."""
    with WIKI_SLOTS:
        resp = _get(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": pages,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max", "redirects": 1,
            "format": "json", "formatversion": 2})
    out = []
    pages_found = sorted(resp.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    for page in pages_found:
        title = page.get("title", "")
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        sentences = [clean_sentence(x) for x in split_sentences(page.get("extract", ""))[:25]]
        sentences = [x for x in sentences if 30 <= len(x) <= 600]
        lead = sentences[0] if sentences else ""
        for i, sent in enumerate(sentences):
            # NLI reads each sentence after the article's first sentence, so "he" or "the party"
            # in a later sentence has a name to refer to.
            context = sent if i == 0 else f"{lead} {sent}"
            out.append({"text": sent, "nli_text": context, "source": f"Wikipedia - {title}", "url": url,
                        "type": "wikipedia"})
    return tuple(out)


PARENTHESES = re.compile(r"\s*\([^()]*\)")


def clean_sentence(text: str) -> str:
    """Removes pronunciation guides and other brackets, e.g. 'Naidu (Telugu pronunciation: [...]; born 1950)'."""
    text = " ".join(text.split())
    for _ in range(2):  # nested brackets
        text = PARENTHESES.sub("", text)
    return text.replace(" ,", ",").strip()


def search_wikipedia(claim: str, language: str, pages: int = 2) -> list[dict]:
    query = keywords(claim)
    if not query:
        return []
    try:
        return [dict(d) for d in _wikipedia(query, "te" if language == "te" else "en", pages)]
    except SearchFailed:
        return []


@lru_cache(maxsize=512)
def _news(query: str, lang: str, limit: int) -> tuple:
    params = {"q": query, "hl": "te" if lang == "te" else "en-IN", "gl": "IN",
              "ceid": "IN:te" if lang == "te" else "IN:en"}
    resp = _get("https://news.google.com/rss/search", params)
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return ()
    out = []
    for item in list(root.iter("item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        publisher = (item.findtext("source") or "").strip()
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(publisher) - 3]
        if len(title) < 20:
            continue
        date = (item.findtext("pubDate") or "")[5:16]
        out.append({"text": title, "source": f"News - {publisher or 'Google News'}" + (f", {date}" if date else ""),
                    "url": item.findtext("link") or "", "type": "news"})
    return tuple(out)


def search_news(claim: str, language: str, limit: int = 10) -> list[dict]:
    query = keywords(claim)
    if not query:
        return []
    try:
        return [dict(d) for d in _news(query, language, limit)]
    except SearchFailed:
        return []


DEBUNK_HEADLINE = re.compile(
    r"\b(fake|false|falsely|hoax|scam|misleading|debunk\w*|rumou?rs?|myth|not true|no truth|fabricated|doctored)\b"
    r"|ఫేక్|నకిలీ|అబద్ధం|అవాస్తవం|తప్పుడు", re.I)
