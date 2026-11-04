"""Live evidence from the internet, fetched per claim (no API key needed).

1. Wikipedia search (English or Telugu): the intro paragraphs of the best-matching articles,
   split into sentences.
2. Google News RSS: recent news headlines from Indian and international outlets.

The candidates are ranked by BGE-M3 similarity to the claim in claim_verifier, exactly like
local corpus passages. If the machine is offline every function returns [] and the system
falls back to the local corpus. Results are cached per claim for the life of the server.
"""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests

from .common import logger

HEADERS = {"User-Agent": "TruthLensAI-student-project/1.0 (educational use)"}
TIMEOUT = 8
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
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


def _get(url: str, params: dict) -> requests.Response | None:
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Live search failed (%s): %s", url.split("/")[2], exc)
        return None


@lru_cache(maxsize=512)
def _wikipedia(query: str, lang: str, pages: int) -> tuple:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    resp = _get(api, {"action": "query", "list": "search", "srsearch": query, "srlimit": pages,
                      "format": "json", "formatversion": 2})
    if resp is None:
        return ()
    titles = [h["title"] for h in resp.json().get("query", {}).get("search", [])]
    if not titles:
        return ()
    resp = _get(api, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                      "redirects": 1, "titles": "|".join(titles), "format": "json", "formatversion": 2})
    if resp is None:
        return ()
    out = []
    for page in resp.json().get("query", {}).get("pages", []):
        title = page.get("title", "")
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        for sent in SENTENCE_END.split(page.get("extract", ""))[:8]:
            sent = " ".join(sent.split())
            if 30 <= len(sent) <= 600:
                out.append({"text": sent, "source": f"Wikipedia - {title}", "url": url, "type": "wikipedia"})
    return tuple(out)


def search_wikipedia(claim: str, language: str, pages: int = 2) -> list[dict]:
    query = keywords(claim)
    if not query:
        return []
    return [dict(d) for d in _wikipedia(query, "te" if language == "te" else "en", pages)]


@lru_cache(maxsize=512)
def _news(query: str, lang: str, limit: int) -> tuple:
    params = {"q": query, "hl": "te" if lang == "te" else "en-IN", "gl": "IN",
              "ceid": "IN:te" if lang == "te" else "IN:en"}
    resp = _get("https://news.google.com/rss/search", params)
    if resp is None:
        return ()
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
    return [dict(d) for d in _news(query, language, limit)]


DEBUNK_HEADLINE = re.compile(
    r"\b(fake|false|falsely|hoax|scam|misleading|debunk\w*|rumou?rs?|myth|not true|no truth|fabricated|doctored)\b"
    r"|ఫేక్|నకిలీ|అబద్ధం|అవాస్తవం|తప్పుడు", re.I)
