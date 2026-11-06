"""Live evidence from the internet, fetched per claim (no API key needed).

1. Wikipedia search (English or Telugu): the intro paragraphs of the best-matching articles,
   split into sentences.
2. Google News RSS: recent news headlines from Indian and international outlets.
3. DuckDuckGo web search (the ddgs package): result snippets from any website.

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
WEB_SLOTS = threading.Semaphore(2)  # search engines block bursts of parallel requests
# Search engines tried by the ddgs package. Yahoo answered best in testing; DuckDuckGo rate-limits
# after heavy use and Brave/Startpage sometimes hang, so they are left out.
WEB_ENGINES = "yahoo,duckduckgo,mojeek"


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


def _page_items(page: dict, lang: str, max_sentences: int) -> list[dict]:
    title = page.get("title", "")
    url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    sentences = [clean_sentence(x) for x in split_sentences(page.get("extract", ""))[:max_sentences]]
    sentences = [x for x in sentences if 30 <= len(x) <= 600]
    lead = sentences[0] if sentences else ""
    out = []
    for i, sent in enumerate(sentences):
        # NLI reads each sentence after the article's first sentence, so "he" or "the party"
        # in a later sentence has a name to refer to.
        context = sent if i == 0 else f"{lead} {sent}"
        out.append({"text": sent, "nli_text": context, "source": f"Wikipedia - {title}", "url": url,
                    "type": "wikipedia"})
    return out


@lru_cache(maxsize=1024)
def _wikipedia(query: str, lang: str, pages: int, max_sentences: int = 25) -> tuple:
    """One request: search for the query and return the intro of the best-matching pages.
    Returns (items, English titles of the same pages) - the second part is used for Telugu."""
    params = {"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": pages,
              "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max", "redirects": 1,
              "format": "json", "formatversion": 2}
    if lang != "en":
        params.update(prop="extracts|langlinks", lllang="en", lllimit="max")
    with WIKI_SLOTS:
        resp = _get(f"https://{lang}.wikipedia.org/w/api.php", params)
    out, english = [], []
    for page in sorted(resp.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 0)):
        out += _page_items(page, lang, max_sentences)
        english += [ll["title"] for ll in page.get("langlinks", []) if ll.get("lang") == "en"]
    return tuple(out), tuple(english)


@lru_cache(maxsize=1024)
def _wikipedia_titles(titles: tuple, lang: str, max_sentences: int = 25) -> tuple:
    if not titles:
        return ()
    with WIKI_SLOTS:
        resp = _get(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "titles": "|".join(titles), "prop": "extracts", "exintro": 1,
            "explaintext": 1, "exlimit": "max", "redirects": 1, "format": "json", "formatversion": 2})
    out = []
    for page in resp.json().get("query", {}).get("pages", []):
        out += _page_items(page, lang, max_sentences)
    return tuple(out)


SKIP_ENTITY = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}


def topics(claim: str, limit: int = 3) -> list[str]:
    """Names and topics in an English claim, each searched on its own: in 'Rahul Gandhi became the Prime
    Minister of India' the answer is on the 'Prime Minister of India' page, not on Rahul Gandhi's."""
    from .claim_extractor import _nlp

    nlp = _nlp.get()
    if nlp is None:
        return []
    doc = nlp(claim)
    found = [e.text for e in doc.ents if e.label_ not in SKIP_ENTITY]
    chunks = list(doc.noun_chunks)
    for i, ch in enumerate(chunks):  # "the Prime Minister" + "of" + "India" -> "Prime Minister of India"
        words = [t for t in ch if t.pos_ not in ("DET", "PRON")]
        if not words:
            continue
        phrase = " ".join(t.text for t in words)
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        if nxt is not None and ch.end < len(doc) and doc[ch.end].lower_ == "of" and nxt.start == ch.end + 1:
            phrase += " of " + " ".join(t.text for t in nxt if t.pos_ != "DET")
        if any(t.pos_ in ("PROPN", "NOUN") for t in words) and (len(phrase.split()) > 1 or words[0].pos_ == "PROPN"):
            found.append(phrase)
    out, seen = [], set()
    for f in sorted(found, key=lambda x: -len(x.split())):  # longer, more specific topics first
        key = f.lower()
        if key not in seen and not any(key in s for s in seen):
            seen.add(key)
            out.append(f)
    return out[:limit]


PARENTHESES = re.compile(r"\s*\([^()]*\)")
CITATION = re.compile(r"\[(?:\d+|[a-z]|citation needed)\]")  # Wikipedia's [1] [a] [citation needed]


def clean_sentence(text: str) -> str:
    """Removes pronunciation guides and other brackets, e.g. 'Naidu (Telugu pronunciation: [...]; born 1950)'."""
    text = CITATION.sub("", " ".join(text.split()))
    for _ in range(2):  # nested brackets
        text = PARENTHESES.sub("", text)
    return text.replace(" ,", ",").strip()


def search_wikipedia(claim: str, language: str, pages: int = 2) -> list[dict]:
    """Keyword search, plus (English) one search per name/topic in the claim,
    plus (Telugu) the English version of the Telugu pages found."""
    query = keywords(claim)
    if not query:
        return []
    lang = "te" if language == "te" else "en"
    items = []
    try:
        found, english = _wikipedia(query, lang, pages)
        items += found
        if lang == "te" and english:
            items += _wikipedia_titles(tuple(english[:pages]), "en")
    except SearchFailed:
        pass
    if lang == "en":
        for topic in topics(claim):
            try:
                items += _wikipedia(topic, "en", 1, 12)[0]
            except SearchFailed:
                continue
    else:
        # Telugu: a whole-sentence search often misses, so the main words are also searched one by one
        # ("చార్మినార్" finds the Charminar page), each with its English version.
        words = [w for w in query.split() if len(w) >= 4]
        candidates = ([" ".join(words[:2])] if len(words) >= 2 else []) + words[:2]  # names are often two words
        for word in candidates:
            try:
                found, english = _wikipedia(word, "te", 1, 12)
                items += found
                if english:
                    items += _wikipedia_titles(tuple(english[:1]), "en", 12)
            except SearchFailed:
                continue
    seen, out = set(), []
    for d in items:
        key = (d["source"], d["text"])
        if key not in seen:
            seen.add(key)
            out.append(dict(d))
    return out


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


SNIPPET_DATE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s*[·-]\s*")  # "May 17, 2019 · ..."


@lru_cache(maxsize=1024)
def _web(query: str, region: str, limit: int) -> tuple:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("Web search needs the ddgs package (pip install ddgs).")
        return ()
    try:
        with WEB_SLOTS:
            results = DDGS(timeout=TIMEOUT).text(query, region=region, max_results=limit, backend=WEB_ENGINES)
    except Exception as exc:  # ddgs raises its own exception types for rate limits and timeouts
        (logger.debug if "No results" in str(exc) else logger.warning)("Web search failed: %s", exc)
        raise SearchFailed(query) from exc
    out = []
    for r in results or []:
        url = r.get("href", "")
        site = url.split("/")[2].removeprefix("www.") if url.count("/") >= 2 else "web"
        body = SNIPPET_DATE.sub("", (r.get("body") or "").replace("...", ". ").strip())
        for sent in split_sentences(body):
            sent = clean_sentence(sent).strip(" .·")
            if 30 <= len(sent) <= 400:
                out.append({"text": sent + ".", "source": f"Web - {site}", "url": url, "type": "web"})
    return tuple(out)


def search_web(claim: str, language: str, limit: int = 6) -> list[dict]:
    """Snippets from a DuckDuckGo search for the claim's keywords (any website, no key)."""
    query = keywords(claim, limit=10)
    if not query:
        return []
    try:
        return [dict(d) for d in _web(query, "in-en" if language == "en" else "in-te", limit)]
    except SearchFailed:
        return []
