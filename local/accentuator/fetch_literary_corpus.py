# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fetch a public-domain Lithuanian literary corpus from lt.wikisource.

The corpus is intended for register fine-tuning against the chrestomatija
benchmark, so every emitted sentence passes an exact/8-word-shingle firewall
against the local gold set.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import safe_relative, strip_accents  # noqa: E402


API_URL = "https://lt.wikisource.org/w/api.php"
PAGE_URL = "https://lt.wikisource.org/wiki/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36 "
    "accentuation-lt-literary-corpus/0.1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "eval" / "literary-corpus.txt"
DEFAULT_GOLD = SCRIPT_DIR / "data" / "eval" / "chrestomatija-gold.jsonl"
MIN_REQUEST_INTERVAL_SECONDS = 1.2
REQUEST_TIMEOUT_SECONDS = 45.0
TITLE_BATCH_SIZE = 40
WORD_RE = re.compile(r"(?:[^\W\d_][\u0300-\u036f]*)+", re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?…]+[\"'”’)\]]*)\s+(?=[A-ZĄČĘĖĮŠŲŪŽ„“\"'])"
)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REF_RE = re.compile(r"<ref\b[^>/]*(?:/>|>.*?</ref\s*>)", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:-+\s*)?\d{1,4}(?:\s*-+)?\s*$")
FOOTNOTE_TOKEN_RE = re.compile(r"(?<=[^\W\d_])\d{1,2}(?=\s|[,.!?;:)]|$)")
OLD_DIGRAPH_RE = re.compile(r"(?i)(?:sz|cz)")
NON_MODERN_CHAR_RE = re.compile(r"[łŁćĆńŃśŚźŹżŻѣѢ]")
EXCLUDED_AUTHORS = [
    {"name": "Lazdynų Pelėda", "death_year": 1957, "reason": "not yet PD / joint-authorship risk"},
    {"name": "Antanas Vienuolis", "death_year": 1957, "reason": "not yet PD"},
    {"name": "Ignas Šeinius", "death_year": 1959, "reason": "not yet PD"},
]


@dataclass(frozen=True)
class Author:
    name: str
    death_year: int
    aliases: tuple[str, ...] = ()


AUTHOR_WHITELIST: tuple[Author, ...] = (
    Author("Maironis", 1932),
    Author("Žemaitė", 1921),
    Author("Jonas Biliūnas", 1907),
    Author("Šatrijos Ragana", 1930),
    Author("Vincas Kudirka", 1899),
    Author("Antanas Baranauskas", 1902),
    Author("Vincas Krėvė-Mickevičius", 1954, ("Vincas Krėvė",)),
    Author("Jurgis Savickis", 1952),
    Author("Vydūnas", 1953),
    Author("Motiejus Valančius", 1875),
    Author("Balys Sruoga", 1947),
    Author("Salomėja Nėris", 1945),
    Author("Vytautas Mačernis", 1944),
    Author("Julius Janonis", 1917),
    Author("Petras Cvirka", 1947),
)


@dataclass(frozen=True)
class AuthorPage:
    author: Author
    title: str
    url: str
    wikitext: str


@dataclass(frozen=True)
class WorkCandidate:
    author: Author
    title: str
    source_title: str
    genre_hint: str
    heading: str
    order: int


@dataclass
class OrthographyReport:
    token_count: int
    non_modern_chars: int
    o_acute: int
    old_digraphs: int
    non_modern_per_1k: float
    o_acute_per_1k: float
    old_digraphs_per_1k: float
    flagged: bool
    reasons: list[str]


@dataclass
class Work:
    author: Author
    title: str
    url: str
    genre: str
    heading: str
    source_pages: list[str]
    units: list[str]
    tokens: int
    poetry_tokens: int
    orthography: OrthographyReport
    raw_tokens: int
    firewall_dropped: int = 0
    firewall_dropped_exact: int = 0
    firewall_dropped_shingle: int = 0
    output_units: list[str] = field(default_factory=list)
    output_tokens: int = 0


@dataclass
class Firewall:
    gold_sentences: int
    gold_shingles: int
    dropped_exact: int
    dropped_shingle: int
    dropped_by_work: dict[str, int]

    @property
    def dropped_total(self) -> int:
        return self.dropped_exact + self.dropped_shingle


@dataclass
class Corpus:
    works: list[Work]
    tokens: int
    poetry_tokens: int
    firewall: Firewall
    skipped_old_orthography: list[dict[str, Any]]
    skipped_empty: list[dict[str, Any]]
    skipped_over_budget: list[dict[str, Any]]
    discovered_candidates: int


class ThrottledMediaWikiClient:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.5, interval_seconds)
        self.last_start = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def __enter__(self) -> "ThrottledMediaWikiClient":
        self.client.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.__exit__(*exc)

    def query(self, params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
        payload = {
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
            **params,
        }
        for attempt in range(retries):
            now = time.monotonic()
            if self.last_start:
                delay = self.interval_seconds - (now - self.last_start)
                if delay > 0:
                    time.sleep(delay)
            self.last_start = time.monotonic()
            response = self.client.get(API_URL, params=payload)
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                try:
                    wait = float(retry_after) if retry_after else 2.0
                except ValueError:
                    wait = 2.0
                wait = max(wait, self.interval_seconds) * (attempt + 1)
                print(f"rate limited by API; sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                code = data["error"].get("code", "api-error")
                info = data["error"].get("info", data["error"])
                raise RuntimeError(f"MediaWiki API {code}: {info}")
            return data
        raise RuntimeError("MediaWiki API kept returning HTTP 429")


def page_url(title: str) -> str:
    return PAGE_URL + title.replace(" ", "_")


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", html.unescape(text)).replace("\xa0", " ")


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFC", text))


def token_count(text: str) -> int:
    return len(word_tokens(text))


def strip_stress_lower(text: str) -> str:
    stripped = strip_accents(normalize_text(text))
    return re.sub(r"\s+", " ", stripped.casefold()).strip()


def ascii_match_key(text: str) -> str:
    lowered = strip_stress_lower(text)
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", lowered)
        if not unicodedata.combining(ch)
    )


def word_keys(text: str) -> list[str]:
    return [strip_stress_lower(token) for token in word_tokens(text)]


def title_key(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ")).strip().casefold()


def visible_title(raw_link: str) -> str | None:
    title = normalize_text(raw_link).strip()
    if not title:
        return None
    if ":" in title:
        namespace = title.split(":", 1)[0].casefold()
        if namespace not in {"s"}:
            return None
    blocked_titles = {
        "vikipedija",
        "vikicitatose",
        "pagrindinis puslapis",
    }
    if title_key(title) in blocked_titles:
        return None
    return title


def genre_from_heading(heading: str) -> str:
    normalized = ascii_match_key(heading)
    if any(key in normalized for key in ("eileras", "poez", "poem", "giesm")):
        return "poetry"
    if any(key in normalized for key in ("vertim", "verte")):
        return "skip"
    if any(
        key in normalized
        for key in (
            "apsak",
            "apysak",
            "novel",
            "roman",
            "proza",
            "pasak",
            "public",
            "drama",
            "pjes",
            "vaizdel",
        )
    ):
        return "prose"
    return "unknown"


def parse_author_work_links(page: AuthorPage, start_order: int) -> list[WorkCandidate]:
    in_works = False
    heading = ""
    genre = "unknown"
    candidates: list[WorkCandidate] = []
    seen: set[str] = set()
    order = start_order

    for line in page.wikitext.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            text = normalize_text(match.group(2))
            low = ascii_match_key(text)
            if level == 2:
                in_works = any(key in low for key in ("darbai", "kuryba", "kuriniai"))
                heading = text
                genre = "unknown"
                continue
            if in_works:
                heading = text
                genre = genre_from_heading(text)
            continue
        if not in_works or genre == "skip":
            continue
        if not line.lstrip().startswith("*"):
            continue
        for raw_link in WIKILINK_RE.findall(line):
            title = visible_title(raw_link)
            if title is None:
                continue
            key = title_key(title)
            if key in seen or key == title_key(page.title) or key == title_key(page.author.name):
                continue
            seen.add(key)
            candidates.append(
                WorkCandidate(
                    author=page.author,
                    title=title,
                    source_title=page.title,
                    genre_hint=genre,
                    heading=heading,
                    order=order,
                )
            )
            order += 1
    return candidates


def fetch_pages(
    client: ThrottledMediaWikiClient,
    titles: list[str],
) -> dict[str, dict[str, Any]]:
    pages_by_requested: dict[str, dict[str, Any]] = {}
    for batch in chunked(titles, TITLE_BATCH_SIZE):
        data = client.query(
            {
                "action": "query",
                "redirects": "1",
                "prop": "info|revisions",
                "inprop": "url",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(batch),
            }
        )
        normalized = {
            item["from"]: item["to"]
            for item in data.get("query", {}).get("normalized", [])
            if "from" in item and "to" in item
        }
        redirects = {
            item["from"]: item["to"]
            for item in data.get("query", {}).get("redirects", [])
            if "from" in item and "to" in item
        }
        pages = {page.get("title", ""): page for page in data.get("query", {}).get("pages", [])}
        for requested in batch:
            target = redirects.get(normalized.get(requested, requested), normalized.get(requested, requested))
            page = pages.get(target) or pages.get(normalized.get(requested, requested)) or pages.get(requested)
            if page is not None:
                pages_by_requested[requested] = page
    return pages_by_requested


def page_content(page: dict[str, Any]) -> str:
    revisions = page.get("revisions") or []
    if not revisions:
        return ""
    slots = revisions[0].get("slots") or {}
    main = slots.get("main") or {}
    return str(main.get("content") or "")


def resolve_author_pages(client: ThrottledMediaWikiClient) -> list[AuthorPage]:
    requested: list[str] = []
    owner: dict[str, Author] = {}
    for author in AUTHOR_WHITELIST:
        for title in (author.name, *author.aliases, f"Autorius:{author.name}"):
            if title not in owner:
                requested.append(title)
                owner[title] = author
    fetched = fetch_pages(client, requested)

    pages: list[AuthorPage] = []
    for author in AUTHOR_WHITELIST:
        best: tuple[str, dict[str, Any], str, int] | None = None
        for title in (author.name, *author.aliases, f"Autorius:{author.name}"):
            page = fetched.get(title)
            if not page or page.get("missing"):
                continue
            content = page_content(page)
            if not content:
                continue
            score = content.casefold().count("[[") + (100 if "== Darbai ==" in content else 0)
            if best is None or score > best[3]:
                best = (str(page.get("title") or title), page, content, score)
        if best is None:
            print(f"author page missing: {author.name} (d.{author.death_year})")
            continue
        title, page, content, _score = best
        pages.append(
            AuthorPage(
                author=author,
                title=title,
                url=str(page.get("fullurl") or page_url(title)),
                wikitext=content,
            )
        )
    return pages


def remove_templates(text: str) -> str:
    previous = None
    current = text
    for _ in range(8):
        if current == previous:
            break
        previous = current
        current = TEMPLATE_RE.sub(" ", current)
    current = re.sub(r"\{\|.*?\|\}", " ", current, flags=re.DOTALL)
    return current


def strip_wiki_markup(wikitext: str) -> str:
    text = normalize_text(wikitext)
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = re.sub(r"<references\b[^>]*(?:/>|>.*?</references\s*>)", " ", text, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(?:p|div|poem|center|blockquote)>", "\n", text, flags=re.I)
    text = re.sub(r"<(?:poem|poem [^>]*|blockquote|center|div|p)[^>]*>", "\n", text, flags=re.I)
    text = remove_templates(text)
    text = re.sub(r"\[\[(?:Vaizdas|File|Image|Kategorija|Category):[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]*)?\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]*)?\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^=+.*?=+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"__\w+__", " ", text)
    text = FOOTNOTE_TOKEN_RE.sub("", text)
    return unicodedata.normalize("NFC", text)


def is_all_caps_line(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    cased = [ch for ch in letters if ch.lower() != ch.upper()]
    return bool(cased) and all(not ch.islower() for ch in cased)


def looks_editorial(line: str) -> bool:
    lowered = ascii_match_key(line)
    blocked_prefixes = (
        "saltinis",
        "sis tekstas",
        "redagavo",
        "isleido",
        "leidykla",
        "turinys",
        "pastabos",
        "nuorodos",
        "puslapis",
        "viki",
    )
    if lowered.startswith(blocked_prefixes):
        return True
    if re.search(r"\b(?:isbn|pdf|djvu)\b", lowered):
        return True
    if re.search(r"\b(?:19|18|20)\d{2}\s*m\.", lowered) and token_count(line) < 8:
        return True
    return False


def clean_lines(wikitext: str) -> list[str]:
    raw = strip_wiki_markup(wikitext)
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = re.sub(r"\s+", " ", raw_line.replace("\u200b", "")).strip()
        line = line.strip("[]{}|")
        if not line:
            continue
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if line.startswith(("*", "#", "|", "!", ";", ":")):
            line = line.lstrip("*#|!;: ").strip()
        if not line or PAGE_NUMBER_RE.fullmatch(line):
            continue
        if is_all_caps_line(line):
            continue
        if looks_editorial(line):
            continue
        if token_count(line) < 3:
            continue
        lines.append(unicodedata.normalize("NFC", line))
    return lines


def split_prose_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end(1)
        piece = text[start:end].strip()
        if piece and token_count(piece) >= 3:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail and token_count(tail) >= 3:
        sentences.append(tail)
    return sentences


def infer_genre(genre_hint: str, lines: list[str]) -> str:
    if genre_hint in {"poetry", "prose"}:
        return genre_hint
    if not lines:
        return genre_hint
    shortish = sum(1 for line in lines if 3 <= token_count(line) <= 8)
    punctuated = sum(1 for line in lines if re.search(r"[.!?…][\"'”’)\]]*$", line))
    if len(lines) >= 5 and shortish / len(lines) >= 0.55 and punctuated / len(lines) <= 0.75:
        return "poetry"
    return "prose"


def units_for_work(genre: str, lines: list[str]) -> list[str]:
    if genre == "poetry":
        return [line for line in lines if token_count(line) >= 3]
    paragraph = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return split_prose_sentences(paragraph)


def orthography_report(text: str) -> OrthographyReport:
    tokens = max(1, token_count(text))
    non_modern = len(NON_MODERN_CHAR_RE.findall(text))
    o_acute = len(re.findall(r"[óÓ]", text))
    old_digraphs = len(OLD_DIGRAPH_RE.findall(text))
    non_modern_per_1k = non_modern * 1000 / tokens
    o_acute_per_1k = o_acute * 1000 / tokens
    old_digraphs_per_1k = old_digraphs * 1000 / tokens
    reasons: list[str] = []
    if non_modern_per_1k >= 0.2 or non_modern >= 3:
        reasons.append(f"non-modern chars {non_modern_per_1k:.2f}/1k")
    if o_acute_per_1k >= 0.8 or o_acute >= 8:
        reasons.append(f"ó {o_acute_per_1k:.2f}/1k")
    if old_digraphs_per_1k >= 1.2 or old_digraphs >= 12:
        reasons.append(f"sz/cz {old_digraphs_per_1k:.2f}/1k")
    return OrthographyReport(
        token_count=tokens,
        non_modern_chars=non_modern,
        o_acute=o_acute,
        old_digraphs=old_digraphs,
        non_modern_per_1k=non_modern_per_1k,
        o_acute_per_1k=o_acute_per_1k,
        old_digraphs_per_1k=old_digraphs_per_1k,
        flagged=bool(reasons),
        reasons=reasons,
    )


def content_links(wikitext: str, parent_title: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    parent_key = title_key(parent_title)
    for line in wikitext.splitlines():
        if not line.lstrip().startswith(("*", "#", "[[")):
            continue
        for raw_link in WIKILINK_RE.findall(line):
            title = visible_title(raw_link)
            if title is None:
                continue
            key = title_key(title)
            if key == parent_key or key in seen:
                continue
            if key.startswith(parent_key + "/") or key.startswith(parent_key + "."):
                seen.add(key)
                links.append(title)
    return links


def expand_container_pages(
    client: ThrottledMediaWikiClient,
    fetched: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    extra_titles: list[str] = []
    seen = {title_key(title) for title in fetched}
    for requested, page in list(fetched.items()):
        content = page_content(page)
        lines = clean_lines(content)
        links = content_links(content, str(page.get("title") or requested))
        if token_count(" ".join(lines)) <= 120 and links:
            for link in links:
                key = title_key(link)
                if key not in seen:
                    seen.add(key)
                    extra_titles.append(link)
    if not extra_titles:
        return fetched
    print(f"expanding {len(extra_titles):,} linked subpages from short work containers")
    fetched.update(fetch_pages(client, extra_titles))
    return fetched


def build_works(
    candidates: list[WorkCandidate],
    pages: dict[str, dict[str, Any]],
) -> tuple[list[Work], list[dict[str, Any]], list[dict[str, Any]]]:
    works: list[Work] = []
    skipped_old: list[dict[str, Any]] = []
    skipped_empty: list[dict[str, Any]] = []

    for candidate in sorted(candidates, key=lambda item: item.order):
        page = pages.get(candidate.title)
        if not page or page.get("missing"):
            skipped_empty.append(
                {
                    "author": candidate.author.name,
                    "title": candidate.title,
                    "reason": "missing page",
                }
            )
            continue
        title = str(page.get("title") or candidate.title)
        source_titles = [title]
        contents = [page_content(page)]
        lines = clean_lines(contents[0])

        if token_count(" ".join(lines)) <= 120:
            for link in content_links(contents[0], title):
                linked = pages.get(link)
                if not linked or linked.get("missing"):
                    continue
                linked_text = page_content(linked)
                linked_lines = clean_lines(linked_text)
                if linked_lines:
                    source_titles.append(str(linked.get("title") or link))
                    contents.append(linked_text)
                    lines.extend(linked_lines)

        if not lines:
            skipped_empty.append(
                {
                    "author": candidate.author.name,
                    "title": candidate.title,
                    "reason": "no clean lines",
                }
            )
            continue

        text_for_flags = "\n".join(lines)
        ortho = orthography_report(text_for_flags)
        if ortho.flagged:
            skipped_old.append(
                {
                    "author": candidate.author.name,
                    "title": title,
                    "url": page_url(title),
                    "tokens": ortho.token_count,
                    "reasons": ortho.reasons,
                    "non_modern_per_1k": round(ortho.non_modern_per_1k, 3),
                    "o_acute_per_1k": round(ortho.o_acute_per_1k, 3),
                    "old_digraphs_per_1k": round(ortho.old_digraphs_per_1k, 3),
                }
            )
            continue

        genre = infer_genre(candidate.genre_hint, lines)
        units = units_for_work(genre, lines)
        units = [re.sub(r"\s+", " ", unit).strip() for unit in units if token_count(unit) >= 3]
        tokens = sum(token_count(unit) for unit in units)
        if tokens <= 0:
            skipped_empty.append(
                {
                    "author": candidate.author.name,
                    "title": title,
                    "reason": "no sentence/verse units",
                }
            )
            continue
        poetry_tokens = tokens if genre == "poetry" else 0
        works.append(
            Work(
                author=candidate.author,
                title=title,
                url=str(page.get("fullurl") or page_url(title)),
                genre=genre,
                heading=candidate.heading,
                source_pages=source_titles,
                units=units,
                tokens=tokens,
                poetry_tokens=poetry_tokens,
                orthography=ortho,
                raw_tokens=token_count(text_for_flags),
            )
        )
    return works, skipped_old, skipped_empty


def load_gold_firewall(path: Path) -> tuple[set[str], set[tuple[str, ...]], int]:
    if not path.exists():
        raise FileNotFoundError(f"missing gold JSONL: {path}")
    exact: set[str] = set()
    shingles: set[tuple[str, ...]] = set()
    sentence_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            normalized = strip_stress_lower(text)
            if not normalized:
                raise ValueError(f"empty gold text at {path}:{line_number}")
            sentence_count += 1
            exact.add(normalized)
            words = word_keys(normalized)
            for index in range(0, max(0, len(words) - 7)):
                shingles.add(tuple(words[index : index + 8]))
    return exact, shingles, sentence_count


def sentence_has_gold_shingle(sentence: str, shingles: set[tuple[str, ...]]) -> bool:
    words = word_keys(sentence)
    if len(words) < 8:
        return False
    return any(tuple(words[index : index + 8]) in shingles for index in range(0, len(words) - 7))


def apply_firewall(works: list[Work], gold_path: Path) -> Firewall:
    exact, shingles, gold_count = load_gold_firewall(gold_path)
    dropped_exact = 0
    dropped_shingle = 0
    dropped_by_work: dict[str, int] = {}

    for work in works:
        kept: list[str] = []
        for unit in work.units:
            normalized = strip_stress_lower(unit)
            drop_reason = ""
            if normalized in exact:
                drop_reason = "exact"
                dropped_exact += 1
                work.firewall_dropped_exact += 1
            elif sentence_has_gold_shingle(unit, shingles):
                drop_reason = "shingle"
                dropped_shingle += 1
                work.firewall_dropped_shingle += 1
            if drop_reason:
                work.firewall_dropped += 1
                dropped_by_work[work.title] = dropped_by_work.get(work.title, 0) + 1
                continue
            kept.append(unit)
        work.output_units = kept
        work.output_tokens = sum(token_count(unit) for unit in kept)
    return Firewall(
        gold_sentences=gold_count,
        gold_shingles=len(shingles),
        dropped_exact=dropped_exact,
        dropped_shingle=dropped_shingle,
        dropped_by_work=dict(sorted(dropped_by_work.items())),
    )


def selected_firewall_report(works: list[Work], base: Firewall) -> Firewall:
    dropped_by_work = {
        work.title: work.firewall_dropped
        for work in works
        if work.firewall_dropped > 0
    }
    return Firewall(
        gold_sentences=base.gold_sentences,
        gold_shingles=base.gold_shingles,
        dropped_exact=sum(work.firewall_dropped_exact for work in works),
        dropped_shingle=sum(work.firewall_dropped_shingle for work in works),
        dropped_by_work=dict(sorted(dropped_by_work.items())),
    )


def selection_score(work: Work, current_poetry_share: float, max_tokens: int) -> tuple[int, int, int]:
    if current_poetry_share < 0.25:
        preferred = 0 if work.genre == "poetry" else 1
    elif current_poetry_share > 0.60:
        preferred = 0 if work.genre != "poetry" else 1
    else:
        preferred = 0
    size_penalty = 0 if work.output_tokens <= max_tokens else 1
    genre_penalty = 0 if work.genre in {"poetry", "prose"} else 1
    return preferred, size_penalty, genre_penalty


def select_works(works: list[Work], max_tokens: int) -> tuple[list[Work], list[dict[str, Any]]]:
    selected: list[Work] = []
    skipped_over_budget: list[dict[str, Any]] = []
    used: set[int] = set()
    total = 0
    poetry = 0

    available = [work for work in works if work.output_tokens > 0]
    while True:
        remaining = [
            (index, work)
            for index, work in enumerate(available)
            if index not in used and total + work.output_tokens <= max_tokens
        ]
        if not remaining:
            break
        share = poetry / total if total else 0.0
        remaining.sort(
            key=lambda item: (
                selection_score(item[1], share, max_tokens),
                item[0],
            )
        )
        index, work = remaining[0]
        used.add(index)
        selected.append(work)
        total += work.output_tokens
        poetry += work.output_tokens if work.genre == "poetry" else 0
        if total >= max_tokens * 0.985:
            break

    for index, work in enumerate(available):
        if index in used:
            continue
        if total + work.output_tokens > max_tokens:
            skipped_over_budget.append(
                {
                    "author": work.author.name,
                    "title": work.title,
                    "tokens": work.output_tokens,
                    "reason": "would exceed max token budget",
                }
            )
    return selected, skipped_over_budget


def build_corpus(client: ThrottledMediaWikiClient, max_tokens: int, gold_path: Path) -> Corpus:
    author_pages = resolve_author_pages(client)
    print("author pages:")
    for page in author_pages:
        print(f"  {page.author.name} (d.{page.author.death_year}) -> {page.title} [{page.url}]")

    candidates: list[WorkCandidate] = []
    order = 0
    for page in author_pages:
        page_candidates = parse_author_work_links(page, order)
        order += len(page_candidates)
        candidates.extend(page_candidates)
    deduped: list[WorkCandidate] = []
    seen_keys: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.author.name, title_key(candidate.title))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    candidates = deduped
    print(f"discovered work links: {len(candidates):,}")

    pages = fetch_pages(client, [candidate.title for candidate in candidates])
    pages = expand_container_pages(client, pages)
    works, skipped_old, skipped_empty = build_works(candidates, pages)
    print(f"clean candidate works: {len(works):,}")

    all_firewall = apply_firewall(works, gold_path)
    selected, skipped_over_budget = select_works(works, max_tokens)
    firewall = selected_firewall_report(selected, all_firewall)
    tokens = sum(work.output_tokens for work in selected)
    poetry_tokens = sum(work.output_tokens for work in selected if work.genre == "poetry")
    return Corpus(
        works=selected,
        tokens=tokens,
        poetry_tokens=poetry_tokens,
        firewall=firewall,
        skipped_old_orthography=skipped_old,
        skipped_empty=skipped_empty,
        skipped_over_budget=skipped_over_budget,
        discovered_candidates=len(candidates),
    )


def write_outputs(corpus: Corpus, output: Path, max_tokens: int, request_interval: float) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, work in enumerate(corpus.works):
            for unit in work.output_units:
                handle.write(unit + "\n")
            if index + 1 < len(corpus.works):
                handle.write("\n")

    meta_path = output.with_suffix(".meta.json")
    meta = {
        "source": API_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": USER_AGENT,
        "request_interval_seconds": request_interval,
        "max_tokens": max_tokens,
        "tokens": corpus.tokens,
        "sentences_or_verse_lines": sum(len(work.output_units) for work in corpus.works),
        "poetry_tokens": corpus.poetry_tokens,
        "poetry_share": corpus.poetry_tokens / corpus.tokens if corpus.tokens else 0.0,
        "author_whitelist": [
            {
                "name": author.name,
                "death_year": author.death_year,
                "public_domain_check": "death year is at least 70 years before 2026",
            }
            for author in AUTHOR_WHITELIST
        ],
        "excluded_authors": EXCLUDED_AUTHORS,
        "segments": [
            {
                "author": work.author.name,
                "death_year": work.author.death_year,
                "title": work.title,
                "url": work.url,
                "genre": work.genre,
                "heading": work.heading,
                "source_pages": work.source_pages,
                "tokens": work.output_tokens,
                "sentences_or_verse_lines": len(work.output_units),
                "firewall_dropped": work.firewall_dropped,
                "orthography": {
                    "non_modern_chars": work.orthography.non_modern_chars,
                    "o_acute": work.orthography.o_acute,
                    "old_digraphs": work.orthography.old_digraphs,
                    "non_modern_per_1k": round(work.orthography.non_modern_per_1k, 3),
                    "o_acute_per_1k": round(work.orthography.o_acute_per_1k, 3),
                    "old_digraphs_per_1k": round(work.orthography.old_digraphs_per_1k, 3),
                    "flagged": work.orthography.flagged,
                    "reasons": work.orthography.reasons,
                },
            }
            for work in corpus.works
        ],
        "firewall": {
            "gold_sentences": corpus.firewall.gold_sentences,
            "gold_8_word_shingles": corpus.firewall.gold_shingles,
            "dropped_exact": corpus.firewall.dropped_exact,
            "dropped_shingle": corpus.firewall.dropped_shingle,
            "dropped_total": corpus.firewall.dropped_total,
            "dropped_by_work": corpus.firewall.dropped_by_work,
        },
        "skipped_old_orthography": corpus.skipped_old_orthography,
        "skipped_empty_or_missing": corpus.skipped_empty,
        "skipped_over_budget": corpus.skipped_over_budget,
        "discovered_candidates": corpus.discovered_candidates,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta_path


def print_work_list(corpus: Corpus) -> None:
    print("per-work list:")
    for work in corpus.works:
        print(
            f"  {work.author.name} (d.{work.author.death_year}) | {work.genre:7} | "
            f"{work.output_tokens:6,} tokens | {work.url} | {work.title}"
        )
        print(
            "    orthography: "
            f"nonmodern={work.orthography.non_modern_per_1k:.2f}/1k, "
            f"ó={work.orthography.o_acute_per_1k:.2f}/1k, "
            f"sz/cz={work.orthography.old_digraphs_per_1k:.2f}/1k"
        )
    if corpus.skipped_old_orthography:
        print("old-orthography skipped works:")
        for item in corpus.skipped_old_orthography:
            print(
                f"  {item['author']} | {item['tokens']:6,} tokens | {item['url']} | "
                f"{item['title']} ({'; '.join(item['reasons'])})"
            )
    else:
        print("old-orthography skipped works: none")


def print_firewall_report(firewall: Firewall) -> None:
    print("firewall report:")
    print(f"  gold sentences: {firewall.gold_sentences:,}")
    print(f"  gold 8-word shingles: {firewall.gold_shingles:,}")
    print(f"  dropped exact: {firewall.dropped_exact:,}")
    print(f"  dropped shingle: {firewall.dropped_shingle:,}")
    print(f"  dropped total: {firewall.dropped_total:,}")
    if firewall.dropped_by_work:
        print("  dropped by work:")
        for title, count in firewall.dropped_by_work.items():
            print(f"    {title}: {count:,}")
    else:
        print("  dropped by work: none")


def print_samples(corpus: Corpus, sample_count: int, seed: int) -> None:
    rows: list[tuple[str, str]] = []
    for work in corpus.works:
        rows.extend((work.title, unit) for unit in work.output_units)
    samples = random.Random(seed).sample(rows, k=min(sample_count, len(rows)))
    print(f"{len(samples)} random sample sentences:")
    for index, (title, unit) in enumerate(samples, start=1):
        print(f"{index}. [{title}] {unit}")


def print_summary(corpus: Corpus) -> None:
    poetry_share = corpus.poetry_tokens / corpus.tokens if corpus.tokens else 0.0
    print(f"works: {len(corpus.works):,}")
    print(f"tokens: {corpus.tokens:,}")
    print(f"poetry tokens: {corpus.poetry_tokens:,}")
    print(f"poetry share: {poetry_share:.1%}")
    print(f"sentence/verse-line units: {sum(len(work.output_units) for work in corpus.works):,}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=220000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=MIN_REQUEST_INTERVAL_SECONDS,
        help="Minimum seconds between MediaWiki API requests; clamped to >=0.5.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.request_interval < 0.5:
        parser.error("--request-interval must be at least 0.5 seconds")

    with ThrottledMediaWikiClient(args.request_interval) as client:
        corpus = build_corpus(client, max_tokens=args.max_tokens, gold_path=args.gold)

    meta_path = write_outputs(
        corpus=corpus,
        output=args.output,
        max_tokens=args.max_tokens,
        request_interval=args.request_interval,
    )
    print_work_list(corpus)
    print_firewall_report(corpus.firewall)
    print_summary(corpus)
    print_samples(corpus, sample_count=args.sample_count, seed=args.seed)
    print(f"wrote: {safe_relative(args.output)}")
    print(f"meta:  {safe_relative(meta_path)}")

    poetry_share = corpus.poetry_tokens / corpus.tokens if corpus.tokens else 0.0
    if args.max_tokens >= 190000 and not (190000 <= corpus.tokens <= 230000):
        print("full-run token count is outside 190k-230k", file=sys.stderr)
        return 1
    if args.max_tokens >= 190000 and not (0.25 <= poetry_share <= 0.60):
        print("full-run poetry share is outside 25%-60%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
