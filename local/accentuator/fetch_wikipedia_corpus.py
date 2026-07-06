# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fetch a balanced modern Lithuanian Wikipedia corpus via MediaWiki API.

The emitted corpus is guarded against both local validation sets: the
chrestomatija gold sentences and the LRT corpus. Sentences matching either set
exactly or by 8-word shingle are dropped before article-level selection.
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


API_URL = "https://lt.wikipedia.org/w/api.php"
PAGE_URL = "https://lt.wikipedia.org/wiki/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36 "
    "accentuation-lt-wikipedia-corpus/0.1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "eval" / "wikipedia-corpus.txt"
DEFAULT_GOLD = SCRIPT_DIR / "data" / "eval" / "chrestomatija-gold.jsonl"
DEFAULT_LRT = SCRIPT_DIR / "data" / "eval" / "lrt-corpus.txt"
MIN_REQUEST_INTERVAL_SECONDS = 1.2
REQUEST_TIMEOUT_SECONDS = 45.0
TITLE_BATCH_SIZE = 40
MIN_ARTICLE_TOKENS = 40
ARTICLE_POOL_MULTIPLIER = 1.25
EXTRACT_INTRO_ONLY = True
WORD_RE = re.compile(r"(?:[^\W\d_][\u0300-\u036f]*)+", re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?…]+[\"'”’)\]]*)\s+(?=[A-ZĄČĘĖĮŠŲŪŽ„“\"'])"
)
ABBREVIATION_RE = re.compile(
    r"\b(?:angl|vok|rus|pranc|lot|gr|lenk|isp|it|pvz|plg|kt|vad|"
    r"pav|prof|dr|šv|m|mėn|d)\.",
    re.I,
)
REF_MARKER_RE = re.compile(r"\[\s*(?:\d+|reikalingas šaltinis|šaltinis\?)\s*\]", re.I)
LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[*#;:•·-]|\d{1,4}\s*[\).:,-]|[IVXLCDM]{1,8}\s*[\).:-])",
    re.I,
)
COORDINATE_RE = re.compile(r"^\s*(?:koordinatės|coordinates)\s*:", re.I)
URL_RE = re.compile(r"https?://\S+")
MONTH_FRAGMENT_RE = re.compile(
    r"^\s*(?:\d{1,4}\s*)?(?:m\.\s*)?"
    r"(?:(?:sausio|vasario|kovo|balandžio|gegužės|birželio|liepos|rugpjūčio|"
    r"rugsėjo|spalio|lapkričio|gruodžio)\s+)?(?:mėn\.|d\.)\s*$",
    re.I,
)

COMMON_SECTION_HEADERS = {
    "aprasymas",
    "apdovanojimai",
    "asmenybes",
    "biografija",
    "budingi bruozai",
    "demografija",
    "ekonomika",
    "etimologija",
    "filmai",
    "geografija",
    "gyvenimas",
    "istorija",
    "isnasos",
    "isvaizda",
    "karjera",
    "klasifikacija",
    "klimatas",
    "kultura",
    "laimejimai",
    "literatura",
    "mityba",
    "mokslas",
    "nuorodos",
    "paplitimas",
    "pastabos",
    "pavadinimas",
    "politika",
    "saltiniai",
    "santykiai",
    "sportas",
    "sudetis",
    "taip pat skaitykite",
    "transportas",
    "valdymas",
    "veikla",
    "ziureti taip pat",
}


@dataclass(frozen=True)
class TopicBucket:
    name: str
    categories: tuple[str, ...]


TOPIC_BUCKETS: tuple[TopicBucket, ...] = (
    TopicBucket(
        "science",
        (
            "Kategorija:Mokslas",
            "Kategorija:Fizika",
            "Kategorija:Chemija",
            "Kategorija:Astronomija",
        ),
    ),
    TopicBucket(
        "history",
        (
            "Kategorija:Istorija",
            "Kategorija:Lietuvos istorija",
            "Kategorija:Istorinės valstybės",
            "Kategorija:Karai",
            "Kategorija:Antrasis pasaulinis karas",
            "Kategorija:Archeologija",
            "Kategorija:Istorikai",
        ),
    ),
    TopicBucket(
        "geography",
        (
            "Kategorija:Geografija",
            "Kategorija:Valstybės",
            "Kategorija:Europos valstybės",
            "Kategorija:Lietuvos miestai",
            "Kategorija:Miestai",
            "Kategorija:Upės",
            "Kategorija:Kalnai",
        ),
    ),
    TopicBucket(
        "sports",
        (
            "Kategorija:Sportas",
            "Kategorija:Sporto šakos",
            "Kategorija:Sportininkai",
            "Kategorija:Lietuvos sportininkai",
            "Kategorija:Krepšinis",
            "Kategorija:Krepšininkai",
            "Kategorija:Lietuvos krepšininkai",
            "Kategorija:Futbolas",
            "Kategorija:Futbolininkai",
            "Kategorija:Olimpinės žaidynės",
        ),
    ),
    TopicBucket(
        "culture",
        (
            "Kategorija:Kultūra",
            "Kategorija:Menas",
            "Kategorija:Literatūra",
            "Kategorija:Muzika",
        ),
    ),
    TopicBucket(
        "technology",
        (
            "Kategorija:Technologijos",
            "Kategorija:Technika",
            "Kategorija:Informacinės technologijos",
            "Kategorija:Kompiuterija",
            "Kategorija:Programinė įranga",
        ),
    ),
    TopicBucket(
        "politics",
        (
            "Kategorija:Politika",
            "Kategorija:Politikai",
            "Kategorija:Politinės partijos",
            "Kategorija:Tarptautiniai santykiai",
        ),
    ),
    TopicBucket(
        "biology",
        (
            "Kategorija:Biologija",
            "Kategorija:Gyvūnai",
            "Kategorija:Augalai",
            "Kategorija:Ekologija",
        ),
    ),
)


@dataclass
class FirewallIndex:
    name: str
    path: Path
    source_units: int
    shingles: set[tuple[str, ...]]
    exact: set[str]


@dataclass
class FirewallDropReport:
    source_units: int
    source_8_word_shingles: int
    dropped_exact: int = 0
    dropped_shingle: int = 0
    dropped_by_article: dict[str, int] = field(default_factory=dict)

    @property
    def dropped_total(self) -> int:
        return self.dropped_exact + self.dropped_shingle


@dataclass
class Article:
    title: str
    bucket: str
    url: str
    sentences: list[str]
    tokens: int
    raw_tokens: int
    source_line_count: int
    firewall_exact: dict[str, int]
    firewall_shingle: dict[str, int]


@dataclass
class Corpus:
    articles: list[Article]
    tokens: int
    firewall: dict[str, FirewallDropReport]
    skipped: list[dict[str, Any]]
    discovered_by_bucket: dict[str, int]
    candidate_titles_by_bucket: dict[str, int]


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
    return re.sub(r"\s+", " ", normalize_text(title).replace("_", " ")).strip().casefold()


def keep_article_title(title: str) -> bool:
    lowered = title_key(title)
    blocked_fragments = (
        "(reikšmės)",
        "(reiksmes)",
        "sąrašas",
        "sarasas",
        "chronologija",
        "bibliografija",
        "diskografija",
        "filmografija",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if lowered.startswith(("sąrašas", "sarasas", "vikipedija:")):
        return False
    return ":" not in title


def category_members(
    client: ThrottledMediaWikiClient,
    category: str,
    member_limit: int,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    cmcontinue: str | None = None
    while len(members) < member_limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": "0|14",
            "cmtype": "page|subcat",
            "cmlimit": "max",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = client.query(params)
        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return members[:member_limit]


def collect_bucket_titles(
    client: ThrottledMediaWikiClient,
    bucket: TopicBucket,
    max_titles: int,
    category_depth: int,
    max_categories: int,
) -> list[str]:
    titles: list[str] = []
    seen_titles: set[str] = set()
    seen_categories: set[str] = set()
    queue: list[tuple[str, int]] = [(category, 0) for category in bucket.categories]

    while queue and len(titles) < max_titles and len(seen_categories) < max_categories:
        category, depth = queue.pop(0)
        category_norm = title_key(category)
        if category_norm in seen_categories:
            continue
        seen_categories.add(category_norm)
        for member in category_members(client, category, member_limit=500):
            title = str(member.get("title") or "")
            ns = int(member.get("ns") or 0)
            if ns == 0:
                key = title_key(title)
                if key not in seen_titles and keep_article_title(title):
                    seen_titles.add(key)
                    titles.append(title)
                    if len(titles) >= max_titles:
                        break
            elif ns == 14 and depth < category_depth:
                subcategory_key = title_key(title)
                if subcategory_key not in seen_categories:
                    queue.append((title, depth + 1))
    return titles


def fetch_extract_pages(
    client: ThrottledMediaWikiClient,
    titles: list[str],
) -> dict[str, dict[str, Any]]:
    pages_by_requested: dict[str, dict[str, Any]] = {}
    for batch in chunked(titles, TITLE_BATCH_SIZE):
        extract_params = {
            "explaintext": "1",
            "exsectionformat": "plain",
            "exlimit": "max",
        }
        if EXTRACT_INTRO_ONLY:
            extract_params["exintro"] = "1"
        data = client.query(
            {
                "action": "query",
                "redirects": "1",
                "prop": "info|extracts",
                "inprop": "url",
                **extract_params,
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
            normalized_title = normalized.get(requested, requested)
            target = redirects.get(normalized_title, normalized_title)
            page = pages.get(target) or pages.get(normalized_title) or pages.get(requested)
            if page is not None:
                pages_by_requested[requested] = page
    return pages_by_requested


def strip_pronunciation_parentheticals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        low = ascii_match_key(inner)
        markers = ("tarimas", "ipa", "klausyti", "listen", "garsas", "pronunciation")
        if not inner.strip():
            return " "
        if any(marker in low for marker in markers):
            return " "
        if re.search(r"[ˈˌɐ-ʯəƏʔθðŋɲʃʒɔɑɡɣχøœæ]", inner):
            return " "
        if re.search(r"/[^/]{1,80}/", inner):
            return " "
        return match.group(0)

    text = re.sub(r"\(([^()]*)\)", replace, text)
    return re.sub(r"\(\s*\)", " ", text)


def is_all_caps_line(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    cased = [ch for ch in letters if ch.lower() != ch.upper()]
    return bool(cased) and all(not ch.islower() for ch in cased)


def looks_section_header(line: str) -> bool:
    stripped = line.strip().strip(":")
    lowered = ascii_match_key(stripped)
    if lowered in COMMON_SECTION_HEADERS:
        return True
    if lowered.startswith(("saltiniai ", "nuorodos ", "taip pat ")):
        return True
    words = token_count(stripped)
    if words <= 6 and len(stripped) <= 80 and not re.search(r"[.!?…;]$", stripped):
        return True
    return False


def clean_line(raw_line: str) -> str:
    line = normalize_text(raw_line).replace("\u200b", "")
    line = REF_MARKER_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = strip_pronunciation_parentheticals(line)
    line = re.sub(r"\s+", " ", line).strip()
    line = line.strip("[]{}|")
    if line.startswith(("↑", "•")):
        line = line.lstrip("↑• ").strip()
    return unicodedata.normalize("NFC", line)


def source_lines(extract: str) -> list[str]:
    return [clean_line(line) for line in extract.splitlines() if clean_line(line)]


def is_list_heavy(extract: str) -> bool:
    lines = source_lines(extract)
    if len(lines) < 10:
        return False
    listish = 0
    for line in lines:
        if token_count(line) < 3 or LIST_PREFIX_RE.match(line):
            listish += 1
    return listish / len(lines) > 0.40


def split_prose_sentences(text: str) -> list[str]:
    placeholder = "\u2e3e"
    protected = ABBREVIATION_RE.sub(lambda match: match.group(0).replace(".", placeholder), text)

    def restore(piece: str) -> str:
        return piece.replace(placeholder, ".")

    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(protected):
        end = match.end(1)
        piece = restore(protected[start:end]).strip()
        if piece and token_count(piece) >= 3:
            sentences.append(piece)
        start = match.end()
    tail = restore(protected[start:]).strip()
    if tail and token_count(tail) >= 3:
        sentences.append(tail)
    return sentences


def looks_sentence_fragment(sentence: str) -> bool:
    if token_count(sentence) <= 4 and MONTH_FRAGMENT_RE.match(sentence):
        return True
    if token_count(sentence) <= 4 and re.search(r"\d", sentence) and re.search(r"\b(?:m|mėn|d)\.$", sentence):
        return True
    return False


def clean_extract_sentences(extract: str) -> list[str]:
    sentences: list[str] = []
    for raw_line in extract.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if COORDINATE_RE.match(line):
            continue
        if LIST_PREFIX_RE.match(line):
            continue
        if is_all_caps_line(line):
            continue
        if looks_section_header(line):
            continue
        if token_count(line) < 3:
            continue
        for sentence in split_prose_sentences(line):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if token_count(sentence) >= 3 and not looks_sentence_fragment(sentence):
                sentences.append(unicodedata.normalize("NFC", sentence))
    return sentences


def build_firewall_index(name: str, path: Path, units: Iterable[str]) -> FirewallIndex:
    exact: set[str] = set()
    shingles: set[tuple[str, ...]] = set()
    unit_count = 0
    for unit in units:
        normalized = strip_stress_lower(unit)
        if not normalized:
            continue
        unit_count += 1
        exact.add(normalized)
        words = word_keys(normalized)
        for index in range(0, max(0, len(words) - 7)):
            shingles.add(tuple(words[index : index + 8]))
    return FirewallIndex(
        name=name,
        path=path,
        source_units=unit_count,
        shingles=shingles,
        exact=exact,
    )


def load_chrestomatija_firewall(path: Path) -> FirewallIndex:
    if not path.exists():
        raise FileNotFoundError(f"missing chrestomatija gold JSONL: {path}")
    units: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            if not text.strip():
                raise ValueError(f"empty gold text at {path}:{line_number}")
            units.append(text)
    return build_firewall_index("chrestomatija", path, units)


def load_lrt_firewall(path: Path) -> FirewallIndex:
    if not path.exists():
        raise FileNotFoundError(f"missing LRT corpus text: {path}")
    units: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = normalize_text(line).strip()
        if not text:
            continue
        units.extend(split_prose_sentences(text))
    return build_firewall_index("lrt", path, units)


def sentence_has_shingle(sentence: str, shingles: set[tuple[str, ...]]) -> bool:
    words = word_keys(sentence)
    if len(words) < 8:
        return False
    return any(tuple(words[index : index + 8]) in shingles for index in range(0, len(words) - 7))


def apply_firewalls(
    title: str,
    sentences: list[str],
    indexes: list[FirewallIndex],
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    exact_drops = {index.name: 0 for index in indexes}
    shingle_drops = {index.name: 0 for index in indexes}
    kept: list[str] = []
    for sentence in sentences:
        normalized = strip_stress_lower(sentence)
        should_drop = False
        for index in indexes:
            if normalized in index.exact:
                exact_drops[index.name] += 1
                should_drop = True
            elif sentence_has_shingle(sentence, index.shingles):
                shingle_drops[index.name] += 1
                should_drop = True
        if not should_drop:
            kept.append(sentence)
    return kept, exact_drops, shingle_drops


def article_from_page(
    requested_title: str,
    bucket: str,
    page: dict[str, Any],
    firewalls: list[FirewallIndex],
) -> tuple[Article | None, dict[str, Any] | None]:
    if page.get("missing"):
        return None, {"bucket": bucket, "title": requested_title, "reason": "missing page"}
    title = str(page.get("title") or requested_title)
    extract = str(page.get("extract") or "")
    if not extract.strip():
        return None, {"bucket": bucket, "title": title, "reason": "empty extract"}
    if is_list_heavy(extract):
        return None, {"bucket": bucket, "title": title, "reason": "list-heavy article"}
    sentences = clean_extract_sentences(extract)
    if not sentences:
        return None, {"bucket": bucket, "title": title, "reason": "no clean sentences"}
    kept, exact_drops, shingle_drops = apply_firewalls(title, sentences, firewalls)
    tokens = sum(token_count(sentence) for sentence in kept)
    if tokens < MIN_ARTICLE_TOKENS:
        return None, {
            "bucket": bucket,
            "title": title,
            "reason": "too few tokens after cleaning/firewall",
            "tokens": tokens,
        }
    return (
        Article(
            title=title,
            bucket=bucket,
            url=str(page.get("fullurl") or page_url(title)),
            sentences=kept,
            tokens=tokens,
            raw_tokens=sum(token_count(sentence) for sentence in sentences),
            source_line_count=len(source_lines(extract)),
            firewall_exact=exact_drops,
            firewall_shingle=shingle_drops,
        ),
        None,
    )


def select_articles(
    articles_by_bucket: dict[str, list[Article]],
    max_tokens: int,
) -> list[Article]:
    bucket_names = [bucket.name for bucket in TOPIC_BUCKETS]
    quota = max_tokens / len(bucket_names)
    selected: list[Article] = []
    used: set[tuple[str, str]] = set()
    bucket_tokens = {name: 0 for name in bucket_names}
    total = 0

    for name in bucket_names:
        for article in articles_by_bucket.get(name, []):
            key = (article.bucket, title_key(article.title))
            if key in used:
                continue
            if total + article.tokens > max_tokens:
                continue
            if bucket_tokens[name] + article.tokens > quota:
                continue
            selected.append(article)
            used.add(key)
            bucket_tokens[name] += article.tokens
            total += article.tokens

    bucket_cap = quota * 1.35
    while total < max_tokens * 0.985:
        progress = False
        for name in sorted(bucket_names, key=lambda item: bucket_tokens[item]):
            remaining_global = max_tokens - total
            remaining_bucket = int(bucket_cap - bucket_tokens[name])
            if remaining_global <= 0 or remaining_bucket <= 0:
                continue
            for article in articles_by_bucket.get(name, []):
                key = (article.bucket, title_key(article.title))
                if key in used:
                    continue
                if article.tokens <= min(remaining_global, remaining_bucket):
                    selected.append(article)
                    used.add(key)
                    bucket_tokens[name] += article.tokens
                    total += article.tokens
                    progress = True
                    break
        if not progress:
            break
    return selected


def aggregate_firewall_report(
    articles: list[Article],
    indexes: list[FirewallIndex],
) -> dict[str, FirewallDropReport]:
    reports = {
        index.name: FirewallDropReport(
            source_units=index.source_units,
            source_8_word_shingles=len(index.shingles),
        )
        for index in indexes
    }
    for article in articles:
        for index in indexes:
            report = reports[index.name]
            exact = article.firewall_exact.get(index.name, 0)
            shingle = article.firewall_shingle.get(index.name, 0)
            report.dropped_exact += exact
            report.dropped_shingle += shingle
            dropped = exact + shingle
            if dropped:
                report.dropped_by_article[article.title] = (
                    report.dropped_by_article.get(article.title, 0) + dropped
                )
    for report in reports.values():
        report.dropped_by_article = dict(sorted(report.dropped_by_article.items()))
    return reports


def build_corpus(
    client: ThrottledMediaWikiClient,
    max_tokens: int,
    gold_path: Path,
    lrt_path: Path,
    seed: int,
    max_candidates_per_bucket: int,
    category_depth: int,
    max_categories_per_bucket: int,
) -> Corpus:
    firewalls = [load_chrestomatija_firewall(gold_path), load_lrt_firewall(lrt_path)]
    rng = random.Random(seed)
    candidate_titles_by_bucket: dict[str, int] = {}
    discovered_by_bucket: dict[str, int] = {}
    articles_by_bucket: dict[str, list[Article]] = {}
    skipped: list[dict[str, Any]] = []
    globally_seen_titles: set[str] = set()

    for bucket in TOPIC_BUCKETS:
        titles = collect_bucket_titles(
            client=client,
            bucket=bucket,
            max_titles=max_candidates_per_bucket,
            category_depth=category_depth,
            max_categories=max_categories_per_bucket,
        )
        candidate_titles_by_bucket[bucket.name] = len(titles)
        rng.shuffle(titles)
        unique_titles: list[str] = []
        for title in titles:
            key = title_key(title)
            if key in globally_seen_titles:
                continue
            globally_seen_titles.add(key)
            unique_titles.append(title)
        discovered_by_bucket[bucket.name] = len(unique_titles)
        bucket_articles: list[Article] = []
        clean_pool_tokens = 0
        bucket_pool_target = int((max_tokens / len(TOPIC_BUCKETS)) * ARTICLE_POOL_MULTIPLIER)
        for title_batch in chunked(unique_titles, TITLE_BATCH_SIZE):
            pages = fetch_extract_pages(client, title_batch)
            for title in title_batch:
                page = pages.get(title)
                if page is None:
                    skipped.append({"bucket": bucket.name, "title": title, "reason": "not returned"})
                    continue
                article, skip = article_from_page(title, bucket.name, page, firewalls)
                if article is not None:
                    bucket_articles.append(article)
                    clean_pool_tokens += article.tokens
                    if clean_pool_tokens >= bucket_pool_target:
                        break
                elif skip is not None:
                    skipped.append(skip)
            if clean_pool_tokens >= bucket_pool_target:
                    break
        articles_by_bucket[bucket.name] = bucket_articles
        print(
            f"{bucket.name}: {len(unique_titles):,} candidate titles, "
            f"{len(bucket_articles):,} clean articles, "
            f"{clean_pool_tokens:,} clean-pool tokens"
        )

    selected = select_articles(articles_by_bucket, max_tokens=max_tokens)
    tokens = sum(article.tokens for article in selected)
    firewall = aggregate_firewall_report(selected, firewalls)
    return Corpus(
        articles=selected,
        tokens=tokens,
        firewall=firewall,
        skipped=skipped,
        discovered_by_bucket=discovered_by_bucket,
        candidate_titles_by_bucket=candidate_titles_by_bucket,
    )


def bucket_distribution(corpus: Corpus) -> dict[str, dict[str, int | float]]:
    distribution: dict[str, dict[str, int | float]] = {}
    for bucket in (bucket.name for bucket in TOPIC_BUCKETS):
        articles = [article for article in corpus.articles if article.bucket == bucket]
        tokens = sum(article.tokens for article in articles)
        distribution[bucket] = {
            "articles": len(articles),
            "tokens": tokens,
            "share": tokens / corpus.tokens if corpus.tokens else 0.0,
        }
    return distribution


def write_outputs(
    corpus: Corpus,
    output: Path,
    max_tokens: int,
    request_interval: float,
    max_candidates_per_bucket: int,
    category_depth: int,
    max_categories_per_bucket: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, article in enumerate(corpus.articles):
            for sentence in article.sentences:
                handle.write(sentence + "\n")
            if index + 1 < len(corpus.articles):
                handle.write("\n")

    meta_path = output.with_suffix(".meta.json")
    meta = {
        "source": API_URL,
        "source_license": "CC BY-SA; article URLs retained for attribution.",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "user_agent": USER_AGENT,
        "request_interval_seconds": request_interval,
        "max_tokens": max_tokens,
        "tokens": corpus.tokens,
        "sentences": sum(len(article.sentences) for article in corpus.articles),
        "topic_buckets": [
            {"name": bucket.name, "categories": list(bucket.categories)}
            for bucket in TOPIC_BUCKETS
        ],
        "candidate_titles_by_bucket": corpus.candidate_titles_by_bucket,
        "deduped_titles_by_bucket": corpus.discovered_by_bucket,
        "selection": bucket_distribution(corpus),
        "fetch_limits": {
            "max_candidates_per_bucket": max_candidates_per_bucket,
            "category_depth": category_depth,
            "max_categories_per_bucket": max_categories_per_bucket,
            "min_article_tokens": MIN_ARTICLE_TOKENS,
            "article_pool_multiplier": ARTICLE_POOL_MULTIPLIER,
            "extract_intro_only": EXTRACT_INTRO_ONLY,
        },
        "articles": [
            {
                "title": article.title,
                "bucket": article.bucket,
                "url": article.url,
                "tokens": article.tokens,
                "sentences": len(article.sentences),
                "raw_tokens_before_firewall": article.raw_tokens,
                "source_line_count": article.source_line_count,
                "firewall_dropped": {
                    name: article.firewall_exact.get(name, 0)
                    + article.firewall_shingle.get(name, 0)
                    for name in corpus.firewall
                },
            }
            for article in corpus.articles
        ],
        "firewall": {
            name: {
                "source_units": report.source_units,
                "source_8_word_shingles": report.source_8_word_shingles,
                "dropped_exact": report.dropped_exact,
                "dropped_shingle": report.dropped_shingle,
                "dropped_total": report.dropped_total,
                "dropped_by_article": report.dropped_by_article,
            }
            for name, report in corpus.firewall.items()
        },
        "skipped": corpus.skipped,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta_path


def print_bucket_distribution(corpus: Corpus) -> None:
    print("bucket distribution:")
    for bucket, data in bucket_distribution(corpus).items():
        print(
            f"  {bucket:10} {int(data['tokens']):7,} tokens "
            f"({float(data['share']):5.1%}) | {int(data['articles']):3,} articles"
        )


def print_firewall_report(corpus: Corpus) -> None:
    print("firewall report:")
    for name, report in corpus.firewall.items():
        print(f"  {name}:")
        print(f"    source units: {report.source_units:,}")
        print(f"    source 8-word shingles: {report.source_8_word_shingles:,}")
        print(f"    dropped exact: {report.dropped_exact:,}")
        print(f"    dropped shingle: {report.dropped_shingle:,}")
        print(f"    dropped total: {report.dropped_total:,}")
        if report.dropped_by_article:
            print("    dropped by article:")
            for title, count in report.dropped_by_article.items():
                print(f"      {title}: {count:,}")
        else:
            print("    dropped by article: none")


def print_samples(corpus: Corpus, sample_count: int, seed: int) -> None:
    rows: list[tuple[str, str]] = []
    for article in corpus.articles:
        rows.extend((article.title, sentence) for sentence in article.sentences)
    samples = random.Random(seed).sample(rows, k=min(sample_count, len(rows)))
    print(f"{len(samples)} random sample sentences:")
    for index, (title, sentence) in enumerate(samples, start=1):
        print(f"{index}. [{title}] {sentence}")


def print_summary(corpus: Corpus) -> None:
    print(f"articles: {len(corpus.articles):,}")
    print(f"tokens: {corpus.tokens:,}")
    print(f"sentences: {sum(len(article.sentences) for article in corpus.articles):,}")
    skipped_by_reason: dict[str, int] = {}
    for item in corpus.skipped:
        reason = str(item.get("reason") or "unknown")
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
    print("skipped:")
    for reason, count in sorted(skipped_by_reason.items()):
        print(f"  {reason}: {count:,}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=280000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--lrt", type=Path, default=DEFAULT_LRT)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-candidates-per-bucket", type=int, default=2200)
    parser.add_argument("--category-depth", type=int, default=2)
    parser.add_argument("--max-categories-per-bucket", type=int, default=28)
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
    if args.max_candidates_per_bucket < 1:
        parser.error("--max-candidates-per-bucket must be positive")
    if args.category_depth < 0:
        parser.error("--category-depth must be non-negative")
    if args.max_categories_per_bucket < len(TOPIC_BUCKETS):
        parser.error("--max-categories-per-bucket must be at least the bucket count")

    with ThrottledMediaWikiClient(args.request_interval) as client:
        corpus = build_corpus(
            client=client,
            max_tokens=args.max_tokens,
            gold_path=args.gold,
            lrt_path=args.lrt,
            seed=args.seed,
            max_candidates_per_bucket=args.max_candidates_per_bucket,
            category_depth=args.category_depth,
            max_categories_per_bucket=args.max_categories_per_bucket,
        )

    meta_path = write_outputs(
        corpus=corpus,
        output=args.output,
        max_tokens=args.max_tokens,
        request_interval=args.request_interval,
        max_candidates_per_bucket=args.max_candidates_per_bucket,
        category_depth=args.category_depth,
        max_categories_per_bucket=args.max_categories_per_bucket,
    )
    print_bucket_distribution(corpus)
    print_firewall_report(corpus)
    print_summary(corpus)
    print_samples(corpus, sample_count=args.sample_count, seed=args.seed)
    print(f"wrote: {safe_relative(args.output)}")
    print(f"meta:  {safe_relative(meta_path)}")

    distribution = bucket_distribution(corpus)
    if args.max_tokens >= 200000:
        weak_buckets = [
            bucket
            for bucket, data in distribution.items()
            if int(data["tokens"]) <= 0 or float(data["share"]) < 0.05
        ]
        if len(distribution) < 8 or weak_buckets:
            print(
                "full-run bucket distribution failed: "
                f"{len(distribution)} buckets, weak buckets={weak_buckets}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
