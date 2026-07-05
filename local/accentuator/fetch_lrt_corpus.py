# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fetch a fresh LRT article corpus for live guess evaluation.

The current LRT homepage advertises `/?rss` as its RSS endpoint; `/rss`
returns an HTML 404 page. Discovery uses that feed, then fetches each article
page and extracts paragraphs from the article body container.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "eval" / "lrt-corpus.txt"
RSS_URL = "https://www.lrt.lt/?rss"
FEED_URLS = [
    RSS_URL,
    "https://www.lrt.lt/naujienos/lietuvoje?rss",
    "https://www.lrt.lt/naujienos/pasaulyje?rss",
    "https://www.lrt.lt/naujienos/verslas?rss",
    "https://www.lrt.lt/naujienos/kultura?rss",
    "https://www.lrt.lt/naujienos/sportas?rss",
    "https://www.lrt.lt/naujienos/sveikata?rss",
    "https://www.lrt.lt/naujienos/mokslas-ir-it?rss",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)
MIN_REQUEST_INTERVAL = 1.05
MIN_ARTICLE_TOKENS = 500
WORD_RE = re.compile(r"[A-Za-zÀ-žĀ-ſ̀-ͯ]+")
LT_STOPWORDS = {
    "apie",
    "ar",
    "bei",
    "bet",
    "buvo",
    "dar",
    "dėl",
    "iki",
    "ir",
    "iš",
    "jis",
    "kad",
    "kaip",
    "ką",
    "kurie",
    "ne",
    "nuo",
    "o",
    "po",
    "prie",
    "su",
    "tačiau",
    "tai",
    "taip",
    "tarp",
    "teigė",
    "už",
    "yra",
}
LT_LETTERS = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")


@dataclass
class FeedItem:
    title: str
    url: str
    published: str | None


@dataclass
class Article:
    title: str
    url: str
    published: str | None
    fetched_at: str
    paragraphs: list[str]
    tokens: int


class ThrottledClient:
    def __init__(self) -> None:
        self._last_start = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            follow_redirects=True,
        )

    def __enter__(self) -> "ThrottledClient":
        self._client.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.__exit__(*exc)

    def get_text(self, url: str) -> str:
        now = time.monotonic()
        if self._last_start:
            delay = MIN_REQUEST_INTERVAL - (now - self._last_start)
            if delay > 0:
                time.sleep(delay)
        self._last_start = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response.text


class ArticleParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.article_div_depth = 0
        self.in_paragraph = False
        self.skip_stack: list[str] = []
        self.current: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name: value or "" for name, value in attrs}
        classes = set(attr.get("class", "").split())
        if not self.in_article:
            if tag == "div" and {"article-content", "js-text-selection"} <= classes:
                self.in_article = True
                self.article_div_depth = 1
            return

        if tag == "div":
            self.article_div_depth += 1
        if tag in {"script", "style", "noscript", "button", "svg"}:
            self.skip_stack.append(tag)
        if tag == "p":
            self.in_paragraph = True
            self.current = []
        elif tag == "br" and self.in_paragraph and not self.skip_stack:
            self.current.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self.in_article:
            return
        if self.skip_stack and tag == self.skip_stack[-1]:
            self.skip_stack.pop()
        if tag == "p" and self.in_paragraph:
            text = clean_paragraph("".join(self.current))
            if keep_paragraph(text):
                self.paragraphs.append(text)
            self.in_paragraph = False
            self.current = []
        if tag == "div":
            self.article_div_depth -= 1
            if self.article_div_depth <= 0:
                self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.in_article and self.in_paragraph and not self.skip_stack:
            self.current.append(data)


def clean_paragraph(text: str) -> str:
    text = html.unescape(text).replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def keep_paragraph(text: str) -> bool:
    if len(text) < 40:
        return False
    lowered = text.casefold()
    blocked_prefixes = (
        "taip pat skaitykite",
        "skaitykite daugiau",
        "daugiau apie",
        "lrt.lt primena",
    )
    return not lowered.startswith(blocked_prefixes)


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFD", text))


def type_key(token: str) -> str:
    return unicodedata.normalize("NFC", token).casefold()


def looks_lithuanian(text: str) -> bool:
    tokens = [type_key(token) for token in word_tokens(text)]
    if not tokens:
        return False
    stopword_hits = sum(1 for token in tokens if token in LT_STOPWORDS)
    lt_char_hits = sum(1 for char in text if char in LT_LETTERS)
    return stopword_hits / len(tokens) >= 0.025 or lt_char_hits >= 8


def parse_feed(xml_text: str) -> list[FeedItem]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []
    seen: set[str] = set()
    items: list[FeedItem] = []
    for item in channel.findall("item"):
        title = clean_paragraph(item.findtext("title") or "")
        url = clean_paragraph(item.findtext("link") or "")
        published = clean_paragraph(item.findtext("pubDate") or "") or None
        if not title or not url or url in seen:
            continue
        if "/naujienos/" not in url:
            continue
        seen.add(url)
        items.append(FeedItem(title=title, url=url, published=published))
    return items


def extract_paragraphs(html_text: str) -> list[str]:
    parser = ArticleParagraphParser()
    parser.feed(html_text)
    return parser.paragraphs


def iter_feed_items(client: ThrottledClient):
    seen: set[str] = set()
    for feed_url in FEED_URLS:
        feed_xml = client.get_text(feed_url)
        for item in parse_feed(feed_xml):
            if item.url in seen:
                continue
            seen.add(item.url)
            yield item


def fetch_articles(count: int) -> list[Article]:
    articles: list[Article] = []
    with ThrottledClient() as client:
        for item in iter_feed_items(client):
            if len(articles) >= count:
                break
            try:
                page = client.get_text(item.url)
            except httpx.HTTPError as exc:
                print(f"skip {item.url}: {exc}")
                continue
            paragraphs = extract_paragraphs(page)
            text = "\n".join(paragraphs)
            tokens = len(word_tokens(text))
            if tokens < MIN_ARTICLE_TOKENS:
                print(f"skip short article ({tokens} tokens): {item.url}")
                continue
            if not looks_lithuanian(text):
                print(f"skip non-Lithuanian-looking article: {item.url}")
                continue
            articles.append(
                Article(
                    title=item.title,
                    url=item.url,
                    published=item.published,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    paragraphs=paragraphs,
                    tokens=tokens,
                )
            )
            print(f"accepted {len(articles)}/{count}: {tokens} tokens  {item.url}")
    return articles


def write_outputs(articles: list[Article], output: Path) -> tuple[int, int, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, article in enumerate(articles):
            for paragraph in article.paragraphs:
                handle.write(paragraph + "\n")
            if index + 1 < len(articles):
                handle.write("\n")

    all_text = "\n".join("\n".join(article.paragraphs) for article in articles)
    tokens = word_tokens(all_text)
    types = {type_key(token) for token in tokens}
    meta_path = output.with_suffix(".meta.json")
    meta = {
        "source": FEED_URLS,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "attribution": "Paragraph text and URLs fetched from LRT.lt for evaluation.",
        "user_agent": USER_AGENT,
        "articles": [
            {
                "title": article.title,
                "url": article.url,
                "published": article.published,
                "fetched_at": article.fetched_at,
                "paragraphs": len(article.paragraphs),
                "tokens": article.tokens,
            }
            for article in articles
        ],
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(tokens), len(types), meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.articles < 1:
        parser.error("--articles must be positive")

    articles = fetch_articles(args.articles)
    if len(articles) < args.articles:
        print(f"only fetched {len(articles)} usable articles out of requested {args.articles}")
        return 1

    token_count, type_count, meta_path = write_outputs(articles, args.output)
    print(f"wrote: {args.output}")
    print(f"meta:  {meta_path}")
    print(f"articles: {len(articles):,}")
    print(f"tokens:   {token_count:,}")
    print(f"types:    {type_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
