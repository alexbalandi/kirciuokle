# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4"]
# ///
"""Fetch VLKK's consolidated recommended-stress-variants list.

https://www.vlkk.lt/aktualiausios-temos/tartis-ir-kirciavimas/
rekomenduojamu-kirciavimo-variantu-sarasas is the alphabetical list of every
word whose normative stress VLKK changed relative to DLKŽ, with kirčiuotė
numbers and the recommendation id (K-nn). VLKK is this project's declared
normative arbiter and official normative acts are not copyright-protected,
so this is dictionary-tier, provenance-clean data.

Entries are separated by the (K-nn) marker; " / " separates equally
acceptable variants. Pronunciation-only variants from the 2022 tarties
recommendation (marked with ō, pusilgumo dots, apostrophes, or a
parenthesised kirčiuotė) are skipped. Writes data/vlkk_recommendations.json.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:  # pragma: no cover
    from ._common import DATA_DIR, normalize_lt, strip_accents
except ImportError:  # pragma: no cover
    from _common import DATA_DIR, normalize_lt, strip_accents

URL = (
    "https://www.vlkk.lt/aktualiausios-temos/tartis-ir-kirciavimas/"
    "rekomenduojamu-kirciavimo-variantu-sarasas"
)
OUTPUT = DATA_DIR / "vlkk_recommendations.json"

ENTRY_RE = re.compile(r"\(K-(\d+)\)")
CLS_RE = re.compile(r"^([1234])\s*([ab])?$")
GLOSS_RE = re.compile(r"[„“]([^„“”]*)[“”]")
# pronunciation-notation characters that mark 2022 tarties-only variants
PRONUNCIATION_CHARS = set("ōʼ·")
WORD_RE = re.compile(r"^[a-ząčęėįšųūžA-ZĄČĘĖĮŠŲŪŽ̀́̃]+$")
POS_HINTS = {"prv.": "adv", "jst.": "intj", "dll.": "particle", "prl.": "prep"}


def fetch_text() -> str:
    response = requests.get(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
            "Accept-Language": "lt,en;q=0.8",
        },
        timeout=60,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(" ")
    text = unicodedata.normalize("NFC", text.replace("\xa0", " "))
    # the list body starts after this heading phrase
    start = text.find("abėcėlinis sąrašas")
    return text[start:] if start > -1 else text


def parse_variant(chunk: str, prev_form: str | None) -> dict | None:
    """One " / "-separated variant -> {forms: [...], cls: str|None} or None."""
    chunk = chunk.replace("*", " ").strip(" ,;")
    if not chunk or PRONUNCIATION_CHARS & set(chunk):
        return None
    if "(" in chunk:  # parenthesised kirčiuotė = pronunciation-only variant
        return None
    gloss = GLOSS_RE.search(chunk)
    if gloss:
        chunk = GLOSS_RE.sub(" ", chunk).strip(" ,;")
    tokens = [t for t in re.split(r"[\s,]+", chunk) if t]
    if not tokens:
        return None

    feminine = None
    fem_tokens = [t for t in tokens if t.startswith("-")]
    if fem_tokens:
        feminine = fem_tokens[0]
        tokens = [t for t in tokens if not t.startswith("-")]

    cls = None
    if tokens and CLS_RE.match(tokens[-1] if len(tokens[-1]) > 1 else tokens[-1]):
        cls = tokens.pop()
    elif len(tokens) >= 2 and CLS_RE.match("".join(tokens[-2:])):
        cls = "".join(tokens[-2:])
        tokens = tokens[:-2]
    elif tokens and CLS_RE.match(tokens[-1]):
        cls = tokens.pop()

    pos_hint = None
    if tokens and tokens[-1] in POS_HINTS:
        pos_hint = POS_HINTS[tokens.pop()]

    forms = [normalize_lt(t) for t in tokens if WORD_RE.match(t)]
    if len(forms) != len(tokens):  # something unparseable inside
        return None
    if not forms:
        if cls and prev_form:  # bare-kirčiuotė variant: same form, new class
            forms = [prev_form]
        else:
            return None
    return {"forms": forms, "cls": cls, "feminine": feminine, "pos": pos_hint}


def parse_entries(text: str) -> list[dict]:
    pieces = ENTRY_RE.split(text)
    entries = []
    seen = set()
    # pieces alternate: body, rec, body, rec, ...
    for body, rec in zip(pieces[0::2], pieces[1::2]):
        body = body.strip()
        # the tail after the previous entry's marker is this entry's text;
        # entries never span sentences, so cut at the last sentence break
        raw_variants = [v for v in body.split(" / ")]
        variants = []
        prev_form = None
        for raw in raw_variants:
            parsed = parse_variant(raw, prev_form)
            if parsed:
                variants.append(parsed)
                if parsed["forms"]:
                    prev_form = parsed["forms"][0]
        if not variants:
            continue
        head = strip_accents(variants[0]["forms"][0]).lower()
        key = (head, f"K-{rec}", json.dumps(variants, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        entries.append({"head": head, "rec": f"K-{rec}", "variants": variants})
    return entries


def main() -> int:
    text = fetch_text()
    entries = parse_entries(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    n_forms = sum(len(v["forms"]) for e in entries for v in e["variants"])
    print(f"wrote {OUTPUT} ({len(entries)} entries, {n_forms} accented forms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
