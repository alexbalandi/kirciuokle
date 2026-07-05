# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fetch the VLKK recommended-names data (vardai.vlkk.lt).

The letter pages list every recommended given name with its accented
nominative and gender; per-name pages add the kirčiuotė and the full accented
singular paradigm. VLKK is this project's declared normative authority and
the source is an official state-commission resource.

Writes data/vlkk_names.json:
  {"Vardas": {"accented": "Var̃das", "gender": "man",
              "class": "2", "cells": {"genitive": "Var̃do", ...}}}

Only names passed via --details-for (a file of lowercase word keys) get the
per-name fetch, keeping the crawl polite; everything else keeps just the
nominative. Re-runs skip names already detailed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

try:  # pragma: no cover
    from ._common import DATA_DIR, lower_key, nfc
except ImportError:  # pragma: no cover
    from _common import DATA_DIR, lower_key, nfc

BASE = "https://vardai.vlkk.lt"
UA = "kirciuokle-open-accentuator/1.0 (open-source Lithuanian accentuation research; polite crawl)"
OUT = DATA_DIR / "vlkk_names.json"

CASE_LABELS = {
    "Vard.": "nominative",
    "Kilm.": "genitive",
    "Naud.": "dative",
    "Gal.": "accusative",
    "Įnag.": "instrumental",
    "Viet.": "locative",
    "Šauksm.": "vocative",
}

NAME_LINK = re.compile(
    r"<a href='(https://vardai\.vlkk\.lt/vardas/[^']+)' class='names_list__links"
    r" names_list__links--(man|woman)'>([^<]+)</a>"
)
LETTER_LINK = re.compile(r'href="(https://vardai\.vlkk\.lt/sarasas/[^"]+)"')
# In the flattened page text, each accented form precedes its case label:
# "Kristinà, Vard. Kristìnos, Kilm. ... Kristìna! Šauksm."
CELL_RE = re.compile(r"([^\s,!<>]+)[,!]?\s+(Vard|Kilm|Naud|Gal|Įnag|Viet|Šauksm)\.")


async def fetch(client: httpx.AsyncClient, url: str) -> str:
    for attempt in range(3):
        try:
            r = await client.get(url)
            # the per-name pages render full content with a 404 status
            # (WordPress quirk) — accept anything with a real body
            if r.status_code == 200 or len(r.text) > 5000:
                return r.text
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.5 * (attempt + 1))
    return ""


def parse_detail(html: str) -> tuple[str | None, dict[str, str]]:
    text = re.sub(r"\s+", " ", html)
    klass = None
    m = re.search(r"Kirčiavimas.{0,80}?(\d\w*)\s*kirčiuotė", text)
    if m:
        klass = m.group(1)
    cells: dict[str, str] = {}
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain)
    for m in CELL_RE.finditer(plain):
        cells[CASE_LABELS[m.group(2) + "."]] = nfc(m.group(1))
    return klass, cells


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-for", type=Path, default=None,
                        help="File with lowercase word keys; matching names get per-name fetches.")
    parser.add_argument("--max-details", type=int, default=800)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    data: dict[str, dict] = {}
    if args.out.exists():
        data = json.loads(args.out.read_text(encoding="utf-8"))

    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=30, follow_redirects=True) as client:
        first = await fetch(client, f"{BASE}/sarasas/a/")
        letters = sorted(set(LETTER_LINK.findall(first)) | {f"{BASE}/sarasas/a/"})
        print(f"letter pages: {len(letters)}")
        for url in letters:
            html = await fetch(client, url)
            for _link, gender, accented in NAME_LINK.findall(html):
                accented = nfc(accented.strip())
                plain = lower_key(accented)
                entry = data.setdefault(plain.capitalize(), {})
                entry.setdefault("accented", accented)
                entry.setdefault("gender", gender)
            await asyncio.sleep(args.delay)
        print(f"names collected: {len(data)}")

        if args.details_for and args.details_for.exists():
            wanted_keys = {w.strip() for w in args.details_for.read_text(encoding="utf-8").splitlines() if w.strip()}
            todo = [
                name for name, entry in sorted(data.items())
                if "cells" not in entry and lower_key(name) in wanted_keys
            ][: args.max_details]
            print(f"detail fetches: {len(todo)}")
            for i, name in enumerate(todo):
                html = await fetch(client, f"{BASE}/vardas/{name}")
                klass, cells = parse_detail(html)
                if klass:
                    data[name]["class"] = klass
                if cells:
                    data[name]["cells"] = cells
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(todo)}")
                    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
                await asyncio.sleep(args.delay)

    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
    detailed = sum(1 for e in data.values() if "cells" in e)
    print(f"wrote {args.out} ({len(data)} names, {detailed} with paradigms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
