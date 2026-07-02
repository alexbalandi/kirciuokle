# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Politely seed the D1 word dictionary from the VDU kirciuokle.

Words come either from a frequency list (default: hermitdave/FrequencyWords
Lithuanian 50k, built from OpenSubtitles) or from a local text file
(--words-from-text). Words already present in D1 are skipped, so the script
is resumable and can be re-run any time.

VDU is queried at a deliberately low rate (default 1.5 req/s) with a
descriptive User-Agent. Results (including negatives) are upserted into D1
in batches via the Cloudflare HTTP API using credentials from .env.

Usage:
    uv run scripts/seed_dictionary.py --limit 5000
    uv run scripts/seed_dictionary.py --words-from-text scripts/sample.txt
"""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import accent_text  # noqa: E402  (nonce + raw VDU calls)


def flatten_variants(accent_info: list[dict]) -> list[dict]:
    """Replicate the worker's flattenVariants exactly: [{form, info, mi}]."""
    out = []
    for entry in accent_info:
        information = entry.get("information") or []
        info = "; ".join(
            filter(
                None,
                (
                    " - ".join(filter(None, (i.get("mi"), i.get("meaning"))))
                    for i in information
                ),
            )
        )
        mi = [i["mi"] for i in information if i.get("mi")]
        for form in entry.get("accented") or []:
            out.append(
                {"form": unicodedata.normalize("NFC", form), "info": info, "mi": mi}
            )
    return out


async def fetch_variants(client: httpx.AsyncClient, nonce: str, word: str) -> list[dict]:
    msg = await accent_text.vdu_call(
        client, nonce, {"action": "word_accent", "word": word}
    )
    return flatten_variants(msg.get("accentInfo") or [])


async def fetch_default(
    client: httpx.AsyncClient, nonce: str, word: str
) -> tuple[str | None, str]:
    """Canonical (default_form, accent_type) from a single-word text_accents call.

    A side with no accented WORD part stores form None and the explicit
    type "NONE" (NULL in the DB means "legacy row, incomplete")."""
    msg = await accent_text.vdu_call(
        client, nonce, {"action": "text_accents", "body": word}
    )
    for part in msg.get("textParts") or []:
        if part.get("type") == "WORD" and part.get("accented"):
            return (
                unicodedata.normalize("NFC", part["accented"]),
                part.get("accentType") or "ONE",
            )
    return None, "NONE"


async def fetch_entry(client: httpx.AsyncClient, nonce: str, word: str) -> dict:
    """Mirror the worker's fetchWordEntry: variants + lower and title sides."""
    variants = await fetch_variants(client, nonce, word)
    lower_form, lower_type = await fetch_default(client, nonce, word)
    title = word[0].upper() + word[1:]
    title_form, title_type = await fetch_default(client, nonce, title)
    return {
        "word": word,
        "variants": variants,
        "default_form": lower_form,
        "accent_type": lower_type,
        "default_form_title": title_form,
        "accent_type_title": title_type,
    }

FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/"
    "content/2018/lt/lt_50k.txt"
)
LT_WORD = re.compile(r"^[a-ząčęėįšųūž]+$")
DB_ID = "09f3ad62-f4b7-4869-bf69-b941f4316bd1"
NEGATIVE_DAYS = 30
PARAM_LIMIT = 100  # D1 bound-parameter limit per statement
ROW_PARAMS = 8


def load_env() -> tuple[str, str]:
    env = {}
    for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env["CLOUDFLARE_API_TOKEN"], env["CLOUDFLARE_ACCOUNT_ID"]


class D1:
    def __init__(self, client: httpx.AsyncClient, token: str, account: str):
        self.client = client
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{DB_ID}/query"
        self.headers = {"Authorization": f"Bearer {token}"}

    async def query(self, sql: str, params: list) -> list[dict]:
        r = await self.client.post(
            self.url, headers=self.headers, json={"sql": sql, "params": params}, timeout=60
        )
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(f"D1 error: {payload.get('errors')}")
        return payload["result"][0].get("results", [])

    async def existing(self, words: list[str]) -> set[str]:
        """Words with a complete entry (both case sides evaluated)."""
        found: set[str] = set()
        for i in range(0, len(words), PARAM_LIMIT):
            chunk = words[i : i + PARAM_LIMIT]
            ph = ",".join("?" * len(chunk))
            rows = await self.query(
                f"SELECT word FROM words WHERE word IN ({ph}) "
                "AND accent_type_title IS NOT NULL",
                chunk,
            )
            found.update(r["word"] for r in rows)
        return found

    async def upsert(self, entries: list[dict]) -> None:
        now = datetime.now(timezone.utc)
        neg_until = (now + timedelta(days=NEGATIVE_DAYS)).isoformat()
        rows_per_stmt = PARAM_LIMIT // ROW_PARAMS
        for i in range(0, len(entries), rows_per_stmt):
            chunk = entries[i : i + rows_per_stmt]
            values = ",".join("(?,?,?,?,?,?,?,?)" for _ in chunk)
            params: list = []
            for e in chunk:
                negative = (
                    not e["variants"]
                    and e["default_form"] is None
                    and e["default_form_title"] is None
                )
                params += [
                    e["word"],
                    json.dumps([] if negative else e["variants"], ensure_ascii=False),
                    now.isoformat(),
                    neg_until if negative else None,
                    e["default_form"],
                    e["accent_type"],
                    e["default_form_title"],
                    e["accent_type_title"],
                ]
            await self.query(
                "INSERT OR REPLACE INTO words "
                "(word, variants, fetched_at, negative_until, default_form, accent_type, "
                "default_form_title, accent_type_title) "
                f"VALUES {values}",
                params,
            )


async def candidate_words(client: httpx.AsyncClient, args) -> list[str]:
    if args.words_from_text:
        text = Path(args.words_from_text).read_text(encoding="utf-8")
        raw = re.findall(r"[A-Za-zÀ-žĀ-ſ]+", text)
        words = [w.lower() for w in raw]
    else:
        r = await client.get(FREQ_URL, timeout=60)
        r.raise_for_status()
        words = [line.split(" ")[0] for line in r.text.splitlines() if line.strip()]
    seen: dict[str, None] = {}
    for w in words:
        w = unicodedata.normalize("NFC", w)
        if LT_WORD.match(w):
            seen.setdefault(w)
    return list(seen)[: args.limit]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument(
        "--rps",
        type=float,
        default=0.8,
        help="max words/sec (each word costs three VDU requests)",
    )
    ap.add_argument("--words-from-text", help="seed the words of a text file instead of the frequency list")
    args = ap.parse_args()

    token, account = load_env()
    async with httpx.AsyncClient(
        headers={"User-Agent": "kirciuokle-dictionary-seeder/1.0 (alexbalandi@gmail.com; polite, rate-limited)"}
    ) as client:
        d1 = D1(client, token, account)
        words = await candidate_words(client, args)
        print(f"candidate words: {len(words)}", file=sys.stderr)

        have = await d1.existing(words)
        todo = [w for w in words if w not in have]
        print(f"already in D1: {len(have)}, to fetch: {len(todo)}", file=sys.stderr)
        if not todo:
            return

        nonce = await accent_text.get_nonce(client)
        interval = 1.0 / args.rps
        batch: list[dict] = []
        done = negatives = 0

        for word in todo:
            started = asyncio.get_event_loop().time()
            try:
                entry = await fetch_entry(client, nonce, word)
            except Exception:
                # refresh nonce once, then skip on repeat failure
                try:
                    nonce = await accent_text.get_nonce(client)
                    entry = await fetch_entry(client, nonce, word)
                except Exception as e:
                    print(f"  skip {word}: {e}", file=sys.stderr)
                    continue
            batch.append(entry)
            done += 1
            negatives += (
                entry["default_form"] is None and entry["default_form_title"] is None
            )
            if len(batch) >= 200:
                await d1.upsert(batch)
                batch.clear()
                print(f"  progress: {done}/{len(todo)} ({negatives} negatives)", file=sys.stderr)
            elapsed = asyncio.get_event_loop().time() - started
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

        if batch:
            await d1.upsert(batch)
        print(f"seeded {done} words ({negatives} negatives)", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
