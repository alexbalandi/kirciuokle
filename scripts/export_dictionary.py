# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Export the production D1 word dictionary into a local SQLite file for
the self-hosted replica (local/).

Usage:
    uv run scripts/export_dictionary.py                    # -> local/data/words.sqlite
    uv run scripts/export_dictionary.py -o path/to.sqlite
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import httpx

DB_ID = "09f3ad62-f4b7-4869-bf69-b941f4316bd1"
PAGE = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
  word TEXT PRIMARY KEY,
  variants TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  negative_until TEXT,
  default_form TEXT,
  accent_type TEXT,
  default_form_title TEXT,
  accent_type_title TEXT
);
"""


def load_env() -> tuple[str, str]:
    env = {}
    for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env["CLOUDFLARE_API_TOKEN"], env["CLOUDFLARE_ACCOUNT_ID"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).parent.parent / "local" / "data" / "words.sqlite"),
    )
    args = ap.parse_args()

    token, account = load_env()
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{DB_ID}/query"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    db = sqlite3.connect(out)
    db.executescript(SCHEMA)

    columns = [
        "word",
        "variants",
        "fetched_at",
        "negative_until",
        "default_form",
        "accent_type",
        "default_form_title",
        "accent_type_title",
    ]
    total = 0
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        offset = 0
        while True:
            r = client.post(
                url,
                json={
                    "sql": f"SELECT {', '.join(columns)} FROM words ORDER BY word LIMIT ? OFFSET ?",
                    "params": [PAGE, offset],
                },
            )
            r.raise_for_status()
            payload = r.json()
            if not payload.get("success"):
                raise RuntimeError(f"D1 error: {payload.get('errors')}")
            rows = payload["result"][0].get("results", [])
            if not rows:
                break
            db.executemany(
                f"INSERT OR REPLACE INTO words ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                [tuple(row[c] for c in columns) for row in rows],
            )
            db.commit()
            total += len(rows)
            offset += PAGE
            print(f"  {total} rows...", file=sys.stderr)

    db.close()
    print(f"exported {total} words -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
