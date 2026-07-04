from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal

from .dictionary_types import WordAccentEntry, WordDictionaryPutEntry
from .disambiguate import AccentVariant
from .vdu import fetch_word_entry


MAX_D1_BOUND_PARAMETERS = 90
PUT_PARAMETERS_PER_ENTRY = 8
NEGATIVE_WORD_TTL = timedelta(days=30)
FallbackMode = Literal["vdu", "none"]


class WordDictionary:
    def __init__(self, path: str | Path, migrations_dir: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.migrations_dir = Path(migrations_dir) if migrations_dir else find_migrations_dir()
        self.ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def ensure_schema(self) -> None:
        with self._lock:
            if not self._table_exists("words"):
                self._apply_all_migrations()
            self._ensure_missing_columns()
            self._conn.commit()

    def get_words(self, words: Iterable[str]) -> dict[str, WordAccentEntry | None]:
        keys = distinct_normalized_words(words)
        result: dict[str, WordAccentEntry | None] = {word: None for word in keys}
        now = datetime.now(UTC)

        with self._lock:
            for chunk in chunks(keys, MAX_D1_BOUND_PARAMETERS):
                if len(chunk) == 0:
                    continue

                placeholders = ", ".join("?" for _ in chunk)
                rows = self._conn.execute(
                    "SELECT word, variants, negative_until, default_form, accent_type, "
                    "default_form_title, accent_type_title "
                    f"FROM words WHERE word IN ({placeholders})",
                    chunk,
                ).fetchall()

                for row in rows:
                    word = normalize_word_key(row["word"])
                    variants = parse_variants(row["variants"])
                    if variants is None:
                        result[word] = None
                        continue

                    if row["accent_type_title"] is None:
                        result[word] = None
                        continue

                    if is_expired_negative(row["negative_until"], now):
                        result[word] = None
                        continue

                    result[word] = {
                        "variants": variants,
                        "defaultForm": nfc(row["default_form"]) if row["default_form"] else None,
                        "accentType": row["accent_type"],
                        "defaultFormTitle": nfc(row["default_form_title"])
                        if row["default_form_title"]
                        else None,
                        "accentTypeTitle": row["accent_type_title"],
                    }

        return result

    def put_words(self, entries: Iterable[WordDictionaryPutEntry]) -> None:
        now = datetime.now(UTC)
        fetched_at = isoformat_z(now)
        negative_until = isoformat_z(now + NEGATIVE_WORD_TTL)
        normalized_entries = normalize_put_entries(entries)

        with self._lock:
            for chunk in chunks(
                normalized_entries,
                MAX_D1_BOUND_PARAMETERS // PUT_PARAMETERS_PER_ENTRY,
            ):
                if len(chunk) == 0:
                    continue

                values_sql = ", ".join("(?, ?, ?, ?, ?, ?, ?, ?)" for _ in chunk)
                values: list[Any] = []
                for entry in chunk:
                    values.extend(
                        [
                            entry["word"],
                            json.dumps(entry["variants"], ensure_ascii=False),
                            fetched_at,
                            negative_until if is_negative_entry(entry) else None,
                            entry["defaultForm"],
                            entry["accentType"],
                            entry["defaultFormTitle"],
                            entry["accentTypeTitle"],
                        ]
                    )

                self._conn.execute(
                    "INSERT OR REPLACE INTO words "
                    "(word, variants, fetched_at, negative_until, default_form, "
                    "accent_type, default_form_title, accent_type_title) "
                    f"VALUES {values_sql}",
                    values,
                )
            self._conn.commit()

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _apply_all_migrations(self) -> None:
        if not self.migrations_dir.exists():
            create_words_schema(self._conn)
            return

        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        if not migration_files:
            create_words_schema(self._conn)
            return

        for migration in migration_files:
            self._conn.executescript(migration.read_text(encoding="utf-8"))

    def _ensure_missing_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(words)").fetchall()
        }
        additions = {
            "default_form": "ALTER TABLE words ADD COLUMN default_form TEXT",
            "accent_type": "ALTER TABLE words ADD COLUMN accent_type TEXT",
            "default_form_title": "ALTER TABLE words ADD COLUMN default_form_title TEXT",
            "accent_type_title": "ALTER TABLE words ADD COLUMN accent_type_title TEXT",
        }
        for column, sql in additions.items():
            if column not in columns:
                self._conn.execute(sql)


async def lookup_word_variants(
    dictionary: WordDictionary,
    word: str,
    fallback: FallbackMode = "vdu",
) -> list[AccentVariant]:
    key = normalize_word_key(word)
    cached = dictionary.get_words([key]).get(key)
    if cached is not None:
        return cached["variants"]

    if fallback == "none":
        return []

    entry = await fetch_word_entry(key)
    dictionary.put_words([{"word": key, **entry}])
    return entry["variants"]


def normalize_word_key(word: str) -> str:
    return nfc(word).lower()


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_put_entries(
    entries: Iterable[WordDictionaryPutEntry],
) -> list[WordDictionaryPutEntry]:
    by_word: dict[str, WordDictionaryPutEntry] = {}

    for entry in entries:
        word = normalize_word_key(entry["word"])
        by_word[word] = {
            "word": word,
            "variants": [
                {
                    "form": nfc(variant["form"]),
                    "info": variant["info"],
                    "mi": list(variant["mi"]),
                }
                for variant in entry["variants"]
            ],
            "defaultForm": nfc(entry["defaultForm"]) if entry["defaultForm"] else None,
            "accentType": entry["accentType"],
            "defaultFormTitle": nfc(entry["defaultFormTitle"])
            if entry["defaultFormTitle"]
            else None,
            "accentTypeTitle": entry["accentTypeTitle"],
        }

    return list(by_word.values())


def distinct_normalized_words(words: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(normalize_word_key(word) for word in words))


def chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def is_negative_entry(entry: WordDictionaryPutEntry) -> bool:
    return (
        len(entry["variants"]) == 0
        and entry["defaultForm"] is None
        and entry["defaultFormTitle"] is None
    )


def is_expired_negative(raw: str | None, now: datetime) -> bool:
    if raw is None:
        return False
    try:
        expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    return expires_at < now


def parse_variants(raw: str) -> list[AccentVariant] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(value, list):
        return None

    variants: list[AccentVariant] = []
    for item in value:
        if not is_accent_variant(item):
            return None
        variants.append(
            {
                "form": nfc(item["form"]),
                "info": item["info"],
                "mi": list(item["mi"]),
            }
        )
    return variants


def is_accent_variant(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("form"), str)
        and isinstance(value.get("info"), str)
        and isinstance(value.get("mi"), list)
        and all(isinstance(label, str) for label in value["mi"])
    )


def create_words_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS words (
          word TEXT PRIMARY KEY,
          variants TEXT NOT NULL,
          fetched_at TEXT NOT NULL,
          negative_until TEXT
        );
        ALTER TABLE words ADD COLUMN default_form TEXT;
        ALTER TABLE words ADD COLUMN accent_type TEXT;
        ALTER TABLE words ADD COLUMN default_form_title TEXT;
        ALTER TABLE words ADD COLUMN accent_type_title TEXT;
        """
    )


def find_migrations_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "migrations"
        if candidate.exists():
            return candidate
    return current.parents[3] / "migrations"


def isoformat_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
