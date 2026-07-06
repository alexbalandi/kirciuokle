# /// script
# requires-python = ">=3.11"
# dependencies = ["pypdf"]
# ///
"""Extract the accented gold sentences from Kirciuotu tekstu chrestomatija.

The source PDF is copyrighted teaching material and must stay in the
gitignored data directory. The local copy was recovered from the Wayback
Machine:
http://web.archive.org/web/20220120104613id_/http://www.esparama.lt/documents/10157/490675/2014_Kirciuotu_tekstu_chrestomatija_mok_knyga.pdf

Output is one JSON object per sentence:
{"text": "<accented sentence>", "page": N}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import count_stress_marks, normalize_lt, safe_relative  # noqa: E402


DEFAULT_PDF = SCRIPT_DIR / "data" / "eval" / "chrestomatija.pdf"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "eval" / "chrestomatija-gold.jsonl"
STRESS_MARKS = frozenset(("\u0300", "\u0301", "\u0303"))
WORD_RE = re.compile(r"(?:[^\W\d_][\u0300-\u036f]*)+", re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(
    r"([.!?…]+[\"'”’)\]]*)\s+(?=[A-ZĄČĘĖĮŠŲŪŽ„“\"'])"
)
SOURCE_WORD_RE = re.compile(
    r"\b("
    r"Vilnius|Kaunas|Vaga|Šviesa|leidykla|leidykla,|"
    r"Lietuvos rašytojų sąjungos|Pirmoji lietuviška knyga"
    r")\b",
    re.IGNORECASE,
)
DATE_LINE_RE = re.compile(r"\((?:g\.\s*)?\??\d{4}(?:[–-]\d{4})?\)")


@dataclass(frozen=True)
class PageDensity:
    page: int
    letters: int
    stress_marks: int
    density: float


def stress_count(text: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFD", text) if ch in STRESS_MARKS)


def letter_count(text: str) -> int:
    return sum(
        1
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch).startswith("L")
    )


def page_density(page: int, text: str) -> PageDensity:
    letters = letter_count(text)
    marks = stress_count(text)
    return PageDensity(
        page=page,
        letters=letters,
        stress_marks=marks,
        density=(marks * 100.0 / letters) if letters else 0.0,
    )


def first_letter(text: str) -> str | None:
    for ch in unicodedata.normalize("NFC", text):
        if unicodedata.category(ch).startswith("L"):
            return ch
        if not ch.isspace() and ch not in "\"'„“”’()[]":
            return None
    return None


def starts_lowercase(text: str) -> bool:
    ch = first_letter(text)
    return bool(ch and ch.islower())


def join_hyphenated_lines(lines: list[str]) -> list[str]:
    joined: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if joined and joined[-1].rstrip().endswith("-") and starts_lowercase(line):
            joined[-1] = joined[-1].rstrip()[:-1] + line.lstrip()
        else:
            joined.append(line)
    return joined


def is_all_caps_line(line: str) -> bool:
    letters = [ch for ch in unicodedata.normalize("NFC", line) if ch.isalpha()]
    cased = [ch for ch in letters if ch.lower() != ch.upper()]
    return bool(cased) and all(not ch.islower() for ch in cased)


def is_noise_line(line: str) -> bool:
    stripped = re.sub(r"\s+", " ", normalize_lt(line)).strip()
    if not stripped:
        return True
    if re.fullmatch(r"\d{1,3}", stripped):
        return True
    if re.match(r"^\d+[\s).,-]", stripped):
        return True
    if not any(ch.isalpha() for ch in stripped):
        return True

    marks = count_stress_marks(stripped)
    letters = letter_count(stripped)
    if marks:
        return False

    if SOURCE_WORD_RE.search(stripped):
        return True
    if DATE_LINE_RE.search(stripped):
        return True
    if is_all_caps_line(stripped):
        return True
    if letters <= 64:
        return True
    return False


def clean_page_text(text: str) -> str:
    lines = join_hyphenated_lines(text.splitlines())
    kept = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", normalize_lt(line)).strip()
        if not is_noise_line(normalized):
            kept.append(normalized)
    return "\n".join(kept)


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end(1)
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_sentence(text: str) -> str:
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", normalize_lt(text)).strip())


def keep_sentence(sentence: str) -> bool:
    return len(word_tokens(sentence)) >= 3 and count_stress_marks(sentence) >= 1


def density_histogram(densities: list[PageDensity]) -> list[str]:
    bins = [
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 10),
        (10, 12),
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
        (20, 999),
    ]
    lines = ["density histogram (stress marks per 100 letters):"]
    for low, high in bins:
        count = sum(1 for item in densities if low <= item.density < high)
        if count:
            label = f"{low:>2}-{high:<3}" if high < 999 else f"{low:>2}+"
            lines.append(f"  {label}: {count}")
    return lines


def format_page_list(pages: list[int], max_items: int = 24) -> str:
    if len(pages) <= max_items:
        return ", ".join(str(page) for page in pages)
    head = ", ".join(str(page) for page in pages[:max_items])
    return f"{head}, ... ({len(pages)} pages)"


def extract_sentences(
    pdf_path: Path,
    min_page: int,
    density_threshold: float,
) -> tuple[list[dict[str, object]], list[PageDensity], list[int], list[int]]:
    reader = PdfReader(str(pdf_path))
    densities: list[PageDensity] = []
    kept_pages: list[int] = []
    dropped_pages: list[int] = []
    rows: list[dict[str, object]] = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        density = page_density(page_index, raw_text)
        densities.append(density)
        keep_page = page_index >= min_page and density.density >= density_threshold
        if not keep_page:
            dropped_pages.append(page_index)
            continue
        kept_pages.append(page_index)
        cleaned = clean_page_text(raw_text)
        for sentence in split_sentences(cleaned):
            normalized = normalize_sentence(sentence)
            if keep_sentence(normalized):
                rows.append({"text": normalized, "page": page_index})

    return rows, densities, kept_pages, dropped_pages


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def print_density_report(
    densities: list[PageDensity],
    kept_pages: list[int],
    dropped_pages: list[int],
) -> None:
    print("per-page accent density:")
    for item in densities:
        print(
            f"  page {item.page:03d}: letters={item.letters:5d} "
            f"stress={item.stress_marks:4d} density={item.density:6.2f}"
        )
    for line in density_histogram(densities):
        print(line)
    print(f"kept pages: {format_page_list(kept_pages)}")
    print(f"dropped pages: {format_page_list(dropped_pages)}")


def print_stats(rows: list[dict[str, object]], sample_count: int, seed: int) -> None:
    token_total = sum(len(word_tokens(str(row["text"]))) for row in rows)
    stress_total = sum(count_stress_marks(str(row["text"])) for row in rows)
    print(f"sentence count: {len(rows):,}")
    print(f"token count: {token_total:,}")
    print(f"total stress marks: {stress_total:,}")
    samples = random.Random(seed).sample(rows, k=min(sample_count, len(rows)))
    print(f"{len(samples)} random sample sentences:")
    for index, row in enumerate(samples, start=1):
        print(f"{index}. [p. {row['page']}] {row['text']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-page", type=int, default=11)
    parser.add_argument("--density-threshold", type=float, default=10.0)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.pdf.exists():
        parser.error(f"missing PDF: {args.pdf}")

    rows, densities, kept_pages, dropped_pages = extract_sentences(
        pdf_path=args.pdf,
        min_page=args.min_page,
        density_threshold=args.density_threshold,
    )
    write_jsonl(args.output, rows)
    print_density_report(densities, kept_pages, dropped_pages)
    print_stats(rows, args.sample_count, args.seed)
    print(f"wrote: {safe_relative(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
