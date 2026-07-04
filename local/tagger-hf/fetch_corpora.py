# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fetch and cache ALKSNIS and MATAS source corpora."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
USER_AGENT = "tagger-hf-fetch/1.0"

ALKSNIS_BASE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Lithuanian-ALKSNIS/master"
)
ALKSNIS_FILES = {
    "train": "lt_alksnis-ud-train.conllu",
    "dev": "lt_alksnis-ud-dev.conllu",
    "test": "lt_alksnis-ud-test.conllu",
}

MATAS_URL = (
    "https://clarin-repo.lt/server/api/core/bitstreams/"
    "c985d423-14b1-408a-ab47-6cb61a69094c/content"
)
MATAS_ZIP = "MATAS3.conllu.zip"
MATAS_CONLLU = "MATAS3.conllu"
MATAS_EXPECTED_ZIP_BYTES = 24_221_501
MATAS_SIZE_TOLERANCE_BYTES = 1_048_576


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())


def validate_matas_zip(path: Path) -> None:
    size = path.stat().st_size
    delta = abs(size - MATAS_EXPECTED_ZIP_BYTES)
    if delta > MATAS_SIZE_TOLERANCE_BYTES:
        raise RuntimeError(
            f"{path} is {size:,} bytes; expected about "
            f"{MATAS_EXPECTED_ZIP_BYTES:,}. Re-run with --force if the cache is stale."
        )


def fetch_file(url: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"reusing {path}")
        return
    print(f"downloading {url} -> {path}")
    download(url, path)


def fetch_alksnis(raw_dir: Path, force: bool) -> None:
    for filename in ALKSNIS_FILES.values():
        fetch_file(f"{ALKSNIS_BASE_URL}/{filename}", raw_dir / filename, force)


def extract_matas(raw_dir: Path, force: bool) -> None:
    zip_path = raw_dir / MATAS_ZIP
    conllu_path = raw_dir / MATAS_CONLLU
    if conllu_path.exists() and not force:
        print(f"reusing {conllu_path}")
        return

    print(f"extracting {MATAS_CONLLU} from {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        try:
            member = archive.getinfo(MATAS_CONLLU)
        except KeyError as exc:
            names = ", ".join(archive.namelist())
            raise RuntimeError(f"{zip_path} does not contain {MATAS_CONLLU}: {names}") from exc

        raw_dir.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, conllu_path.open("wb") as destination:
            destination.write(source.read())


def fetch_matas(raw_dir: Path, force: bool) -> None:
    zip_path = raw_dir / MATAS_ZIP
    fetch_file(MATAS_URL, zip_path, force)
    validate_matas_zip(zip_path)
    extract_matas(raw_dir, force)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="corpus cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-extract even when cached files exist",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    fetch_alksnis(args.raw_dir, args.force)
    fetch_matas(args.raw_dir, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
