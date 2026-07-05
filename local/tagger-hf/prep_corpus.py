# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare MATAS + ALKSNIS for combined UPOS|FEATS token classification."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from coverage_diff import (
    FeatsToken,
    coverage_rows,
    filter_feats_keys,
    format_coverage_table,
    key_filter,
)
from lemma_scripts import make_lemma_script, script_inventory, top_script_coverage
from metrics import combined_label, feats_string, parse_feats, split_label


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "combined"
DEFAULT_SEED = 13

ALKSNIS_FILES = {
    "train": "lt_alksnis-ud-train.conllu",
    "dev": "lt_alksnis-ud-dev.conllu",
    "test": "lt_alksnis-ud-test.conllu",
}
MATAS_FILE = "MATAS3.conllu"
VALID_SOURCES = {"matas", "alksnis"}
COVERAGE_DELTA = 0.10

# Scoring-slot label mode keeps this fixed UD FEATS key set:
# Case, Gender, Number, Tense, Person, Voice, Degree, VerbForm, Mood, Reflex.
# See coverage_diff.SLOT_FEATS_KEYS for the shared definition used by reports.

# Jablonskis XPOS is dot-separated. Unknown segments are intentionally ignored;
# existing UD FEATS values win and only missing mapped keys are repaired.
JABLONSKIS_XPOS_TO_UD = {
    # Case
    "V": ("Case", "Nom"),
    "K": ("Case", "Gen"),
    "N": ("Case", "Dat"),
    "G": ("Case", "Acc"),
    "Įn": ("Case", "Ins"),
    "Vt": ("Case", "Loc"),
    "Š": ("Case", "Voc"),
    "Il": ("Case", "Ill"),
    # Gender
    "vyr": ("Gender", "Masc"),
    "mot": ("Gender", "Fem"),
    "bevrd": ("Gender", "Neut"),
    "bev": ("Gender", "Neut"),
    # Number
    "vns": ("Number", "Sing"),
    "dgs": ("Number", "Plur"),
    "dvisk": ("Number", "Dual"),
    # Tense
    "es": ("Tense", "Pres"),
    "būt": ("Tense", "Past"),
    "būt-k": ("Tense", "Past"),
    "būt-d": ("Tense", "Past"),
    "būs": ("Tense", "Fut"),
    # Person
    "1": ("Person", "1"),
    "2": ("Person", "2"),
    "3": ("Person", "3"),
    # Mood
    "tiesiog": ("Mood", "Ind"),
    "tar": ("Mood", "Cnd"),
    "liep": ("Mood", "Imp"),
    # Voice
    "veik": ("Voice", "Act"),
    "neveik": ("Voice", "Pass"),
    "reik": ("Voice", "Nec"),
    # Definiteness / degree / polarity
    "įvardž": ("Definite", "Def"),
    "neįvardž": ("Definite", "Ind"),
    "nelygin": ("Degree", "Pos"),
    "teig": ("Polarity", "Pos"),
    "neig": ("Polarity", "Neg"),
    "savyb": ("Poss", "Yes"),
    # VerbForm
    "asm": ("VerbForm", "Fin"),
    "bndr": ("VerbForm", "Inf"),
    "dlv": ("VerbForm", "Part"),
    "pusd": ("VerbForm", "Conv"),
    "padlv": ("VerbForm", "Ger"),
    "būdn": ("VerbForm", "Part"),
    # Reflex
    "sngr": ("Reflex", "Yes"),
}

GENDER_VALUES = {"m": "Masc", "f": "Fem", "n": "Neut", "c": "Com"}
NUMBER_VALUES = {"s": "Sing", "p": "Plur", "d": "Dual"}
CASE_VALUES = {
    "n": "Nom",
    "g": "Gen",
    "d": "Dat",
    "a": "Acc",
    "i": "Ins",
    "l": "Loc",
    "v": "Voc",
    "x": "Ill",
}
DEGREE_VALUES = {"p": "Pos", "c": "Cmp", "s": "Sup", "d": "Dim"}
DEFINITE_VALUES = {"n": "Ind", "y": "Def"}
NUM_TYPE_VALUES = {"c": "Card", "o": "Ord", "m": "Mult", "l": "Sets"}
NUM_FORM_VALUES = {"d": "Digit", "r": "Roman", "l": "Word", "m": "Word"}
VERB_FORM_VALUES = {
    "m": "Fin",
    "i": "Inf",
    "p": "Part",
    "a": "Ger",
    "h": "Conv",
    "b": "Vadv",
}
TENSE_VALUES = {"p": "Pres", "a": "Past", "s": "Past", "q": "Past", "f": "Fut"}
PERSON_VALUES = {"1": "1", "2": "2", "3": "3"}
MOOD_VALUES = {"i": "Ind", "m": "Imp", "s": "Sub", "o": "Opt"}
VOICE_VALUES = {"a": "Act", "p": "Pass", "n": "Nec"}
NAME_TYPE_VALUES = {"g": "Geo", "f": "Giv", "s": "Sur"}

PRON_TYPE_LEMMAS = {
    "Dem": {
        "tas",
        "šis",
        "šitas",
        "anas",
        "toks",
        "šitoks",
        "šioks",
        "anoks",
        "tiek",
        "šitiek",
    },
    "Emp": {"pats"},
    "Ind": {
        "vienas",
        "kitas",
        "kitoks",
        "kažkas",
        "kažkuris",
        "kažkoks",
        "kažin kas",
        "bet kas",
        "bet kuris",
        "bet koks",
        "keletas",
        "keli",
    },
    "Int,Rel": {"kas", "kuris", "koks", "katras", "kelintas", "kiek", "keliese"},
    "Neg": {"niekas", "joks", "nė vienas", "nė koks", "nė kuris"},
    "Prs": {"aš", "tu", "jis", "ji", "mes", "jūs", "savęs", "savas", "tavas"},
    "Tot": {"visas", "kiekvienas", "abu", "abudu", "abi", "abidvi", "abeji"},
}


def parse_sources(value: str) -> list[str]:
    sources = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("at least one source is required")
    unknown = sorted(set(sources) - VALID_SOURCES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown source(s): {', '.join(unknown)}")
    return list(dict.fromkeys(sources))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run local/tagger-hf/fetch_corpora.py first"
        )


def normalized_text(row: dict) -> str:
    text = row.get("text") or " ".join(row["tokens"])
    return " ".join(str(text).casefold().split())


def xpos_ud_feats(
    xpos: str,
    kept_keys: Iterable[str] | None,
) -> dict[str, str]:
    allowed = None if kept_keys is None else set(kept_keys)
    feats: dict[str, str] = {}
    for segment in (part for part in xpos.split(".") if part):
        mapped = JABLONSKIS_XPOS_TO_UD.get(segment)
        if mapped is None:
            continue
        key, value = mapped
        if allowed is not None and key not in allowed:
            continue
        feats.setdefault(key, value)
    return feats


def misc_value(misc: str, key: str) -> str:
    prefix = f"{key}="
    for part in (item for item in misc.split("|") if item):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _char(value: str, position: int) -> str:
    index = position - 1
    if index < 0 or index >= len(value):
        return ""
    char = value[index]
    return "" if char == "-" else char


def _add_feat(
    feats: dict[str, str],
    key: str,
    value: str | None,
    allowed: set[str] | None,
) -> None:
    if not value:
        return
    if allowed is not None and key not in allowed:
        return
    feats.setdefault(key, value)


def pron_type_from_lemma(lemma: str) -> str | None:
    normalized = " ".join(lemma.casefold().split())
    if not normalized:
        return None
    for pron_type, lemmas in PRON_TYPE_LEMMAS.items():
        if normalized in lemmas:
            return pron_type
    if normalized.startswith(("kaž", "bet ")):
        return "Ind"
    if normalized.startswith("nie"):
        return "Neg"
    return None


def multext_ud_feats(
    multext: str,
    kept_keys: Iterable[str] | None = None,
    lemma: str = "",
) -> dict[str, str]:
    """Decode Lithuanian MULTEXT-East-style MSD positions into UD FEATS."""
    allowed = None if kept_keys is None else set(kept_keys)
    feats: dict[str, str] = {}
    if not multext:
        return feats

    category = multext[0]
    if category == "N":
        _add_feat(feats, "Gender", GENDER_VALUES.get(_char(multext, 3)), allowed)
        _add_feat(feats, "Number", NUMBER_VALUES.get(_char(multext, 4)), allowed)
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 5)), allowed)
        _add_feat(feats, "Reflex", "Yes" if _char(multext, 6) == "y" else None, allowed)
        _add_feat(feats, "NameType", NAME_TYPE_VALUES.get(_char(multext, 7)), allowed)
    elif category == "A":
        _add_feat(feats, "Degree", DEGREE_VALUES.get(_char(multext, 3)), allowed)
        _add_feat(feats, "Gender", GENDER_VALUES.get(_char(multext, 4)), allowed)
        _add_feat(feats, "Number", NUMBER_VALUES.get(_char(multext, 5)), allowed)
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 6)), allowed)
        _add_feat(feats, "Definite", DEFINITE_VALUES.get(_char(multext, 7)), allowed)
    elif category == "P":
        _add_feat(feats, "Gender", GENDER_VALUES.get(_char(multext, 3)), allowed)
        _add_feat(feats, "Number", NUMBER_VALUES.get(_char(multext, 4)), allowed)
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 5)), allowed)
        _add_feat(feats, "Definite", DEFINITE_VALUES.get(_char(multext, 6)), allowed)
        _add_feat(feats, "PronType", pron_type_from_lemma(lemma), allowed)
    elif category == "M":
        _add_feat(feats, "NumType", NUM_TYPE_VALUES.get(_char(multext, 2)), allowed)
        _add_feat(feats, "Gender", GENDER_VALUES.get(_char(multext, 3)), allowed)
        _add_feat(feats, "Number", NUMBER_VALUES.get(_char(multext, 4)), allowed)
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 5)), allowed)
        _add_feat(feats, "NumForm", NUM_FORM_VALUES.get(_char(multext, 6)), allowed)
        _add_feat(feats, "Definite", DEFINITE_VALUES.get(_char(multext, 7)), allowed)
    elif category == "V":
        _add_feat(feats, "VerbForm", VERB_FORM_VALUES.get(_char(multext, 3)), allowed)
        _add_feat(feats, "Tense", TENSE_VALUES.get(_char(multext, 4)), allowed)
        if _char(multext, 4) == "q":
            _add_feat(feats, "Aspect", "Iter", allowed)
        _add_feat(feats, "Person", PERSON_VALUES.get(_char(multext, 5)), allowed)
        _add_feat(feats, "Number", NUMBER_VALUES.get(_char(multext, 6)), allowed)
        _add_feat(feats, "Gender", GENDER_VALUES.get(_char(multext, 7)), allowed)
        _add_feat(feats, "Voice", VOICE_VALUES.get(_char(multext, 8)), allowed)
        polarity_char = _char(multext, 9)
        if polarity_char in {"n", "y"}:
            _add_feat(feats, "Polarity", "Neg" if polarity_char == "y" else "Pos", allowed)
        _add_feat(feats, "Definite", DEFINITE_VALUES.get(_char(multext, 10)), allowed)
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 11)), allowed)
        _add_feat(feats, "Reflex", "Yes" if _char(multext, 12) == "y" else None, allowed)
        _add_feat(feats, "Mood", MOOD_VALUES.get(_char(multext, 13)), allowed)
        _add_feat(feats, "Degree", DEGREE_VALUES.get(_char(multext, 14)), allowed)
    elif category == "R":
        _add_feat(feats, "Degree", DEGREE_VALUES.get(_char(multext, 3)), allowed)
    elif category == "S":
        _add_feat(feats, "Case", CASE_VALUES.get(_char(multext, 3)), allowed)
    elif category == "X":
        _add_feat(feats, "Foreign", "Yes" if _char(multext, 2) == "f" else None, allowed)
        _add_feat(feats, "Abbr", "Yes" if _char(multext, 2) == "r" else None, allowed)
    elif category == "Y":
        if _char(multext, 2) in {"a", "s"}:
            _add_feat(feats, "Abbr", "Yes", allowed)
    return feats


# The accent pipeline consumes VDU's label space, and Lithuanian traditional
# grammar (hence VDU) has no determiner category — everything is įvardis
# (PRON) — and no auxiliary category — all verbs are veiksmažodis (vksm.).
# ALKSNIS says DET/AUX where MATAS says PRON/VERB for the same words:
# contradictory supervision the model should not learn. The scoring
# projection merges both pairs anyway; the yra→būti lemma need is covered
# by the sidecar shim (see server.py lemma_for).
VDU_UPOS_NORMALIZATION = {"DET": "PRON", "AUX": "VERB"}


def label_from_parts(
    upos: str,
    raw_feats: str,
    xpos: str,
    misc: str,
    lemma: str,
    kept_keys: Iterable[str] | None,
    repair_from_xpos: bool,
    normalize_vdu: bool = False,
) -> str:
    if normalize_vdu:
        upos = VDU_UPOS_NORMALIZATION.get(upos, upos)
    feats = parse_feats(raw_feats)
    if repair_from_xpos:
        for key, value in xpos_ud_feats(xpos, kept_keys).items():
            feats.setdefault(key, value)
    for key, value in multext_ud_feats(
        misc_value(misc, "Multext"),
        kept_keys,
        lemma=lemma,
    ).items():
        feats.setdefault(key, value)
    feats = align_alksnis_conventions(upos, feats)
    feats = filter_feats_keys(feats, kept_keys)
    return combined_label(upos, feats_string(feats))


def align_alksnis_conventions(upos: str, feats: dict[str, str]) -> dict[str, str]:
    """Deterministic peripheral-key alignment with ALKSNIS (the gold eval
    conventions): every ADP carries AdpType=Prep there; NameType and
    Degree-on-verbs are not used. Exact-match UFeats punishes both over-
    and under-emission, so align both directions."""
    if upos == "ADP":
        feats.setdefault("AdpType", "Prep")
    feats.pop("NameType", None)
    if upos in ("VERB", "AUX"):
        feats.pop("Degree", None)
    return feats


def read_conllu(
    path: Path,
    sentence_prefix: str,
    kept_keys: Iterable[str] | None,
    repair_from_xpos: bool = False,
    normalize_vdu: bool = False,
    emit_lemma_scripts: bool = False,
) -> list[dict]:
    require_file(path)
    sentences: list[dict] = []
    text = ""
    tokens: list[str] = []
    lemmas: list[str] = []
    labels: list[str] = []
    lemma_scripts: list[str] = []
    raw_labels: list[str] = []

    def flush() -> None:
        nonlocal text, tokens, lemmas, labels, lemma_scripts, raw_labels
        if tokens:
            row = {
                "id": f"{sentence_prefix}-{len(sentences) + 1}",
                "text": text or " ".join(tokens),
                "tokens": tokens,
                "labels": labels,
                "_raw_labels": raw_labels,
            }
            if emit_lemma_scripts:
                row["lemmas"] = lemmas
                row["lemma_scripts"] = lemma_scripts
            sentences.append(row)
        text = ""
        tokens = []
        lemmas = []
        labels = []
        lemma_scripts = []
        raw_labels = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            flush()
            continue
        if line.startswith("# text = "):
            text = line[len("# text = ") :]
            continue
        if line.startswith("#"):
            continue

        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue
        tokens.append(columns[1])
        if emit_lemma_scripts:
            lemmas.append(columns[2])
            lemma_scripts.append(make_lemma_script(columns[1], columns[2]))
        raw_labels.append(combined_label(columns[3], columns[5]))
        labels.append(
            label_from_parts(
                upos=columns[3],
                raw_feats=columns[5],
                xpos=columns[4],
                misc=columns[9] if len(columns) > 9 else "",
                lemma=columns[2],
                kept_keys=kept_keys,
                repair_from_xpos=repair_from_xpos,
                normalize_vdu=normalize_vdu,
            )
        )

    flush()
    return sentences


def dedupe_sentences(rows: Iterable[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    deduped: list[dict] = []
    dropped = 0
    for row in rows:
        key = normalized_text(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def drop_leaks(rows: Iterable[dict], heldout_keys: set[str]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        if normalized_text(row) in heldout_keys:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            public_row = {
                key: value for key, value in row.items() if not key.startswith("_")
            }
            handle.write(json.dumps(public_row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_labels(path: Path, labels: Iterable[str]) -> None:
    label_list = sorted(set(labels))
    payload = {
        "labels": label_list,
        "label2id": {label: index for index, label in enumerate(label_list)},
        "id2label": {str(index): label for index, label in enumerate(label_list)},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_lemma_scripts(path: Path, scripts: Iterable[str]) -> list[str]:
    script_list = script_inventory(scripts)
    payload = {
        "lemma_scripts": script_list,
        "lemma_script2id": {
            script: index for index, script in enumerate(script_list)
        },
        "id2lemma_script": {
            str(index): script for index, script in enumerate(script_list)
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return script_list


def token_count(rows: Iterable[dict]) -> int:
    return sum(len(row["tokens"]) for row in rows)


def labels_in(rows: Iterable[dict]) -> list[str]:
    return [label for row in rows for label in row["labels"]]


def raw_labels_in(rows: Iterable[dict]) -> list[str]:
    return [label for row in rows for label in row["_raw_labels"]]


def lemma_scripts_in(rows: Iterable[dict]) -> list[str]:
    return [script for row in rows for script in row.get("lemma_scripts", [])]


def coverage_tokens(rows: Iterable[dict]) -> list[FeatsToken]:
    tokens: list[FeatsToken] = []
    for row in rows:
        for label in row["labels"]:
            upos, feats = split_label(label)
            tokens.append(FeatsToken(upos, feats))
    return tokens


def oov_label_stats(rows: Iterable[dict], train_labels: set[str]) -> tuple[int, int, float]:
    labels = labels_in(rows)
    oov = sum(1 for label in labels if label not in train_labels)
    total = len(labels)
    return oov, total, (oov / total if total else 0.0)


def print_split_stats(name: str, rows: list[dict]) -> None:
    print(f"{name}: {len(rows):,} sentences / {token_count(rows):,} tokens")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=parse_sources("matas,alksnis"),
        help="comma-separated training sources: matas,alksnis",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="source corpus cache directory",
    )
    parser.add_argument(
        "--matas-file",
        type=Path,
        help="MATAS CoNLL-U file; defaults to --raw-dir/MATAS3.conllu",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output dataset directory",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        help="deterministic smoke limit after shuffling",
    )
    parser.add_argument(
        "--feats-keys",
        choices=("slots", "all"),
        default="slots",
        help="FEATS keys kept in labels: scoring slots only or all keys",
    )
    parser.add_argument(
        "--repair-from-xpos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fill missing MATAS UD FEATS from Jablonskis XPOS",
    )
    parser.add_argument(
        "--normalize-vdu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize UPOS toward VDU conventions in all splits (DET->PRON; "
        "Lithuanian grammar and the VDU dictionary have no determiner category)",
    )
    parser.add_argument(
        "--emit-lemma-scripts",
        action="store_true",
        help="store per-token FORM→LEMMA edit-script labels and inventory",
    )
    parser.add_argument(
        "--lemma-top-n",
        type=int,
        default=25,
        help="top-N lemma scripts included in prep coverage summary",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.max_train_sentences is not None and args.max_train_sentences < 1:
        parser.error("--max-train-sentences must be positive")

    kept_keys = key_filter(args.feats_keys)

    alksnis_dev = read_conllu(
        args.raw_dir / ALKSNIS_FILES["dev"],
        "alksnis-dev",
        kept_keys=kept_keys,
        normalize_vdu=args.normalize_vdu,
        emit_lemma_scripts=args.emit_lemma_scripts,
    )
    alksnis_test = read_conllu(
        args.raw_dir / ALKSNIS_FILES["test"],
        "alksnis-test",
        kept_keys=kept_keys,
        normalize_vdu=args.normalize_vdu,
        emit_lemma_scripts=args.emit_lemma_scripts,
    )
    heldout_keys = {normalized_text(row) for row in alksnis_dev + alksnis_test}

    train_rows: list[dict] = []
    matas_deduped_dropped = 0
    if "matas" in args.sources:
        matas_path = args.matas_file if args.matas_file is not None else args.raw_dir / MATAS_FILE
        matas_rows = read_conllu(
            matas_path,
            "matas",
            kept_keys=kept_keys,
            repair_from_xpos=args.repair_from_xpos,
            normalize_vdu=args.normalize_vdu,
            emit_lemma_scripts=args.emit_lemma_scripts,
        )
        matas_rows, matas_deduped_dropped = dedupe_sentences(matas_rows)
        train_rows.extend(matas_rows)

    if "alksnis" in args.sources:
        train_rows.extend(
            read_conllu(
                args.raw_dir / ALKSNIS_FILES["train"],
                "alksnis-train",
                kept_keys=kept_keys,
                normalize_vdu=args.normalize_vdu,
                emit_lemma_scripts=args.emit_lemma_scripts,
            )
        )

    train_rows, leaked_dropped = drop_leaks(train_rows, heldout_keys)
    random.Random(args.seed).shuffle(train_rows)
    if args.max_train_sentences is not None:
        train_rows = train_rows[: args.max_train_sentences]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "dev.jsonl", alksnis_dev)
    write_jsonl(args.out / "test.jsonl", alksnis_test)

    all_raw_labels = (
        raw_labels_in(train_rows) + raw_labels_in(alksnis_dev) + raw_labels_in(alksnis_test)
    )
    all_labels = labels_in(train_rows) + labels_in(alksnis_dev) + labels_in(alksnis_test)
    write_labels(args.out / "labels.json", all_labels)
    all_lemma_scripts = (
        lemma_scripts_in(train_rows)
        + lemma_scripts_in(alksnis_dev)
        + lemma_scripts_in(alksnis_test)
    )
    if args.emit_lemma_scripts:
        lemma_inventory = write_lemma_scripts(args.out / "lemma_scripts.json", all_lemma_scripts)
    else:
        lemma_inventory = []

    train_label_set = set(labels_in(train_rows))
    dev_oov, dev_total, dev_rate = oov_label_stats(alksnis_dev, train_label_set)
    test_oov, test_total, test_rate = oov_label_stats(alksnis_test, train_label_set)

    print(f"sources: {','.join(args.sources)}")
    print(f"feats keys: {args.feats_keys}")
    print(f"repair from XPOS: {'on' if args.repair_from_xpos else 'off'}")
    print(f"VDU UPOS normalization: {'on' if args.normalize_vdu else 'off'}")
    if "matas" in args.sources:
        print(f"matas duplicate sentences dropped: {matas_deduped_dropped:,}")
    print(f"leakage guard dropped training sentences: {leaked_dropped:,}")
    print_split_stats("train", train_rows)
    print_split_stats("dev", alksnis_dev)
    print_split_stats("test", alksnis_test)
    print(f"label set before: {len(set(all_raw_labels)):,}")
    print(f"label set after: {len(set(all_labels)):,}")
    print(f"label inventory size: {len(set(all_labels)):,}")
    if args.emit_lemma_scripts:
        covered, total, rate = top_script_coverage(all_lemma_scripts, args.lemma_top_n)
        print(f"lemma script inventory size: {len(lemma_inventory):,}")
        print(
            f"top-{args.lemma_top_n} lemma script coverage: "
            f"{covered:,}/{total:,} ({rate:.2%})"
        )
        for script, count in Counter(all_lemma_scripts).most_common(args.lemma_top_n):
            print(f"  {script}: {count:,}")
    print(f"dev OOV-label rate: {dev_oov:,}/{dev_total:,} ({dev_rate:.2%})")
    print(f"test OOV-label rate: {test_oov:,}/{test_total:,} ({test_rate:.2%})")
    print(
        "post-prep FEATS coverage: train vs ALKSNIS dev "
        f"(|delta| >= {COVERAGE_DELTA:.0%}, kept keys)"
    )
    print(
        format_coverage_table(
            coverage_rows(
                coverage_tokens(train_rows),
                coverage_tokens(alksnis_dev),
                keys=kept_keys,
            ),
            "train",
            "alksnis-dev",
            min_delta=COVERAGE_DELTA,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
