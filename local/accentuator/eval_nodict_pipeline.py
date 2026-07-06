"""Evaluate the no-dictionary accentuation pipeline on SPEC21 silver truth."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"

sys.path.insert(0, str(SCRIPT_DIR))
from _common import (  # noqa: E402
    DATA_DIR,
    DEFAULT_GENERATED,
    REPORTS_DIR,
    normalize_lt,
    safe_relative,
    strip_accents,
)

sys.path.append(str(LOCAL_DIR / "app"))
from kirciuokle import disambiguate as disamb  # noqa: E402
from train_guesser import stress_of  # noqa: E402

DEFAULT_CORPUS = DATA_DIR / "eval" / "lrt-smoke.txt"
DEFAULT_SILVER = DATA_DIR / "eval" / "lrt-smoke-silver.jsonl"
DEFAULT_REPORT = REPORTS_DIR / "nodict-eval.md"
DEFAULT_CKPT = DATA_DIR / "stress_nn2" / "stress_nn2.pt"
DEFAULT_AUDIT = DATA_DIR / "eval" / "lrt-silver-audit.json"
TAGGER_PRIMARY_MODEL = TAGGER_DIR / "release" / "hf-vdu"
TAGGER_FALLBACK_MODEL = TAGGER_DIR / "artifacts" / "litlat-gen2-onnx" / "int8"
TAGGER_READY_TEXT = "Vilnius yra grazus miestas."
UDPIPE_MODEL = "lithuanian-alksnis"
SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"')\]]+)?\s+")


@dataclass(frozen=True)
class SilverToken:
    word: str
    accented: str
    mi: str | None
    ambiguous: bool


@dataclass(frozen=True)
class DbEntry:
    variants: list[dict[str, Any]]
    default_form: str | None


@dataclass(frozen=True)
class LabelCandidate:
    label: str
    slots: dict[str, str]
    filled_slots: int


@dataclass(frozen=True)
class EvalRow:
    word: str
    silver: str
    token: Any
    label: str


@dataclass(frozen=True)
class Metrics:
    pipeline: str
    confidence: str
    total_tokens: int
    answered_tokens: int
    token_exact: int
    token_position: int
    total_types: int
    answered_types: int
    type_exact: int


@dataclass(frozen=True)
class AuditSummary:
    excluded_tokens: int
    foreign_unmarked_tokens: int
    foreign_unmarked_ok: int


def norm_form(text: str | None) -> str:
    return normalize_lt(text or "").lower()


def word_key(text: str | None) -> str:
    return strip_accents(normalize_lt(text or "")).lower()


def load_audit(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"audit file must be a JSON object: {path}")
    audit: dict[str, dict[str, Any]] = {}
    for raw_word, raw_entry in raw.items():
        if not isinstance(raw_entry, dict):
            continue
        word = word_key(str(raw_word))
        action = str(raw_entry.get("action") or "").strip()
        accept = {
            norm_form(str(form))
            for form in (raw_entry.get("accept") or [])
            if norm_form(str(form))
        }
        if word and action:
            audit[word] = {"action": action, "accept": accept}
    return audit


def observed_silver_forms(silver: Iterable[Any]) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    for token in silver:
        word = word_key(getattr(token, "word", ""))
        form = norm_form(getattr(token, "accented", ""))
        if word and form:
            observed.setdefault(word, set()).add(form)
    return observed


def audited_gold_forms(
    word: str,
    silver_form: str,
    audit: dict[str, dict[str, Any]],
    observed_forms: dict[str, set[str]],
) -> tuple[str, set[str]]:
    key = word_key(word)
    raw_gold = {norm_form(silver_form)}
    entry = audit.get(key)
    if entry is None:
        return "score", raw_gold

    action = entry["action"]
    accept = set(entry.get("accept") or set())
    if action == "replace":
        return "score", accept or raw_gold
    if action == "accept-also":
        return "score", raw_gold | accept
    if action == "accept-any-observed":
        return "score", set(observed_forms.get(key) or raw_gold)
    if action == "exclude":
        return "exclude", set()
    if action == "unmarked-ok":
        return "foreign-unmarked", set()
    return "score", raw_gold


def prediction_unmarked_or_abstained(word: str, predicted: str | None) -> bool:
    if predicted is None:
        return True
    if predicted == "":
        return True
    return word_key(predicted) == word_key(word) and stress_of(predicted) is None


def prediction_answer_form(word: str, predicted: str | None) -> str | None:
    if predicted is None:
        return None
    return word if predicted == "" else predicted


def has_letter(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in text)


def load_silver(path: Path) -> list[SilverToken]:
    tokens: list[SilverToken] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            word = word_key(raw.get("word"))
            accented = norm_form(raw.get("accented"))
            if not word or not accented:
                raise ValueError(f"bad silver row at {path}:{line_number}")
            tokens.append(
                SilverToken(
                    word=word,
                    accented=accented,
                    mi=raw.get("mi") or None,
                    ambiguous=bool(raw.get("ambiguous")),
                )
            )
    return tokens


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        start = 0
        for match in SENTENCE_END_RE.finditer(paragraph):
            piece = paragraph[start : match.end()].strip()
            if piece:
                sentences.append(piece)
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences or ([text.strip()] if text.strip() else [])


def endpoint_for(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/process") else f"{stripped}/process"


def base_url_for(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped[: -len("/process")] if stripped.endswith("/process") else stripped


def post_tagger(url: str, text: str, timeout: float = 180.0) -> str:
    payload = urllib.parse.urlencode(
        {
            "tokenizer": "",
            "tagger": "",
            "model": UDPIPE_MODEL,
            "data": text,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint_for(url),
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "nodict-eval/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    result = decoded.get("result")
    if not isinstance(result, str):
        raise RuntimeError(f"tagger response did not include result: {decoded!r}")
    return result


def parse_conllu(conllu: str) -> list[Any]:
    parser = getattr(disamb, "parse_conllu", None)
    if parser is not None:
        return parser(conllu)

    tokens: list[Any] = []
    token_type = getattr(disamb, "Token")
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue
        feats: dict[str, str] = {}
        if columns[5] != "_":
            for item in columns[5].split("|"):
                key, sep, value = item.partition("=")
                if sep:
                    feats[key] = value
        tokens.append(
            token_type(
                form=columns[1] or "",
                lemma=columns[2] or "",
                upos=columns[3] or "",
                xpos=columns[4] or "_",
                feats=feats,
            )
        )
    return tokens


def wait_for_tagger(
    url: str,
    timeout: float,
    proc: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"tagger exited early with status {proc.returncode}")
        try:
            conllu = post_tagger(url, TAGGER_READY_TEXT, timeout=20.0)
            tokens = parse_conllu(conllu)
            if tokens:
                first = tokens[0]
                print(
                    f"tagger ready: {first.form}/{first.upos} "
                    f"({len(tokens)} readiness tokens)"
                )
                return
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"tagger did not become ready within {timeout:.0f}s: {last_error}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_onnx(model_dir: Path) -> Path | None:
    for name in ("model_quantized.onnx", "model.onnx"):
        path = model_dir / name
        if path.exists():
            return path
    matches = sorted(model_dir.glob("*.onnx"))
    return matches[0] if matches else None


def pick_tagger_model() -> tuple[Path, Path]:
    primary_onnx = find_onnx(TAGGER_PRIMARY_MODEL)
    if primary_onnx is not None:
        return TAGGER_PRIMARY_MODEL, primary_onnx
    fallback_onnx = find_onnx(TAGGER_FALLBACK_MODEL)
    if fallback_onnx is not None:
        return TAGGER_FALLBACK_MODEL, fallback_onnx
    raise FileNotFoundError(
        f"no ONNX tagger found in {TAGGER_PRIMARY_MODEL} or {TAGGER_FALLBACK_MODEL}"
    )


def terminate_tagger(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        print(f"tagger subprocess already exited: pid={proc.pid} status={proc.returncode}")
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)
    print(f"tagger subprocess terminated: pid={proc.pid} status={proc.returncode}")


@contextmanager
def tagger_url(tagger_override: str | None, timeout: float) -> Iterable[str]:
    if tagger_override:
        url = base_url_for(tagger_override)
        print(f"using external tagger: {url}")
        wait_for_tagger(url, timeout)
        yield url
        return

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    model_dir, onnx_file = pick_tagger_model()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        str(TAGGER_DIR / "server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-dir",
        str(model_dir.resolve()),
        "--onnx-file",
        onnx_file.name,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(TAGGER_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    print(
        f"started tagger subprocess: pid={proc.pid} url={url} "
        f"model={safe_relative(model_dir)} onnx={onnx_file.name}"
    )
    try:
        wait_for_tagger(url, timeout, proc)
        yield url
    finally:
        terminate_tagger(proc)


def tag_corpus(url: str, corpus_text: str, timeout: float) -> list[Any]:
    sentences = split_sentences(corpus_text)
    tokens: list[Any] = []
    for index, sentence in enumerate(sentences, start=1):
        conllu = post_tagger(url, sentence, timeout=timeout)
        tokens.extend(parse_conllu(conllu))
        if index % 25 == 0:
            print(f"tagged {index}/{len(sentences)} sentences")
    print(f"tagged {len(sentences)} sentences -> {len(tokens):,} tagger tokens")
    return tokens


def align_tagger_tokens(
    silver: list[SilverToken],
    tagger_tokens: list[Any],
    window: int = 8,
) -> tuple[list[tuple[SilverToken, Any]], int, int]:
    tokens = [token for token in tagger_tokens if has_letter(token.form)]
    token_keys = [word_key(token.form) for token in tokens]
    aligned: list[tuple[SilverToken, Any]] = []
    skipped_silver = 0
    skipped_tagger = 0
    silver_index = 0
    token_index = 0

    while silver_index < len(silver):
        target = silver[silver_index].word
        if token_index < len(tokens) and token_keys[token_index] == target:
            aligned.append((silver[silver_index], tokens[token_index]))
            silver_index += 1
            token_index += 1
            continue

        found_token = None
        for lookahead in range(token_index + 1, min(len(tokens), token_index + window + 1)):
            if token_keys[lookahead] == target:
                found_token = lookahead
                break

        found_silver = None
        if token_index < len(tokens):
            current = token_keys[token_index]
            for lookahead in range(
                silver_index + 1, min(len(silver), silver_index + window + 1)
            ):
                if silver[lookahead].word == current:
                    found_silver = lookahead
                    break

        if found_token is not None and (
            found_silver is None or found_token - token_index <= found_silver - silver_index
        ):
            skipped_tagger += found_token - token_index
            token_index = found_token
            continue
        if found_silver is not None:
            skipped_silver += found_silver - silver_index
            silver_index = found_silver
            continue

        skipped_silver += 1
        silver_index += 1

    skipped_tagger += max(0, len(tokens) - token_index)
    return aligned, skipped_silver, skipped_tagger


def variant_labels(variant: dict[str, Any]) -> list[str]:
    raw_mi = variant.get("mi")
    labels: list[str] = []
    if isinstance(raw_mi, list):
        labels.extend(str(item).strip() for item in raw_mi if str(item).strip())
    elif raw_mi:
        labels.append(str(raw_mi).strip())
    if not labels and variant.get("info"):
        labels.append(str(variant["info"]).strip())
    return [label for label in labels if label]


def load_generated(
    path: Path,
    target_words: set[str],
) -> tuple[list[LabelCandidate], dict[str, DbEntry], dict[str, dict[str, str]]]:
    labels: set[str] = set()
    entries: dict[str, DbEntry] = {}
    db = sqlite3.connect(path)
    try:
        for word, variants_json, default_form in db.execute(
            "SELECT word, variants, default_form FROM words"
        ):
            try:
                variants = json.loads(variants_json or "[]")
            except json.JSONDecodeError:
                variants = []
            if word in target_words:
                entries[word] = DbEntry(variants=variants, default_form=default_form)
            for variant in variants:
                labels.update(variant_labels(variant))
    finally:
        db.close()

    slot_cache: dict[str, dict[str, str]] = {}
    candidates: list[LabelCandidate] = []
    for label in sorted(labels):
        slots = disamb.parse_mi(label)
        slot_cache[label] = slots
        candidates.append(LabelCandidate(label=label, slots=slots, filled_slots=len(slots)))
    return candidates, entries, slot_cache


def bridge_label(
    token: Any,
    candidates: list[LabelCandidate],
    cache: dict[tuple[tuple[str, str], ...], str],
) -> str:
    context_slots = disamb.token_tags(token)
    key = tuple(sorted((str(k), str(v)) for k, v in context_slots.items()))
    if key in cache:
        return cache[key]

    best_label = ""
    best_score: int | None = None
    best_spurious: int | None = None
    for candidate in candidates:
        score = disamb.score_tags(candidate.slots, context_slots)
        # ties: prefer the label with the FEWEST slots absent from context —
        # score_tags does not penalize a slot only one side fills, so a
        # spurious "aukšt." on a noun label ties with the clean label
        spurious = sum(1 for slot in candidate.slots if slot not in context_slots)
        if (
            best_score is None
            or score > best_score
            or (score == best_score and (best_spurious is None or spurious < best_spurious))
        ):
            best_label = candidate.label
            best_score = score
            best_spurious = spurious
    if best_score is None or best_score <= 0:
        best_label = ""
    cache[key] = best_label
    return best_label


def make_eval_rows(
    aligned: list[tuple[SilverToken, Any]],
    candidates: list[LabelCandidate],
) -> list[EvalRow]:
    cache: dict[tuple[tuple[str, str], ...], str] = {}
    return [
        EvalRow(
            word=silver.word,
            silver=silver.accented,
            token=token,
            label=bridge_label(token, candidates, cache),
        )
        for silver, token in aligned
    ]


def default_form(entry: DbEntry) -> str | None:
    if entry.default_form:
        return norm_form(entry.default_form)
    for variant in entry.variants:
        form = norm_form(variant.get("form"))
        if form:
            return form
    return None


def slots_for_label(label: str, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    if label not in cache:
        cache[label] = disamb.parse_mi(label)
    return cache[label]


def pick_dict_form(
    entry: DbEntry | None,
    bridged_label: str,
    slot_cache: dict[str, dict[str, str]],
) -> str | None:
    if entry is None:
        return None
    fallback = default_form(entry)
    if not bridged_label:
        return fallback

    context_slots = slots_for_label(bridged_label, slot_cache)
    best_form: str | None = None
    best_score: int | None = None
    best_filled = -1
    for variant in entry.variants:
        form = norm_form(variant.get("form"))
        if not form:
            continue
        labels = variant_labels(variant)
        if not labels:
            score = 0
            filled = 0
        else:
            scored = [
                (
                    disamb.score_tags(slots_for_label(label, slot_cache), context_slots),
                    len(slots_for_label(label, slot_cache)),
                )
                for label in labels
            ]
            score, filled = max(scored, key=lambda item: (item[0], item[1]))
        if best_score is None or score > best_score or (
            score == best_score and filled > best_filled
        ):
            best_score = score
            best_filled = filled
            best_form = form
    if best_score is not None and best_score > 0 and best_form:
        return best_form
    return fallback


class StressNN2:
    def __init__(self, checkpoint_path: Path = DEFAULT_CKPT) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from train_stress_nn import ENCODER, StressModel, batch_predict

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(ckpt.get("encoder", ENCODER))
        model = StressModel(
            AutoModel.from_pretrained(ckpt.get("encoder", ENCODER)),
            len(ckpt["char_vocab"]) + 2,
            no_stress=bool(ckpt.get("no_stress")),
        )
        model.load_state_dict(ckpt["state_dict"])
        model = model.to("cpu")
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.char_vocab = ckpt["char_vocab"]
        self.batch_predict = batch_predict

        from train_stress_nn import MAX_CHARS

        self.max_chars = MAX_CHARS

    def raw_predict(
        self,
        words: list[str],
        labels: list[str],
    ) -> list[tuple[str, float] | None]:
        out: list[tuple[str, float] | None] = [None] * len(words)
        positions = [index for index, word in enumerate(words) if len(word) <= self.max_chars]
        if not positions:
            return out
        eligible_words = [words[index] for index in positions]
        eligible_labels = [labels[index] for index in positions]
        preds = self.batch_predict(
            self.model,
            self.tokenizer,
            self.char_vocab,
            eligible_words,
            "cpu",
            labels=eligible_labels,
        )
        for index, pred in zip(positions, preds):
            if pred is not None:
                out[index] = (norm_form(pred[0]), float(pred[1]))
        return out


def threshold_predictions(
    raw_preds: list[tuple[str, float] | None],
    threshold: float,
) -> list[str | None]:
    return [pred[0] if pred is not None and pred[1] >= threshold else None for pred in raw_preds]


def liepa_predictions(words: list[str]) -> list[str | None]:
    from guess_uncovered import engine_accent
    from phonology_engine import PhonologyEngine

    pe = PhonologyEngine()
    cache: dict[str, str | None] = {}
    for word in sorted(set(words)):
        form = engine_accent(pe, word)
        cache[word] = norm_form(form) if form else None
    return [cache[word] for word in words]


def dict_predictions(
    rows: list[EvalRow],
    entries: dict[str, DbEntry],
    slot_cache: dict[str, dict[str, str]],
) -> list[str | None]:
    return [pick_dict_form(entries.get(row.word), row.label, slot_cache) for row in rows]


def exact_match(predicted: str | None, silver: str) -> bool:
    return predicted is not None and norm_form(predicted) == norm_form(silver)


def exact_match_any(predicted: str | None, gold_forms: set[str]) -> bool:
    return predicted is not None and norm_form(predicted) in gold_forms


def position_match(predicted: str | None, silver: str) -> bool:
    if predicted is None:
        return False
    predicted_stress = stress_of(predicted)
    silver_stress = stress_of(silver)
    return (
        predicted_stress is not None
        and silver_stress is not None
        and predicted_stress[0] == silver_stress[0]
    )


def position_match_any(predicted: str | None, gold_forms: set[str]) -> bool:
    return any(position_match(predicted, gold_form) for gold_form in gold_forms)


def _score_predictions(
    pipeline: str,
    confidence: str,
    rows: list[EvalRow],
    predictions: list[str | None],
    audit: dict[str, dict[str, Any]] | None = None,
    observed_forms: dict[str, set[str]] | None = None,
) -> tuple[Metrics, AuditSummary]:
    answered = exact = position = 0
    type_seen: set[str] = set()
    answered_types = type_exact = 0
    total_tokens = 0
    excluded_tokens = 0
    foreign_unmarked_tokens = 0
    foreign_unmarked_ok = 0
    for row, predicted in zip(rows, predictions):
        answer = prediction_answer_form(row.word, predicted)
        gold_forms = {norm_form(row.silver)}
        if audit:
            category, gold_forms = audited_gold_forms(
                row.word,
                row.silver,
                audit,
                observed_forms or {},
            )
            if category == "exclude":
                excluded_tokens += 1
                continue
            if category == "foreign-unmarked":
                foreign_unmarked_tokens += 1
                if prediction_unmarked_or_abstained(row.word, predicted):
                    foreign_unmarked_ok += 1
                continue
        total_tokens += 1
        if predicted is not None:
            answered += 1
        if exact_match_any(answer, gold_forms):
            exact += 1
        if position_match_any(answer, gold_forms):
            position += 1
        if row.word in type_seen:
            continue
        type_seen.add(row.word)
        if predicted is not None:
            answered_types += 1
        if exact_match_any(answer, gold_forms):
            type_exact += 1
    return (
        Metrics(
            pipeline=pipeline,
            confidence=confidence,
            total_tokens=total_tokens,
            answered_tokens=answered,
            token_exact=exact,
            token_position=position,
            total_types=len(type_seen),
            answered_types=answered_types,
            type_exact=type_exact,
        ),
        AuditSummary(
            excluded_tokens=excluded_tokens,
            foreign_unmarked_tokens=foreign_unmarked_tokens,
            foreign_unmarked_ok=foreign_unmarked_ok,
        ),
    )


def score_predictions(
    pipeline: str,
    confidence: str,
    rows: list[EvalRow],
    predictions: list[str | None],
) -> Metrics:
    metrics, _summary = _score_predictions(pipeline, confidence, rows, predictions)
    return metrics


def score_predictions_with_audit(
    pipeline: str,
    confidence: str,
    rows: list[EvalRow],
    predictions: list[str | None],
    audit: dict[str, dict[str, Any]],
    observed_forms: dict[str, set[str]],
) -> tuple[Metrics, AuditSummary]:
    return _score_predictions(
        pipeline,
        confidence,
        rows,
        predictions,
        audit=audit,
        observed_forms=observed_forms,
    )


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def count_pct(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,} ({pct(numerator, denominator)})"


def metric_rows(metrics: list[Metrics]) -> list[str]:
    lines = [
        "| pipeline | min confidence | answered | token exact | token position | type exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.pipeline,
                    item.confidence,
                    count_pct(item.answered_tokens, item.total_tokens),
                    count_pct(item.token_exact, item.answered_tokens),
                    count_pct(item.token_position, item.answered_tokens),
                    count_pct(item.type_exact, item.answered_types),
                ]
            )
            + " |"
        )
    return lines


def audit_summary_rows(metrics: list[Metrics], summaries: list[AuditSummary]) -> list[str]:
    lines = [
        "| pipeline | min confidence | excluded tokens | foreign-unmarked | desired unmarked/abstained |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item, summary in zip(metrics, summaries):
        lines.append(
            "| "
            + " | ".join(
                [
                    item.pipeline,
                    item.confidence,
                    f"{summary.excluded_tokens:,}",
                    f"{summary.foreign_unmarked_tokens:,}",
                    count_pct(
                        summary.foreign_unmarked_ok,
                        summary.foreign_unmarked_tokens,
                    ),
                ]
            )
            + " |"
        )
    return lines


def nodict_disagreements(
    rows: list[EvalRow],
    predictions: list[str | None],
    limit: int = 25,
) -> list[tuple[str, str, str, str]]:
    samples: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row, predicted in zip(rows, predictions):
        answer = prediction_answer_form(row.word, predicted)
        if exact_match(answer, row.silver):
            continue
        sample = (
            row.word,
            row.silver,
            answer if answer is not None else "(unanswered)",
            row.label or "(empty)",
        )
        if sample in seen:
            continue
        seen.add(sample)
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def format_report(
    corpus_path: Path,
    silver_path: Path,
    generated_path: Path,
    checkpoint_path: Path,
    raw_metrics: list[Metrics],
    audited_metrics: list[Metrics],
    audit_path: Path,
    audit_entry_count: int,
    audit_summaries: list[AuditSummary],
    total_silver: int,
    aligned_count: int,
    skipped_silver: int,
    skipped_tagger: int,
    label_count: int,
    disagreements: list[tuple[str, str, str, str]],
) -> str:
    skip_rate = skipped_silver / (total_silver or 1)
    lines = [
        "# No-Dictionary Pipeline Evaluation",
        "",
        "## Corpus",
        f"- corpus: `{safe_relative(corpus_path)}`",
        f"- silver: `{safe_relative(silver_path)}`",
        f"- generated DB: `{safe_relative(generated_path)}`",
        f"- stress checkpoint: `{safe_relative(checkpoint_path)}`",
        f"- silver tokens: {total_silver:,}",
        f"- aligned tokens: {aligned_count:,}",
        f"- skipped silver tokens: {skipped_silver:,} ({skip_rate:.2%})",
        f"- skipped tagger tokens: {skipped_tagger:,}",
        f"- label vocabulary: {label_count:,}",
        f"- audit overlay: `{safe_relative(audit_path)}` ({audit_entry_count:,} entries)",
        "",
        "## Pipelines (Raw Silver)",
        "",
        "Token exact and position are measured over answered tokens. Type exact is measured over answered first-seen word types.",
        "",
        *metric_rows(raw_metrics),
        "",
        "## Pipelines (Audited Silver)",
        "",
        *metric_rows(audited_metrics),
        "",
        "## Audit Diagnostics",
        "",
        *audit_summary_rows(audited_metrics, audit_summaries),
        "",
        "## Nodict Disagreements",
        "",
    ]
    if disagreements:
        lines.extend(
            [
                "| word | silver | nodict | label |",
                "| --- | --- | --- | --- |",
            ]
        )
        for word, silver, nodict, label in disagreements:
            lines.append(
                f"| {escape_cell(word)} | {escape_cell(silver)} | "
                f"{escape_cell(nodict)} | {escape_cell(label)} |"
            )
    else:
        lines.append("No nodict disagreements.")
    return "\n".join(lines) + "\n"


def print_metrics(metrics: list[Metrics]) -> None:
    for line in metric_rows(metrics):
        print(line)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tagger-url", help="Existing UDPipe-compatible tagger URL.")
    parser.add_argument("--tagger-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for path, label in (
        (args.corpus, "corpus"),
        (args.silver, "silver JSONL"),
        (args.generated, "generated DB"),
        (args.checkpoint, "stress checkpoint"),
    ):
        if not path.exists():
            parser.error(f"missing {label}: {path}")

    silver = load_silver(args.silver)
    audit = load_audit(args.audit)
    observed_forms = observed_silver_forms(silver)
    corpus_text = args.corpus.read_text(encoding="utf-8")
    print(f"silver tokens: {len(silver):,}")

    with tagger_url(args.tagger_url, args.tagger_timeout) as url:
        tagger_tokens = tag_corpus(url, corpus_text, args.request_timeout)

    aligned, skipped_silver, skipped_tagger = align_tagger_tokens(silver, tagger_tokens)
    skip_rate = skipped_silver / (len(silver) or 1)
    print(
        f"alignment skip rate: {skip_rate:.2%} "
        f"({skipped_silver:,}/{len(silver):,} silver tokens; "
        f"{skipped_tagger:,} tagger tokens skipped)"
    )
    if skip_rate >= 0.05:
        raise RuntimeError(f"alignment skip rate too high: {skip_rate:.2%}")

    target_words = {silver_token.word for silver_token, _token in aligned}
    print("loading generated.sqlite labels and target dictionary rows")
    candidates, entries, slot_cache = load_generated(args.generated, target_words)
    print(f"label vocabulary: {len(candidates):,}; dictionary rows loaded: {len(entries):,}")

    rows = make_eval_rows(aligned, candidates)
    words = [row.word for row in rows]
    labels = [row.label for row in rows]

    print(f"loading stress checkpoint on CPU: {safe_relative(args.checkpoint)}")
    stress_model = StressNN2(args.checkpoint)
    nodict_raw = stress_model.raw_predict(words, labels)
    uncond_raw = stress_model.raw_predict(words, [""] * len(words))

    nodict_0 = threshold_predictions(nodict_raw, 0.0)
    nodict_09 = threshold_predictions(nodict_raw, 0.9)
    nodict_uncond = threshold_predictions(uncond_raw, 0.0)

    print("running LIEPA per distinct word")
    liepa = liepa_predictions(words)
    dictionary = dict_predictions(rows, entries, slot_cache)

    raw_metrics = [
        score_predictions("nodict", "0", rows, nodict_0),
        score_predictions("nodict", "0.9", rows, nodict_09),
        score_predictions("nodict-uncond", "0", rows, nodict_uncond),
        score_predictions("liepa", "n/a", rows, liepa),
        score_predictions("dict", "n/a", rows, dictionary),
    ]
    audited_pairs = [
        score_predictions_with_audit("nodict", "0", rows, nodict_0, audit, observed_forms),
        score_predictions_with_audit("nodict", "0.9", rows, nodict_09, audit, observed_forms),
        score_predictions_with_audit("nodict-uncond", "0", rows, nodict_uncond, audit, observed_forms),
        score_predictions_with_audit("liepa", "n/a", rows, liepa, audit, observed_forms),
        score_predictions_with_audit("dict", "n/a", rows, dictionary, audit, observed_forms),
    ]
    audited_metrics = [metrics for metrics, _summary in audited_pairs]
    audit_summaries = [summary for _metrics, summary in audited_pairs]
    print("raw silver metrics:")
    print_metrics(raw_metrics)
    print("audited silver metrics:")
    print_metrics(audited_metrics)

    disagreements = nodict_disagreements(rows, nodict_0)
    report = format_report(
        corpus_path=args.corpus,
        silver_path=args.silver,
        generated_path=args.generated,
        checkpoint_path=args.checkpoint,
        raw_metrics=raw_metrics,
        audited_metrics=audited_metrics,
        audit_path=args.audit,
        audit_entry_count=len(audit),
        audit_summaries=audit_summaries,
        total_silver=len(silver),
        aligned_count=len(aligned),
        skipped_silver=skipped_silver,
        skipped_tagger=skipped_tagger,
        label_count=len(candidates),
        disagreements=disagreements,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(f"report written: {safe_relative(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
