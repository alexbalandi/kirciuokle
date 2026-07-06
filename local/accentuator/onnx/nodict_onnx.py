# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "onnxruntime",
#   "tokenizers",
# ]
# ///
"""Torch-free ONNX no-dictionary accentuation pipeline."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FORBIDDEN_IMPORT_ROOTS = {"torch", "transformers"}


class ForbiddenImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        root = fullname.partition(".")[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            raise ImportError(
                f"{root!r} is intentionally blocked in nodict_onnx.py; "
                "use tokenizers.Tokenizer and ONNX Runtime only"
            )
        return None


def install_import_guard() -> None:
    present = sorted(root for root in FORBIDDEN_IMPORT_ROOTS if root in sys.modules)
    if present:
        raise RuntimeError(
            "nodict_onnx.py must run without torch/transformers already imported; "
            f"found: {', '.join(present)}"
        )
    if not any(isinstance(finder, ForbiddenImportBlocker) for finder in sys.meta_path):
        sys.meta_path.insert(0, ForbiddenImportBlocker())


install_import_guard()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"
APP_DIR = LOCAL_DIR / "app"

sys.path.insert(0, str(ACCENTUATOR_DIR))
sys.path.insert(0, str(TAGGER_DIR))
sys.path.insert(0, str(APP_DIR))

from _common import DEFAULT_GENERATED, normalize_lt, strip_accents  # noqa: E402
from head_config import load_head_config  # noqa: E402
from inference_utils import outputs_to_labels  # noqa: E402
from kirciuokle import disambiguate as disamb  # noqa: E402
from train_guesser import apply_stress, valid_target  # noqa: E402

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"')\]]+)?\s+")
DEFAULT_STRESS_MODEL = SCRIPT_DIR / "stress.int8.onnx"
DEFAULT_STRESS_META = SCRIPT_DIR / "stress.meta.json"
DEFAULT_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "lrt-smoke.txt"
DEFAULT_CHECKPOINT = ACCENTUATOR_DIR / "data" / "stress_nn2" / "stress_nn2.pt"
TAGGER_PRIMARY_MODEL = TAGGER_DIR / "release" / "hf-vdu"
TAGGER_FALLBACK_MODEL = TAGGER_DIR / "artifacts" / "litlat-gen2-onnx" / "int8"
TAGGER_SECONDARY_FALLBACK = TAGGER_DIR / "artifacts" / "litlat-v2-onnx" / "int8"
MARKS = ["̀", "́", "̃"]


@dataclass(frozen=True)
class WordSpan:
    form: str
    start: int
    end: int


@dataclass(frozen=True)
class LabelCandidate:
    label: str
    slots: dict[str, str]
    filled_slots: int


@dataclass(frozen=True)
class PipelineToken:
    original: str
    word: str
    tagger_label: str
    bridge_label: str
    prediction: str | None
    confidence: float
    output: str


@dataclass
class TaggerRuntime:
    model_dir: Path
    onnx_file: Path
    tokenizer: Tokenizer
    session: ort.InferenceSession
    head_config: dict
    input_names: set[str]
    output_names: list[str]
    max_length: int
    chunk_words: int

    @classmethod
    def load(cls, model_dir: Path, onnx_file: Path | None = None, chunk_words: int = 128) -> "TaggerRuntime":
        model_dir = model_dir.resolve()
        onnx_path = onnx_file or find_onnx(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"missing tagger tokenizer.json: {tokenizer_path}")
        head_config = load_head_config(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return cls(
            model_dir=model_dir,
            onnx_file=onnx_path,
            tokenizer=Tokenizer.from_file(str(tokenizer_path)),
            session=session,
            head_config=head_config,
            input_names={item.name for item in session.get_inputs()},
            output_names=[item.name for item in session.get_outputs()],
            max_length=int(head_config["max_length"]),
            chunk_words=chunk_words,
        )

    def predict_labels(self, words: list[str]) -> list[str]:
        labels: list[str] = []
        self.tokenizer.enable_truncation(max_length=self.max_length)
        for start in range(0, len(words), self.chunk_words):
            chunk = words[start : start + self.chunk_words]
            if not chunk:
                continue
            encoded = self.tokenizer.encode(chunk, is_pretokenized=True)
            inputs = {
                "input_ids": np.asarray([encoded.ids], dtype=np.int64),
                "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in self.input_names:
                inputs["token_type_ids"] = np.asarray([encoded.type_ids], dtype=np.int64)
            inputs = {key: value for key, value in inputs.items() if key in self.input_names}
            values = self.session.run(self.output_names, inputs)
            outputs = dict(zip(self.output_names, values))
            labels.extend(
                outputs_to_labels(
                    outputs=outputs,
                    word_ids=encoded.word_ids,
                    word_count=len(chunk),
                    head_config=self.head_config,
                )
            )
        return labels

    def predict_tokens(self, words: list[str]) -> list[disamb.Token]:
        return [
            token_from_label(word, label)
            for word, label in zip(words, self.predict_labels(words))
        ]


@dataclass
class StressRuntime:
    onnx_file: Path
    meta_file: Path
    tokenizer_file: Path
    tokenizer: Tokenizer
    session: ort.InferenceSession
    input_names: set[str]
    output_names: list[str]
    char_vocab: dict[str, int]
    marks: list[str]
    max_chars: int
    max_length: int
    no_stress: bool
    batch_size: int

    @classmethod
    def load(
        cls,
        onnx_file: Path,
        meta_file: Path,
        tokenizer_file: Path,
        max_length: int | None = None,
        batch_size: int = 64,
    ) -> "StressRuntime":
        if not onnx_file.exists():
            raise FileNotFoundError(f"missing stress ONNX: {onnx_file}; run export_stress_onnx.py first")
        if not meta_file.exists():
            raise FileNotFoundError(f"missing stress metadata: {meta_file}; run export_stress_onnx.py first")
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"missing stress tokenizer.json: {tokenizer_file}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        session = ort.InferenceSession(
            str(onnx_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        labeled = bool(meta.get("labeled"))
        return cls(
            onnx_file=onnx_file,
            meta_file=meta_file,
            tokenizer_file=tokenizer_file,
            tokenizer=Tokenizer.from_file(str(tokenizer_file)),
            session=session,
            input_names={item.name for item in session.get_inputs()},
            output_names=[item.name for item in session.get_outputs()],
            char_vocab={str(key): int(value) for key, value in meta["char_vocab"].items()},
            marks=[str(item) for item in meta.get("marks", MARKS)],
            max_chars=int(meta.get("max_chars", 30)),
            max_length=int(max_length or (48 if labeled else 24)),
            no_stress=bool(meta.get("no_stress")),
            batch_size=batch_size,
        )

    def predict(
        self,
        words: list[str],
        labels: list[str],
    ) -> list[tuple[str, float] | None]:
        out: list[tuple[str, float] | None] = [None] * len(words)
        positions = [
            index
            for index, word in enumerate(words)
            if 0 < len(word) <= self.max_chars and word.isalpha()
        ]
        if not positions:
            return out
        self.tokenizer.enable_truncation(max_length=self.max_length)
        for start in range(0, len(positions), self.batch_size):
            batch_positions = positions[start : start + self.batch_size]
            batch_words = [words[index] for index in batch_positions]
            batch_labels = [labels[index] for index in batch_positions]
            input_ids, attention_mask = self._encode_batch(batch_words, batch_labels)
            char_ids = self._char_ids(batch_words)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "char_ids": char_ids,
            }
            inputs = {key: value for key, value in inputs.items() if key in self.input_names}
            values = self.session.run(self.output_names, inputs)
            outputs = dict(zip(self.output_names, values))
            predictions = self._decode_logits(
                batch_words,
                outputs["logits"],
                outputs.get("no_stress_logits"),
            )
            for index, prediction in zip(batch_positions, predictions):
                out[index] = prediction
        return out

    def _encode_batch(self, words: list[str], labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = [
            self.tokenizer.encode(word, pair=label) if label else self.tokenizer.encode(word)
            for word, label in zip(words, labels)
        ]
        width = max(1, max(len(encoding.ids) for encoding in encodings))
        input_ids = np.zeros((len(encodings), width), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = np.asarray(encoding.ids, dtype=np.int64)
            attention_mask[row, :length] = np.asarray(encoding.attention_mask, dtype=np.int64)
        return input_ids, attention_mask

    def _char_ids(self, words: list[str]) -> np.ndarray:
        width = max(1, max(len(word) for word in words))
        char_ids = np.zeros((len(words), width), dtype=np.int64)
        for row, word in enumerate(words):
            for col, ch in enumerate(word):
                char_ids[row, col] = self.char_vocab.get(ch, 1)
        return char_ids

    def _decode_logits(
        self,
        words: list[str],
        logits: np.ndarray,
        no_stress_logits: np.ndarray | None,
    ) -> list[tuple[str, float] | None]:
        predictions: list[tuple[str, float] | None] = []
        for row, word in enumerate(words):
            mask = self._valid_mask(word, logits.shape[1])
            flat = np.where(mask, logits[row], -1e9).reshape(-1).astype(np.float32)
            no_stress_index = flat.shape[0]
            if no_stress_logits is not None:
                flat = np.concatenate(
                    [flat, np.asarray([float(no_stress_logits[row])], dtype=np.float32)]
                )
            elif not mask.any():
                predictions.append(None)
                continue
            best = int(flat.argmax())
            confidence = softmax_confidence(flat, best)
            if no_stress_logits is not None and best == no_stress_index:
                predictions.append(("", confidence))
                continue
            pos, mark_index = divmod(best, len(self.marks))
            if pos >= len(word) or not valid_target(word, pos, self.marks[mark_index]):
                predictions.append(None)
            else:
                predictions.append((normalize_lt(apply_stress(word, pos, self.marks[mark_index])), confidence))
        return predictions

    def _valid_mask(self, word: str, width: int) -> np.ndarray:
        mask = np.zeros((width, len(self.marks)), dtype=bool)
        for pos, _ch in enumerate(word[:width]):
            for mark_index, mark in enumerate(self.marks):
                mask[pos, mark_index] = valid_target(word, pos, mark)
        return mask


@dataclass
class NodictPipeline:
    tagger: TaggerRuntime
    stress: StressRuntime
    candidates: list[LabelCandidate]
    bridge_cache: dict[tuple[tuple[str, str], ...], str]
    threshold: float

    @classmethod
    def load(
        cls,
        tagger_model_dir: Path,
        stress_model: Path,
        stress_meta: Path,
        generated: Path,
        stress_tokenizer: Path | None = None,
        tagger_onnx: Path | None = None,
        threshold: float = 0.0,
        tagger_chunk_words: int = 128,
        stress_batch_size: int = 64,
    ) -> "NodictPipeline":
        tagger = TaggerRuntime.load(tagger_model_dir, tagger_onnx, chunk_words=tagger_chunk_words)
        stress_tokenizer = stress_tokenizer or find_stress_tokenizer(stress_meta, tagger.model_dir)
        stress = StressRuntime.load(
            stress_model,
            stress_meta,
            stress_tokenizer,
            batch_size=stress_batch_size,
        )
        candidates = load_label_candidates(generated)
        return cls(
            tagger=tagger,
            stress=stress,
            candidates=candidates,
            bridge_cache={},
            threshold=threshold,
        )

    def analyze(self, text: str) -> tuple[str, list[PipelineToken]]:
        spans = find_word_spans(text)
        if not spans:
            return text, []
        tagger_words = [span.form for span in spans]
        tagger_labels = self.tagger.predict_labels(tagger_words)
        tokens = [
            token_from_label(span.form, label)
            for span, label in zip(spans, tagger_labels)
        ]
        bridge_labels = [self.bridge_label(token) for token in tokens]
        stress_words = [word_key(span.form) for span in spans]
        raw_predictions = self.stress.predict(stress_words, bridge_labels)

        pieces: list[str] = []
        token_outputs: list[PipelineToken] = []
        cursor = 0
        for span, tagger_label, bridge_label, stress_word, raw_prediction in zip(
            spans,
            tagger_labels,
            bridge_labels,
            stress_words,
            raw_predictions,
        ):
            pieces.append(text[cursor : span.start])
            prediction: str | None = None
            confidence = 0.0
            if raw_prediction is not None:
                prediction, confidence = raw_prediction
            output = render_prediction(span.form, stress_word, prediction, confidence, self.threshold)
            pieces.append(output)
            cursor = span.end
            token_outputs.append(
                PipelineToken(
                    original=span.form,
                    word=stress_word,
                    tagger_label=tagger_label,
                    bridge_label=bridge_label,
                    prediction=prediction if confidence >= self.threshold else None,
                    confidence=confidence,
                    output=output,
                )
            )
        pieces.append(text[cursor:])
        return "".join(pieces), token_outputs

    def accent_text(self, text: str) -> str:
        accented, _tokens = self.analyze(text)
        return accented

    def bridge_label(self, token: disamb.Token) -> str:
        context_slots = disamb.token_tags(token)
        key = tuple(sorted((str(k), str(v)) for k, v in context_slots.items()))
        if key in self.bridge_cache:
            return self.bridge_cache[key]

        best_label = ""
        best_score: int | None = None
        best_spurious: int | None = None
        for candidate in self.candidates:
            score = disamb.score_tags(candidate.slots, context_slots)
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
        self.bridge_cache[key] = best_label
        return best_label


def find_onnx(model_dir: Path) -> Path:
    for name in ("model_quantized.onnx", "model.onnx"):
        path = model_dir / name
        if path.exists():
            return path
    matches = sorted(model_dir.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"no ONNX file found in {model_dir}")
    return matches[0]


def pick_tagger_model() -> Path:
    for model_dir in (TAGGER_PRIMARY_MODEL, TAGGER_FALLBACK_MODEL, TAGGER_SECONDARY_FALLBACK):
        if model_dir.exists():
            try:
                find_onnx(model_dir)
                return model_dir
            except FileNotFoundError:
                continue
    raise FileNotFoundError(
        f"no ONNX tagger found in {TAGGER_PRIMARY_MODEL}, {TAGGER_FALLBACK_MODEL}, "
        f"or {TAGGER_SECONDARY_FALLBACK}"
    )


def find_stress_tokenizer(stress_meta: Path, preferred_tagger_dir: Path) -> Path:
    meta = json.loads(stress_meta.read_text(encoding="utf-8"))
    encoder_id = str(meta.get("encoder") or meta.get("stress_tokenizer_id") or "")
    candidates = [
        preferred_tagger_dir,
        TAGGER_FALLBACK_MODEL,
        TAGGER_PRIMARY_MODEL,
        TAGGER_SECONDARY_FALLBACK,
    ]
    for model_dir in candidates:
        tokenizer_path = model_dir / "tokenizer.json"
        head_config_path = model_dir / "head_config.json"
        if not tokenizer_path.exists() or not head_config_path.exists():
            continue
        try:
            head_config = load_head_config(model_dir)
        except (OSError, ValueError, KeyError):
            continue
        if not encoder_id or str(head_config.get("base_model")) == encoder_id:
            return tokenizer_path
    raise FileNotFoundError(
        "could not locate a tokenizer.json for the stress encoder; pass --stress-tokenizer"
    )


def token_from_label(form: str, label: str) -> disamb.Token:
    upos, feats = split_label(label)
    lemma = "būti" if form.lower() == "yra" and upos in {"AUX", "VERB"} else form.lower()
    return disamb.Token(form=form, lemma=lemma, upos=upos, xpos="_", feats=feats)


def split_label(label: str) -> tuple[str, dict[str, str]]:
    if "|" not in label:
        return label or "X", {}
    upos, raw_feats = label.split("|", 1)
    feats: dict[str, str] = {}
    if raw_feats and raw_feats != "_":
        for item in raw_feats.split("|"):
            key, sep, value = item.partition("=")
            if sep:
                feats[key] = value
    return upos or "X", feats


def find_word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text)]


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


def word_key(text: str) -> str:
    return strip_accents(normalize_lt(text)).lower()


def render_prediction(
    original: str,
    plain_word: str,
    prediction: str | None,
    confidence: float,
    threshold: float,
) -> str:
    if prediction is None or confidence < threshold:
        return original
    if prediction == "":
        return disamb.match_case(plain_word, original)
    return disamb.match_case(prediction, original)


def prediction_answer(word: str, prediction: str | None) -> str | None:
    if prediction is None:
        return None
    return word if prediction == "" else normalize_lt(prediction).lower()


def softmax_confidence(values: np.ndarray, best: int) -> float:
    shifted = values.astype(np.float64) - float(values.max())
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0:
        return 0.0
    return float(exp[best] / total)


def variant_labels(variant: dict) -> list[str]:
    raw_mi = variant.get("mi")
    labels: list[str] = []
    if isinstance(raw_mi, list):
        labels.extend(str(item).strip() for item in raw_mi if str(item).strip())
    elif raw_mi:
        labels.append(str(raw_mi).strip())
    if not labels and variant.get("info"):
        labels.append(str(variant["info"]).strip())
    return [label for label in labels if label]


def load_label_candidates(generated: Path) -> list[LabelCandidate]:
    labels: set[str] = set()
    db = sqlite3.connect(generated)
    try:
        for (variants_json,) in db.execute("SELECT variants FROM words"):
            try:
                variants = json.loads(variants_json or "[]")
            except json.JSONDecodeError:
                variants = []
            for variant in variants:
                if isinstance(variant, dict):
                    labels.update(variant_labels(variant))
    finally:
        db.close()
    return [
        LabelCandidate(
            label=label,
            slots=disamb.parse_mi(label),
            filled_slots=len(disamb.parse_mi(label)),
        )
        for label in sorted(labels)
    ]


def run_bench(pipeline: NodictPipeline, token_count: int) -> tuple[int, float, float]:
    base = "Vilnius yra gražus miestas, o lietuviškas sakinys skamba aiškiai. "
    words_per_base = len(find_word_spans(base))
    repeats = max(1, (token_count + words_per_base - 1) // words_per_base)
    text = (base * repeats).strip()
    spans = find_word_spans(text)[:token_count]
    if spans:
        text = text[: spans[-1].end]
    pipeline.accent_text(text)
    start = time.perf_counter()
    _accented = pipeline.accent_text(text)
    elapsed = time.perf_counter() - start
    tokens = len(find_word_spans(text))
    return tokens, elapsed, tokens / elapsed if elapsed > 0 else 0.0


def run_verify(
    pipeline: NodictPipeline,
    corpus: Path,
    sentences: int,
    checkpoint: Path,
    reference_python: Path | None,
) -> tuple[int, int, float]:
    corpus_sentences = split_sentences(corpus.read_text(encoding="utf-8"))[:sentences]
    rows: list[dict[str, str]] = []
    ours: list[str | None] = []
    for sentence in corpus_sentences:
        _accented, tokens = pipeline.analyze(sentence)
        for token in tokens:
            rows.append({"word": token.word, "label": token.bridge_label})
            ours.append(prediction_answer(token.word, token.prediction))
    reference = reference_predictions(rows, checkpoint, reference_python)
    matches = sum(1 for left, right in zip(ours, reference) if left == right)
    total = len(ours)
    rate = matches / (total or 1)
    print(f"agreement vs eval_nodict_pipeline StressNN2: {matches}/{total} = {rate:.2%}")
    if rate < 0.98:
        raise RuntimeError(f"agreement below 98%: {rate:.2%}")
    return matches, total, rate


def reference_predictions(
    rows: list[dict[str, str]],
    checkpoint: Path,
    reference_python: Path | None,
) -> list[str | None]:
    code = r"""
import contextlib
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
sys.path.insert(0, str(repo / "local" / "accentuator"))

payload = json.load(sys.stdin)
words = [row["word"] for row in payload]
labels = [row["label"] for row in payload]

with contextlib.redirect_stdout(sys.stderr):
    from eval_nodict_pipeline import StressNN2, threshold_predictions
    model = StressNN2(checkpoint)
    raw = model.raw_predict(words, labels)
    predictions = threshold_predictions(raw, 0.0)

def answer(word, prediction):
    if prediction is None:
        return None
    return word if prediction == "" else prediction

json.dump([answer(word, pred) for word, pred in zip(words, predictions)], sys.stdout, ensure_ascii=False)
"""
    command: list[str]
    if reference_python is None:
        candidate = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"
        reference_python = candidate if candidate.exists() else Path(sys.executable)
    if reference_python.exists():
        command = [
            "uv",
            "run",
            "--python",
            str(reference_python),
            "python",
            "-c",
            code,
            str(REPO_ROOT),
            str(checkpoint),
        ]
    else:
        command = [
            "uv",
            "run",
            "python",
            "-c",
            code,
            str(REPO_ROOT),
            str(checkpoint),
        ]
    proc = subprocess.run(
        command,
        input=json.dumps(rows, ensure_ascii=False),
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "reference eval_nodict_pipeline subprocess failed:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def assert_no_forbidden_imports(print_status: bool = False) -> None:
    present = sorted(root for root in FORBIDDEN_IMPORT_ROOTS if root in sys.modules)
    if present:
        raise RuntimeError(f"forbidden imports present: {', '.join(present)}")
    if print_status:
        print("import check: torch/transformers absent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="sentence to accent")
    parser.add_argument("--tagger-model-dir", type=Path, default=None)
    parser.add_argument("--tagger-onnx", type=Path, default=None)
    parser.add_argument("--stress-model", type=Path, default=DEFAULT_STRESS_MODEL)
    parser.add_argument("--stress-meta", type=Path, default=DEFAULT_STRESS_META)
    parser.add_argument("--stress-tokenizer", type=Path, default=None)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-tokens", type=int, default=500)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-sentences", type=int, default=30)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-python", type=Path, default=None)
    parser.add_argument("--check-imports", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.text and not args.bench and not args.verify and not args.check_imports:
        parser.error("provide text, --bench, --verify, or --check-imports")

    if args.check_imports and not (args.text or args.bench or args.verify):
        assert_no_forbidden_imports(print_status=True)
        return 0

    tagger_model_dir = args.tagger_model_dir or pick_tagger_model()
    pipeline = NodictPipeline.load(
        tagger_model_dir=tagger_model_dir,
        tagger_onnx=args.tagger_onnx,
        stress_model=args.stress_model,
        stress_meta=args.stress_meta,
        stress_tokenizer=args.stress_tokenizer,
        generated=args.generated,
        threshold=args.threshold,
    )

    if args.text:
        print(pipeline.accent_text(" ".join(args.text)))
    if args.bench:
        tokens, elapsed, tokens_per_second = run_bench(pipeline, args.bench_tokens)
        print(
            f"bench: {tokens} tokens in {elapsed:.3f}s = "
            f"{tokens_per_second:.1f} tokens/s (CPU int8)"
        )
    if args.verify:
        run_verify(
            pipeline,
            corpus=args.corpus,
            sentences=args.verify_sentences,
            checkpoint=args.checkpoint,
            reference_python=args.reference_python,
        )
    assert_no_forbidden_imports(print_status=args.check_imports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
