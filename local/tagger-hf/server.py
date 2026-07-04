from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from head_config import load_head_config
from inference_utils import outputs_to_labels

try:
    from fastapi import FastAPI, Form
except ImportError:  # pragma: no cover - keeps --help dependency-light
    FastAPI = None

    def Form(default):  # type: ignore[no-redef]
        return default


WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
SENTENCE_BREAK_RE = re.compile(r"[.!?\n]+")


app = (
    FastAPI(title="HF ONNX UDPipe-compatible Lithuanian tagger")
    if FastAPI is not None
    else None
)


@dataclass(frozen=True)
class Word:
    form: str
    start: int
    end: int


@dataclass
class Runtime:
    tokenizer: object
    session: object
    head_config: dict
    input_names: set[str]
    output_names: list[str]
    max_length: int
    chunk_words: int


def find_onnx(model_dir: Path, requested: str | None) -> Path:
    if requested:
        path = model_dir / requested
        if not path.exists():
            raise FileNotFoundError(f"ONNX file not found: {path}")
        return path

    for name in ("model_quantized.onnx", "model.onnx"):
        path = model_dir / name
        if path.exists():
            return path
    matches = sorted(model_dir.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"no ONNX file found in {model_dir}")
    return matches[0]


@lru_cache(maxsize=1)
def runtime() -> Runtime:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    model_dir = Path(os.getenv("MODEL_DIR", "/model"))
    head_config = load_head_config(model_dir)
    onnx_file = find_onnx(model_dir, os.getenv("ONNX_FILE"))

    session_options = ort.SessionOptions()
    threads = os.getenv("ORT_INTRA_OP_THREADS")
    if threads:
        session_options.intra_op_num_threads = int(threads)

    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return Runtime(
        tokenizer=AutoTokenizer.from_pretrained(model_dir, use_fast=True),
        session=session,
        head_config=head_config,
        input_names={item.name for item in session.get_inputs()},
        output_names=[item.name for item in session.get_outputs()],
        max_length=int(os.getenv("MAX_LENGTH", str(head_config["max_length"]))),
        chunk_words=int(os.getenv("CHUNK_WORDS", "128")),
    )


def split_sentences(text: str) -> list[list[Word]]:
    words = [Word(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text)]
    if not words:
        return []

    sentences: list[list[Word]] = []
    current: list[Word] = []
    for index, word in enumerate(words):
        current.append(word)
        next_start = words[index + 1].start if index + 1 < len(words) else len(text)
        gap = text[word.end : next_start]
        if SENTENCE_BREAK_RE.search(gap):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def split_label(label: str) -> tuple[str, str]:
    if "|" not in label:
        return label or "X", "_"
    upos, feats = label.split("|", 1)
    return upos or "X", feats or "_"


def lemma_for(form: str, upos: str) -> str:
    lowered = form.lower()
    # The accent pipeline's yra→yrà exception keys on lemma "būti". MATAS
    # labels copular būti VERB (not AUX per UD), so accept both — the
    # competing reading (ỹra from "irti", to crumble) is vanishingly rare
    # in running text and stays user-overridable in the UI.
    if lowered == "yra" and upos in ("AUX", "VERB"):
        return "būti"
    return lowered


def predict_labels(words: list[str], rt: Runtime) -> list[str]:
    labels: list[str] = []
    for start in range(0, len(words), rt.chunk_words):
        chunk = words[start : start + rt.chunk_words]
        encoded = rt.tokenizer(
            chunk,
            is_split_into_words=True,
            truncation=True,
            max_length=rt.max_length,
            return_tensors="np",
        )
        ort_inputs = {
            key: value for key, value in encoded.items() if key in rt.input_names
        }
        values = rt.session.run(rt.output_names, ort_inputs)
        outputs = dict(zip(rt.output_names, values))
        labels.extend(
            outputs_to_labels(
                outputs=outputs,
                word_ids=encoded.word_ids(batch_index=0),
                word_count=len(chunk),
                head_config=rt.head_config,
            )
        )
    return labels


def to_conllu(text: str, sentences: list[list[Word]], labels: list[str]) -> str:
    lines: list[str] = []
    label_index = 0
    for sentence_index, sentence in enumerate(sentences, start=1):
        sentence_text = " ".join(word.form for word in sentence)
        lines.append(f"# sent_id = {sentence_index}")
        lines.append(f"# text = {sentence_text}")
        for token_index, word in enumerate(sentence, start=1):
            upos, feats = split_label(labels[label_index])
            label_index += 1
            columns = [
                str(token_index),
                word.form,
                lemma_for(word.form, upos),
                upos,
                "_",
                feats if feats else "_",
                "_",
                "_",
                "_",
                "_",
            ]
            lines.append("\t".join(columns))
        lines.append("")

    if not lines:
        return f"# sent_id = 1\n# text = {text}\n"
    return "\n".join(lines)


async def process_request(
    data: str,
    tokenizer: str = "",
    tagger: str = "",
    model: str = "lithuanian-alksnis",
) -> dict[str, str]:
    del tokenizer, tagger, model
    sentences = split_sentences(data)
    forms = [word.form for sentence in sentences for word in sentence]
    labels = predict_labels(forms, runtime()) if forms else []
    return {"result": to_conllu(data, sentences, labels)}


if app is not None:

    @app.post("/process")
    async def process(
        data: str = Form(...),
        tokenizer: str = Form(""),
        tagger: str = Form(""),
        model: str = Form("lithuanian-alksnis"),
    ) -> dict[str, str]:
        return await process_request(data, tokenizer, tagger, model)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/model"))
    parser.add_argument("--onnx-file", default=os.getenv("ONNX_FILE"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    if app is None:
        raise RuntimeError("fastapi is required to run the tagger server")

    os.environ["MODEL_DIR"] = args.model_dir
    if args.onnx_file:
        os.environ["ONNX_FILE"] = args.onnx_file

    import uvicorn

    uvicorn.run("server:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
