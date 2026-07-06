# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "onnx",
#   "onnxruntime",
#   "torch",
#   "transformers",
# ]
# ///
"""Export the neural stress model to ONNX and dynamic INT8.

The exported graph exposes raw stress logits for the character-by-mark grid.
For v3 checkpoints it also exposes the learned no-stress logit as a second
output. Parity checks compare the production masked argmax decision against
Torch for the FP32 and INT8 ONNX files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent

sys.path.insert(0, str(ACCENTUATOR_DIR))

from _common import DEFAULT_GENERATED, strip_accents  # noqa: E402
from train_guesser import valid_target  # noqa: E402
from train_stress_nn import (  # noqa: E402
    ENCODER,
    MARKS,
    MAX_CHARS,
    StressModel,
    tokenize_words,
)

DEFAULT_CHECKPOINT = ACCENTUATOR_DIR / "data" / "stress_nn2" / "stress_nn2.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_FP32 = "stress.onnx"
DEFAULT_INT8 = "stress.int8.onnx"
DEFAULT_META = "stress.meta.json"
DEFAULT_OPSET = 17


@dataclass(frozen=True)
class LoadedStress:
    model: StressModel
    tokenizer: object
    checkpoint: dict
    char_vocab: dict[str, int]
    encoder_id: str
    labeled: bool
    no_stress: bool


class StressExportWrapper(nn.Module):
    def __init__(self, model: StressModel, no_stress: bool) -> None:
        super().__init__()
        self.model = model
        self.no_stress = no_stress

    def forward(self, input_ids, attention_mask, char_ids):  # noqa: ANN001
        if self.no_stress:
            return self.model(
                input_ids,
                attention_mask,
                char_ids,
                include_no_stress=True,
            )
        return self.model(input_ids, attention_mask, char_ids)


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"stress checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_stress(path: Path) -> LoadedStress:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    ckpt = load_checkpoint(path)
    char_vocab = {str(key): int(value) for key, value in ckpt["char_vocab"].items()}
    encoder_id = str(ckpt.get("encoder") or ENCODER)
    no_stress = bool(ckpt.get("no_stress"))
    labeled = bool(ckpt.get("labeled")) or no_stress

    config = AutoConfig.from_pretrained(encoder_id)
    encoder = AutoModel.from_config(config)
    model = StressModel(encoder, len(char_vocab) + 2, no_stress=no_stress)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(encoder_id, use_fast=True)

    return LoadedStress(
        model=model,
        tokenizer=tokenizer,
        checkpoint=ckpt,
        char_vocab=char_vocab,
        encoder_id=encoder_id,
        labeled=labeled,
        no_stress=no_stress,
    )


def char_tensor(words: list[str], char_vocab: dict[str, int]) -> torch.Tensor:
    width = max(1, min(MAX_CHARS, max(len(word) for word in words)))
    out = torch.zeros(len(words), width, dtype=torch.long)
    for row, word in enumerate(words):
        for col, ch in enumerate(word[:width]):
            out[row, col] = char_vocab.get(ch, 1)
    return out


def make_dummy_inputs(loaded: LoadedStress) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    words = ["pavyzdys", "miestas"]
    labels = ["dkt. vns. vard.", "dkt. vns. vard."] if loaded.labeled else None
    encoded = tokenize_words(loaded.tokenizer, words, labels)
    return (
        encoded["input_ids"].to(torch.long),
        encoded["attention_mask"].to(torch.long),
        char_tensor(words, loaded.char_vocab),
    )


def export_fp32(
    loaded: LoadedStress,
    output_path: Path,
    opset: int,
) -> list[str]:
    input_ids, attention_mask, char_ids = make_dummy_inputs(loaded)
    output_names = ["logits"]
    if loaded.no_stress:
        output_names.append("no_stress_logits")
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "subwords"},
        "attention_mask": {0: "batch", 1: "subwords"},
        "char_ids": {0: "batch", 1: "chars"},
        "logits": {0: "batch", 1: "chars"},
    }
    if loaded.no_stress:
        dynamic_axes["no_stress_logits"] = {0: "batch"}

    wrapper = StressExportWrapper(loaded.model, loaded.no_stress).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, char_ids),
            output_path,
            input_names=["input_ids", "attention_mask", "char_ids"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            external_data=False,
        )
    stale_external_data = output_path.with_name(f"{output_path.name}.data")
    if stale_external_data.exists():
        stale_external_data.unlink()
    return output_names


def encoder_weight_nodes(fp32_path: Path, encoder_layers: int) -> list[str] | None:
    if encoder_layers < 0:
        return None

    import onnx

    model = onnx.load(str(fp32_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    weighted_nodes: list[str] = []
    for node in model.graph.node:
        if node.op_type == "Range":
            break
        if node.op_type in {"MatMul", "Gemm"} and any(
            input_name in initializer_names for input_name in node.input
        ):
            weighted_nodes.append(node.name)
    return weighted_nodes[: encoder_layers * 6]


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    per_channel: bool,
    encoder_layers: int,
) -> dict[str, object]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    nodes_to_quantize = encoder_weight_nodes(fp32_path, encoder_layers)
    scope = (
        "all supported nodes"
        if nodes_to_quantize is None
        else f"{len(nodes_to_quantize)} encoder weight MatMuls/Gemms"
    )
    print(f"INT8 quantization scope: {scope}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes_to_quantize,
    )
    return {
        "encoder_layers": "all" if encoder_layers < 0 else encoder_layers,
        "nodes_to_quantize": None if nodes_to_quantize is None else nodes_to_quantize,
        "per_channel": per_channel,
        "weight_type": "QInt8",
    }


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


def sample_dictionary_rows(
    db_path: Path,
    count: int,
    labeled: bool,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    db = sqlite3.connect(db_path)
    try:
        for word, variants_json in db.execute("SELECT word, variants FROM words"):
            word = strip_accents(unicodedata.normalize("NFC", str(word))).lower()
            if not word.isalpha() or len(word) > MAX_CHARS:
                continue

            label = ""
            if labeled:
                labels: list[str] = []
                try:
                    variants = json.loads(variants_json or "[]")
                except json.JSONDecodeError:
                    variants = []
                for variant in variants:
                    if isinstance(variant, dict):
                        labels.extend(variant_labels(variant))
                labels = sorted(set(labels))
                if not labels:
                    continue
                label = rng.choice(labels)

            seen += 1
            item = (word, label)
            if len(reservoir) < count:
                reservoir.append(item)
            else:
                index = rng.randrange(seen)
                if index < count:
                    reservoir[index] = item
    finally:
        db.close()

    if len(reservoir) < count:
        raise RuntimeError(
            f"only found {len(reservoir)} eligible dictionary rows in {db_path}; "
            f"need {count}"
        )
    rng.shuffle(reservoir)
    return [word for word, _label in reservoir], [label for _word, label in reservoir]


def valid_mask(words: list[str], width: int) -> np.ndarray:
    mask = np.zeros((len(words), width, len(MARKS)), dtype=bool)
    for row, word in enumerate(words):
        for pos, _ch in enumerate(word[:width]):
            for mark_index, mark in enumerate(MARKS):
                mask[row, pos, mark_index] = valid_target(word, pos, mark)
    return mask


def torch_argmax(
    loaded: LoadedStress,
    words: list[str],
    labels: list[str],
    batch_size: int,
) -> list[int]:
    decisions: list[int] = []
    use_labels = loaded.labeled
    with torch.no_grad():
        for start in range(0, len(words), batch_size):
            chunk_words = words[start : start + batch_size]
            chunk_labels = labels[start : start + batch_size] if use_labels else None
            encoded = tokenize_words(loaded.tokenizer, chunk_words, chunk_labels)
            input_ids = encoded["input_ids"].to(torch.long)
            attention_mask = encoded["attention_mask"].to(torch.long)
            chars = char_tensor(chunk_words, loaded.char_vocab)
            if loaded.no_stress:
                logits, no_stress_logits = loaded.model(
                    input_ids,
                    attention_mask,
                    chars,
                    include_no_stress=True,
                )
            else:
                logits = loaded.model(input_ids, attention_mask, chars)
                no_stress_logits = None
            mask = torch.from_numpy(valid_mask(chunk_words, logits.size(1)))
            logits = logits.masked_fill(~mask, -1e9)
            flat = logits.flatten(1)
            if no_stress_logits is not None:
                flat = torch.cat([flat, no_stress_logits[:, None].float()], dim=1)
            decisions.extend(int(item) for item in flat.argmax(-1).tolist())
    return decisions


def numpy_argmax(
    logits: np.ndarray,
    words: list[str],
    no_stress_logits: np.ndarray | None,
) -> list[int]:
    mask = valid_mask(words, logits.shape[1])
    masked = np.where(mask, logits, -1e9)
    flat = masked.reshape(masked.shape[0], -1)
    if no_stress_logits is not None:
        flat = np.concatenate([flat, no_stress_logits.reshape(-1, 1).astype(np.float32)], axis=1)
    return [int(item) for item in flat.argmax(axis=1)]


def onnx_argmax(
    onnx_path: Path,
    loaded: LoadedStress,
    words: list[str],
    labels: list[str],
    batch_size: int,
) -> list[int]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [item.name for item in session.get_outputs()]
    input_names = {item.name for item in session.get_inputs()}
    decisions: list[int] = []
    use_labels = loaded.labeled
    for start in range(0, len(words), batch_size):
        chunk_words = words[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size] if use_labels else None
        encoded = tokenize_words(loaded.tokenizer, chunk_words, chunk_labels)
        chars = char_tensor(chunk_words, loaded.char_vocab)
        ort_inputs = {
            "input_ids": encoded["input_ids"].detach().cpu().numpy().astype(np.int64),
            "attention_mask": encoded["attention_mask"].detach().cpu().numpy().astype(np.int64),
            "char_ids": chars.detach().cpu().numpy().astype(np.int64),
        }
        ort_inputs = {key: value for key, value in ort_inputs.items() if key in input_names}
        values = session.run(output_names, ort_inputs)
        outputs = dict(zip(output_names, values))
        decisions.extend(
            numpy_argmax(
                outputs["logits"],
                chunk_words,
                outputs.get("no_stress_logits"),
            )
        )
    return decisions


def agreement(reference: list[int], candidate: list[int]) -> tuple[int, int, float]:
    if len(reference) != len(candidate):
        raise ValueError("decision lists have different lengths")
    matches = sum(1 for left, right in zip(reference, candidate) if left == right)
    total = len(reference)
    return matches, total, matches / (total or 1)


def run_parity(
    loaded: LoadedStress,
    generated: Path,
    fp32_path: Path,
    int8_path: Path,
    samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, float | int]:
    words, labels = sample_dictionary_rows(generated, samples, loaded.labeled, seed)
    torch_decisions = torch_argmax(loaded, words, labels, batch_size)
    fp32_decisions = onnx_argmax(fp32_path, loaded, words, labels, batch_size)
    int8_decisions = onnx_argmax(int8_path, loaded, words, labels, batch_size)

    fp32_matches, total, fp32_rate = agreement(torch_decisions, fp32_decisions)
    int8_matches, _total, int8_rate = agreement(torch_decisions, int8_decisions)
    print(
        f"parity fp32: {fp32_matches}/{total} = {fp32_rate:.2%} "
        f"(threshold 99.50%)"
    )
    print(
        f"parity int8: {int8_matches}/{total} = {int8_rate:.2%} "
        f"(threshold 98.00%)"
    )
    if fp32_rate < 0.995:
        raise RuntimeError(f"FP32 parity below gate: {fp32_rate:.2%}")
    if int8_rate < 0.98:
        raise RuntimeError(f"INT8 parity below gate: {int8_rate:.2%}")
    return {
        "samples": total,
        "fp32_matches": fp32_matches,
        "fp32_agreement": fp32_rate,
        "int8_matches": int8_matches,
        "int8_agreement": int8_rate,
        "seed": seed,
    }


def write_meta(
    path: Path,
    loaded: LoadedStress,
    checkpoint: Path,
    fp32_path: Path,
    int8_path: Path,
    opset: int,
    parity: dict[str, float | int] | None,
    quantization: dict[str, object],
) -> None:
    meta = {
        "checkpoint": safe_relative(checkpoint),
        "char_vocab": loaded.char_vocab,
        "encoder": loaded.encoder_id,
        "fp32_onnx": fp32_path.name,
        "int8_onnx": int8_path.name,
        "labeled": loaded.labeled,
        "marks": MARKS,
        "max_chars": MAX_CHARS,
        "no_stress": loaded.no_stress,
        "opset": opset,
        "parity": parity,
        "quantization": quantization,
        "stress_tokenizer_id": loaded.encoder_id,
    }
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parity-samples", type=int, default=200)
    parser.add_argument("--parity-seed", type=int, default=20260705)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument(
        "--quantized-encoder-layers",
        type=int,
        default=4,
        help=(
            "Number of lower encoder layers to dynamically quantize. "
            "Use -1 to quantize every supported node."
        ),
    )
    parser.add_argument("--skip-parity", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    torch.set_grad_enabled(False)
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / DEFAULT_FP32
    int8_path = args.output_dir / DEFAULT_INT8
    meta_path = args.output_dir / DEFAULT_META

    print(f"loading checkpoint on CPU: {safe_relative(args.checkpoint)}")
    loaded = load_stress(args.checkpoint)
    print(
        "checkpoint shape: "
        f"labeled={loaded.labeled} no_stress={loaded.no_stress} "
        f"chars={len(loaded.char_vocab)} encoder={loaded.encoder_id}"
    )

    print(f"exporting FP32 ONNX: {safe_relative(fp32_path)}")
    export_fp32(loaded, fp32_path, args.opset)
    print(f"quantizing INT8 ONNX: {safe_relative(int8_path)}")
    quantization = quantize_int8(
        fp32_path,
        int8_path,
        args.per_channel,
        args.quantized_encoder_layers,
    )

    parity = None
    if not args.skip_parity:
        parity = run_parity(
            loaded=loaded,
            generated=args.generated,
            fp32_path=fp32_path,
            int8_path=int8_path,
            samples=args.parity_samples,
            batch_size=args.batch_size,
            seed=args.parity_seed,
        )
    write_meta(
        meta_path,
        loaded=loaded,
        checkpoint=args.checkpoint,
        fp32_path=fp32_path,
        int8_path=int8_path,
        opset=args.opset,
        parity=parity,
        quantization=quantization,
    )
    print(f"metadata written: {safe_relative(meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
