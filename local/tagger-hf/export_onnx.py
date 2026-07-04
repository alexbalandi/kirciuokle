# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Export a trained Lithuanian tagger to ONNX and dynamic INT8."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "labels.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)


def find_onnx(directory: Path) -> Path:
    preferred = ("model_quantized.onnx", "model.onnx")
    for name in preferred:
        path = directory / name
        if path.exists():
            return path
    matches = sorted(directory.glob("*.onnx"))
    if not matches:
        raise FileNotFoundError(f"no ONNX file found in {directory}")
    return matches[0]


def copy_runtime_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


def read_dev_rows(data_dir: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with (data_dir / "dev.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def verify_predictions(args: argparse.Namespace, onnx_file: Path) -> int:
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    torch_model = AutoModelForTokenClassification.from_pretrained(args.model_dir).eval()
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}

    mismatches = 0
    compared = 0
    for row in read_dev_rows(args.data_dir, args.max_dev_sentences):
        encoded = tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            torch_logits = torch_model(**encoded).logits.detach().cpu().numpy()

        ort_inputs = {
            key: value.detach().cpu().numpy()
            for key, value in encoded.items()
            if key in input_names
        }
        ort_logits = session.run(None, ort_inputs)[0]
        torch_pred = np.argmax(torch_logits, axis=-1)
        ort_pred = np.argmax(ort_logits, axis=-1)
        mask = encoded.get("attention_mask")
        if mask is None:
            active = np.ones(torch_pred.shape, dtype=bool)
        else:
            active = mask.detach().cpu().numpy().astype(bool)
        compared += int(active.sum())
        mismatches += int((torch_pred[active] != ort_pred[active]).sum())

    print(f"verified {compared} ONNX/Torch token decisions; mismatches={mismatches}")
    if mismatches > args.max_mismatches:
        raise RuntimeError(
            f"ONNX/Torch mismatch count {mismatches} exceeds "
            f"--max-mismatches {args.max_mismatches}"
        )
    return mismatches


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "modernbert-alksnis",
        help="trained HF model directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "alksnis",
        help="prepared ALKSNIS directory for verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "modernbert-alksnis-onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-dev-sentences", type=int, default=32)
    parser.add_argument("--max-mismatches", type=int, default=0)
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from optimum.exporters.onnx import main_export  # type: ignore[import-not-found]
    from optimum.onnxruntime import ORTQuantizer  # type: ignore[import-not-found]
    from optimum.onnxruntime.configuration import (  # type: ignore[import-not-found]
        AutoQuantizationConfig,
    )

    fp32_dir = args.output_dir / "fp32"
    int8_dir = args.output_dir / "int8"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=str(args.model_dir),
        output=fp32_dir,
        task="token-classification",
        opset=args.opset,
    )
    copy_runtime_files(args.model_dir, fp32_dir)
    fp32_onnx = find_onnx(fp32_dir)

    quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=fp32_onnx.name)
    quantization_config = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=args.per_channel,
    )
    quantizer.quantize(save_dir=int8_dir, quantization_config=quantization_config)
    copy_runtime_files(fp32_dir, int8_dir)

    int8_onnx = find_onnx(int8_dir)
    mismatches = verify_predictions(args, int8_onnx)
    (int8_dir / "onnx_meta.json").write_text(
        json.dumps(
            {
                "source_model": str(args.model_dir),
                "fp32_onnx": str(fp32_onnx),
                "int8_onnx": int8_onnx.name,
                "opset": args.opset,
                "verification_mismatches": mismatches,
                "verification_sentences": args.max_dev_sentences,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"INT8 model ready at {int8_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
