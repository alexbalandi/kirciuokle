# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Export a trained Lithuanian tagger to ONNX and dynamic INT8."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Iterable

from head_config import load_head_config, output_names_for_config
from inference_utils import outputs_to_labels


RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "head_config.json",
    "labels.json",
    "lemma_scripts.json",
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


def encoded_model_inputs(encoded: object, input_names: set[str]) -> dict:
    return {
        key: value
        for key, value in encoded.items()
        if key in input_names
    }


def tensor_to_numpy(value: object) -> object:
    return value.detach().cpu().numpy()


def run_torch_model(torch_runner: object, encoded: object, output_names: list[str]) -> dict:
    import torch  # type: ignore[import-not-found]

    with torch.no_grad():
        output = torch_runner(**dict(encoded))
    if hasattr(output, "logits"):
        output = output.logits
    if not isinstance(output, tuple):
        output = (output,)
    return {
        name: tensor_to_numpy(value)
        for name, value in zip(output_names, output)
    }


def load_torch_runner(model_dir: Path, head_config: dict) -> tuple[object, list[str]]:
    if (
        head_config["head"] == "combined"
        and head_config["pooling"] in {"first", "last"}
        and not head_config.get("lemma_scripts")
    ):
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        return model, ["logits"]

    from head_modeling import FullSequenceExportWrapper, load_custom_model

    model = load_custom_model(model_dir, head_config).eval()
    return FullSequenceExportWrapper(model).eval(), output_names_for_config(head_config)


def verify_predictions(
    args: argparse.Namespace,
    onnx_file: Path,
    head_config: dict,
) -> int:
    import onnxruntime as ort  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    torch_runner, torch_output_names = load_torch_runner(args.model_dir, head_config)
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    onnx_output_names = [item.name for item in session.get_outputs()]

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
        word_ids = list(encoded.word_ids(batch_index=0))
        torch_outputs = run_torch_model(torch_runner, encoded, torch_output_names)
        torch_labels = outputs_to_labels(
            outputs=torch_outputs,
            word_ids=word_ids,
            word_count=len(row["tokens"]),
            head_config=head_config,
        )

        ort_inputs = {
            key: value.detach().cpu().numpy()
            for key, value in encoded_model_inputs(encoded, input_names).items()
        }
        ort_values = session.run(onnx_output_names, ort_inputs)
        ort_outputs = dict(zip(onnx_output_names, ort_values))
        ort_labels = outputs_to_labels(
            outputs=ort_outputs,
            word_ids=word_ids,
            word_count=len(row["tokens"]),
            head_config=head_config,
        )

        for torch_label, ort_label in zip(torch_labels, ort_labels):
            compared += 1
            if torch_label != ort_label:
                mismatches += 1

    print(f"verified {compared} ONNX/Torch word labels; mismatches={mismatches}")
    if mismatches > args.max_mismatches:
        raise RuntimeError(
            f"ONNX/Torch mismatch count {mismatches} exceeds "
            f"--max-mismatches {args.max_mismatches}"
        )
    return mismatches


def export_with_optimum(args: argparse.Namespace, fp32_dir: Path) -> Path:
    from optimum.exporters.onnx import main_export  # type: ignore[import-not-found]

    main_export(
        model_name_or_path=str(args.model_dir),
        output=fp32_dir,
        task="token-classification",
        opset=args.opset,
    )
    copy_runtime_files(args.model_dir, fp32_dir)
    return find_onnx(fp32_dir)


def export_custom(args: argparse.Namespace, head_config: dict, fp32_dir: Path) -> Path:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    from head_modeling import FullSequenceExportWrapper, load_custom_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    encoded = tokenizer("pavyzdys", return_tensors="pt")
    output_names = output_names_for_config(head_config)
    model = load_custom_model(args.model_dir, head_config).eval()
    parameters = inspect.signature(model.encoder.forward).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    input_names = ["input_ids"]
    if "attention_mask" in encoded and ("attention_mask" in parameters or accepts_kwargs):
        input_names.append("attention_mask")
    if "token_type_ids" in encoded and ("token_type_ids" in parameters or accepts_kwargs):
        input_names.append("token_type_ids")
    dynamic_axes = {
        name: {0: "batch", 1: "sequence"}
        for name in input_names + output_names
    }
    wrapper = FullSequenceExportWrapper(model).eval()
    onnx_path = fp32_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        tuple(encoded[name] for name in input_names),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    copy_runtime_files(args.model_dir, fp32_dir)
    return onnx_path


def quantize_optimum(args: argparse.Namespace, fp32_dir: Path, int8_dir: Path, fp32_onnx: Path) -> Path:
    from optimum.onnxruntime import ORTQuantizer  # type: ignore[import-not-found]
    from optimum.onnxruntime.configuration import (  # type: ignore[import-not-found]
        AutoQuantizationConfig,
    )

    quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=fp32_onnx.name)
    quantization_config = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=args.per_channel,
    )
    quantizer.quantize(save_dir=int8_dir, quantization_config=quantization_config)
    copy_runtime_files(fp32_dir, int8_dir)
    return find_onnx(int8_dir)


def quantize_dynamic(args: argparse.Namespace, fp32_dir: Path, int8_dir: Path, fp32_onnx: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import-not-found]

    int8_onnx = int8_dir / "model_quantized.onnx"
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    copy_runtime_files(fp32_dir, int8_dir)
    return int8_onnx


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "lt-mlkm-modernbert__combined__first"
        / "best",
        help="trained HF model directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "combined",
        help="prepared corpus directory for verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "artifacts"
        / "lt-mlkm-modernbert-onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--max-length",
        type=int,
        help="verification token length; defaults to head_config.json",
    )
    parser.add_argument("--max-dev-sentences", type=int, default=32)
    parser.add_argument("--max-mismatches", type=int, default=0)
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    head_config = load_head_config(args.model_dir)
    args.max_length = args.max_length or head_config["max_length"]

    fp32_dir = args.output_dir / "fp32"
    int8_dir = args.output_dir / "int8"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    use_optimum = (
        head_config["head"] == "combined"
        and head_config["pooling"] in {"first", "last"}
        and not head_config.get("lemma_scripts")
    )
    if use_optimum:
        fp32_onnx = export_with_optimum(args, fp32_dir)
        int8_onnx = quantize_optimum(args, fp32_dir, int8_dir, fp32_onnx)
    else:
        fp32_onnx = export_custom(args, head_config, fp32_dir)
        int8_onnx = quantize_dynamic(args, fp32_dir, int8_dir, fp32_onnx)

    mismatches = verify_predictions(args, int8_onnx, head_config)
    (int8_dir / "onnx_meta.json").write_text(
        json.dumps(
            {
                "source_model": str(args.model_dir),
                "fp32_onnx": str(fp32_onnx),
                "int8_onnx": int8_onnx.name,
                "opset": args.opset,
                "head": head_config["head"],
                "lemma_head": bool(head_config.get("lemma_scripts")),
                "pooling": head_config["pooling"],
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
