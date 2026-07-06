# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub",
#   "sentencepiece",
#   "safetensors",
#   "torch",
#   "transformers<5",
# ]
# ///
"""Assemble and optionally upload the joint model Hugging Face release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import save_file


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

from joint_lib import ENCODER, safe_relative  # noqa: E402


DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "joint_v2_literary.best.pt"
DEFAULT_RELEASE_DIR = SCRIPT_DIR / "hf_release"
DEFAULT_REPO_ID = "alexbalandi/litlat-bert-lithuanian-accentuator"
ONNX_FILES = ("joint.onnx", "joint.int8.onnx", "joint.meta.json")
PARITY_GATES = {
    "fp32_pos_agreement": 0.995,
    "fp32_stress_agreement": 0.995,
    "int8_pos_agreement": 0.98,
    "int8_stress_agreement": 0.98,
}


CC_BY_SA_4_LICENSE = """Creative Commons Attribution-ShareAlike 4.0 International

This model is licensed under the Creative Commons Attribution-ShareAlike 4.0
International Public License (CC BY-SA 4.0).

You are free to:

- Share: copy and redistribute the material in any medium or format.
- Adapt: remix, transform, and build upon the material for any purpose.

Under the following terms:

- Attribution: give appropriate credit, provide a link to the license, and
  indicate if changes were made.
- ShareAlike: if you remix, transform, or build upon the material, distribute
  your contributions under the same license as the original.
- No additional restrictions: do not apply legal terms or technological
  measures that legally restrict others from doing anything the license permits.

Full legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode
SPDX-License-Identifier: CC-BY-SA-4.0
"""


MODEL_CARD_TEMPLATE = """---
language: lt
license: cc-by-sa-4.0
base_model: EMBEDDIA/litlat-bert
tags:
- lithuanian
- accentuation
- stress
- morphology
- token-classification
---

# litlat-bert-lithuanian-accentuator

Single-pass Lithuanian **accentuation + morphology** model: one encoder
(litlat-bert), one forward pass per sentence, two heads — per-token
morphological labels (804 traditional-grammar labels) and per-token
stress placement (choose the letter within the word + the accent mark:
grave/acute/circumflex, or "no stress" for unadapted foreign words).
156M parameters; ~1,660 tokens/s on a laptop RTX 3080 Ti, ~46 tokens/s
CPU int8.

Built by the open kirčiuoklė project:
https://github.com/alexbalandi/kirciuokle — architecture, training
recipes, evaluation methodology and the full experiment log are
documented there (docs/).

## Quality (measured, 2026-07)

| benchmark | metric | score |
|---|---|---|
| Kirčiuotų tekstų chrestomatija (hand-stressed gold, literary; never trained on) | token exact / position / sentence | **89.9% / 92.1% / 37.6%** |
| LRT news 37.7k tokens (audited silver reference) | token exact / position | **89.8% / 91.2%** |
| ALKSNIS gold test (morphology) | combined label / UPOS | **88.8% / 96.8%** |

Reference points on the same gold benchmark and extraction: the VDU
kirčiuoklė + UDPipe pipeline scores 95.8% token exact;
phonology_engine (LIEPA) 76.7%. Published thesis baselines on their own
extraction of the same book (indicative only): transformer 0.711,
VDU Kirčiuoklis 0.702 sequence accuracy.

## Training data & provenance

- Base training: 120k MATAS v3.0 sentences (CC BY 4.0, gold
  morphology) with stress PROJECTED from this project's own open
  accentuation dictionary (built from Wiktionary/kaikki CC BY-SA +
  published accentology rules + VLKK normative data; zero
  unadjudicated disagreements against its QA reference).
- Literary fine-tune: 220k tokens of public-domain Lithuanian classics
  (lt.wikisource; authors verified dead ≥70 years), labeled by a
  CALIBRATED consensus teacher (accept a token only where the
  agreement pattern of several independent systems measures ≥98%
  accurate on gold; achieved purity ≈99.5%). Transparency note: the
  teacher's voters include the VDU+UDPipe silver pipeline, so the
  literary fine-tune distills a consensus that contains VDU-derived
  signal; the base training does not.
- The gold benchmark was firewalled out of training (exact + 8-word
  shingle dedup; 159 sentences dropped).

## Files

- `model.safetensors` — torch weights ({SIZE_TORCH} MB). Rebuild the
  model with `joint_lib.py` (in this repo) + EMBEDDIA/litlat-bert
  tokenizer.
- `joint.onnx` / `joint.int8.onnx` — ONNX export ({SIZE_INT8} MB int8),
  inputs/outputs documented in `joint.meta.json`. Torch↔ONNX parity:
  fp32 ≥99.5%, int8 ≥98% token agreement.
- `labels.json` — the 804-label morphology inventory.

## Limitations

- Foreign/unadapted words should stay unaccented; the model does this
  for ~67% of such tokens (a known regression from 76% caused by the
  literary fine-tune — literary data contains no foreign words).
- Literary/poetic register trails the VDU reference by ~6pp (was ~9pp
  before the fine-tune); archaic orthography is out of scope.
- Sentences are processed independently (max 128 subwords per
  sentence; longer sentences are truncated by the reference serving
  code).
- The stress head only proposes linguistically valid (letter, mark)
  cells (audited validity mask); it cannot express marks the standard
  language forbids.

## License & attribution

CC BY-SA 4.0. Base model: EMBEDDIA/litlat-bert. Morphology training
data: MATAS v3.0 (CC BY 4.0, CLARIN-LT). Accent supervision:
this project's open dictionary (kaikki.org Wiktionary extraction
CC BY-SA; VLKK normative resources; published accentology per the
project docs) and public-domain literary texts.
"""


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def ensure_release_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    root = SCRIPT_DIR.resolve()
    resolved = path.resolve()
    if root not in (resolved, *resolved.parents):
        raise RuntimeError(f"refusing to prepare release folder outside {root}: {path}")
    keep = set(ONNX_FILES)
    for child in path.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_checkpoint(path: Path) -> dict:
    require_file(path, "checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"checkpoint does not look like a joint checkpoint: {path}")
    return payload


def write_safetensors(checkpoint: dict, output_path: Path) -> None:
    tensors = {
        str(key): value.detach().cpu().contiguous()
        for key, value in checkpoint["model_state"].items()
        if torch.is_tensor(value)
    }
    save_file(tensors, output_path, metadata={"format": "pt"})


def write_labels(checkpoint: dict, output_path: Path) -> None:
    labels = [str(label) for label in checkpoint["labels"]]
    payload = {
        "labels": labels,
        "label2id": {label: index for index, label in enumerate(labels)},
        "id2label": {str(index): label for index, label in enumerate(labels)},
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_tokenizer(output_dir: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENCODER, use_fast=True)
    tokenizer.save_pretrained(output_dir)


def measured_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1_000_000:.1f}"


def write_readme(output_dir: Path) -> None:
    torch_size = measured_mb(output_dir / "model.safetensors")
    int8_size = measured_mb(output_dir / "joint.int8.onnx")
    readme = MODEL_CARD_TEMPLATE.replace("{SIZE_TORCH}", torch_size).replace(
        "{SIZE_INT8}",
        int8_size,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def verify_onnx_files(release_dir: Path) -> None:
    for name in ONNX_FILES:
        require_file(release_dir / name, name)


def assemble_release(checkpoint_path: Path, release_dir: Path) -> None:
    verify_onnx_files(release_dir)
    ensure_release_dir(release_dir)
    verify_onnx_files(release_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    write_safetensors(checkpoint, release_dir / "model.safetensors")
    shutil.copy2(SCRIPT_DIR / "joint_lib.py", release_dir / "joint_lib.py")
    save_tokenizer(release_dir)
    write_labels(checkpoint, release_dir / "labels.json")
    (release_dir / "LICENSE").write_text(CC_BY_SA_4_LICENSE, encoding="utf-8")
    write_readme(release_dir)


def format_size(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size} B"


def print_release_listing(release_dir: Path) -> None:
    print(f"HF release dry-run listing: {safe_relative(release_dir)}")
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file()):
        rel = path.relative_to(release_dir).as_posix()
        print(f"  {rel}\t{format_size(path.stat().st_size)}")


def load_meta(release_dir: Path) -> dict:
    payload = json.loads((release_dir / "joint.meta.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("joint.meta.json must contain a JSON object")
    return payload


def assert_parity_passed(release_dir: Path) -> None:
    meta = load_meta(release_dir)
    parity = meta.get("parity")
    if not isinstance(parity, dict):
        raise RuntimeError("--upload requires joint.meta.json parity results")
    failures = []
    for key, gate in PARITY_GATES.items():
        value = parity.get(key)
        if not isinstance(value, int | float) or float(value) < gate:
            failures.append(f"{key}={value!r} < {gate:.3f}")
    if failures:
        raise RuntimeError("--upload blocked by parity gate: " + "; ".join(failures))


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'").strip('"')
        values[key.strip()] = value
    return values


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token = parse_env_file(REPO_ROOT / ".env").get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("--upload requires HF_TOKEN in the repo root .env or environment")
    return token


def upload_release(release_dir: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    assert_parity_passed(release_dir)
    api = HfApi(token=hf_token())
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(release_dir))
    return f"https://huggingface.co/{repo_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--upload", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    assemble_release(args.checkpoint, args.release_dir)
    print_release_listing(args.release_dir)
    if args.upload:
        url = upload_release(args.release_dir, args.repo_id)
        print(f"uploaded model: {url}")
    else:
        print("dry run only; use --upload after parity gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
