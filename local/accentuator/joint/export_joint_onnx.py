# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "onnx",
#   "onnxscript",
#   "onnxruntime",
#   "sentencepiece",
#   "torch",
#   "transformers<5",
# ]
# ///
"""Export the joint POS + stress model to ONNX and dynamic INT8.

The exported graph exposes token-aligned outputs:
  pos_logits: (batch, tokens, labels)
  stress_logits: (batch, tokens, chars, marks)
  no_stress_logits: (batch, tokens)

Parity checks compare Torch vs ONNX FP32 vs ONNX INT8 token argmax decisions
for POS and stress separately on 100 LRT smoke sentences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ACCENTUATOR_DIR))

from eval_nodict_pipeline import split_sentences  # noqa: E402
from joint_lib import (  # noqa: E402
    ENCODER,
    MARKS,
    MAX_SUBWORDS,
    JointCollator,
    JointModel,
    rows_from_plain_sentences,
    safe_relative,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "hf_release"
DEFAULT_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "lrt-smoke.txt"
DEFAULT_FP32 = "joint.onnx"
DEFAULT_INT8 = "joint.int8.onnx"
DEFAULT_META = "joint.meta.json"
DEFAULT_OPSET = 17
DEFAULT_PARITY_SENTENCES = 100
FP32_GATE = 0.995
INT8_GATE = 0.98


@dataclass(frozen=True)
class LoadedJoint:
    model: JointModel
    tokenizer: object
    checkpoint: dict
    char_vocab: dict[str, int]
    labels: list[str]
    encoder_id: str


class JointExportWrapper(nn.Module):
    def __init__(self, model: JointModel) -> None:
        super().__init__()
        self.model = model

    def forward(  # noqa: PLR0913
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor,
        first_subword: torch.Tensor,
        last_subword: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        word_reps = gather_word_reps(hidden, first_subword)
        pos_logits = self.model.pos_head(word_reps)

        batch_size = char_ids.shape[0]
        token_count = char_ids.shape[1]
        char_count = char_ids.shape[2]
        sequence_length = hidden.shape[1]
        hidden_size = hidden.shape[2]

        positions = torch.arange(sequence_length, device=hidden.device)
        valid_word = (first_subword >= 0) & (last_subword >= first_subword)
        starts = first_subword.clamp(min=0)
        ends = last_subword.clamp(min=0)
        in_span = (
            (positions.view(1, 1, -1) >= starts.unsqueeze(-1))
            & (positions.view(1, 1, -1) <= ends.unsqueeze(-1))
            & valid_word.unsqueeze(-1)
            & attention_mask.to(torch.bool).unsqueeze(1)
        )
        fallback_span = (~valid_word).unsqueeze(-1) & (
            positions.view(1, 1, -1) == 0
        )
        subword_pad_mask = ~(in_span | fallback_span)

        subword_states = (
            hidden.unsqueeze(1)
            .expand(-1, token_count, -1, -1)
            .reshape(batch_size * token_count, sequence_length, hidden_size)
        )
        flat_char_ids = char_ids.reshape(batch_size * token_count, char_count)
        flat_char_mask = flat_char_ids != 0
        flat_pad_mask = subword_pad_mask.reshape(batch_size * token_count, sequence_length)

        stress_logits, no_stress_logits = self.stress_forward(
            flat_char_ids,
            subword_states,
            flat_pad_mask,
            flat_char_mask,
        )
        stress_logits = stress_logits.reshape(
            batch_size,
            token_count,
            char_count,
            len(MARKS),
        )
        no_stress_logits = no_stress_logits.reshape(batch_size, token_count)
        return pos_logits, stress_logits, no_stress_logits

    def stress_forward(
        self,
        char_ids: torch.Tensor,
        subword_states: torch.Tensor,
        subword_pad_mask: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.model.stress_head
        pos = torch.arange(char_ids.size(1), device=char_ids.device)
        q = head.q_norm(head.char_emb(char_ids) + head.pos_emb(pos)[None])
        attended = self.cross_attention(q, subword_states, subword_pad_mask)
        x = head.attn_norm(q + attended)
        x = head.ffn_norm(x + head.ffn(x))
        weights = char_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return head.out(x), head.no_stress(pooled).squeeze(-1)

    def cross_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn = self.model.stress_head.attn
        embed_dim = query.shape[-1]
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads
        q_weight, k_weight, v_weight = attn.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key_value, k_weight, k_bias)
        v = F.linear(key_value, v_weight, v_bias)

        batch_size = query.shape[0]
        query_length = query.shape[1]
        key_length = key_value.shape[1]
        q = q.reshape(batch_size, query_length, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, key_length, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, key_length, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (float(head_dim) ** -0.5)
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch_size, query_length, embed_dim)
        return attn.out_proj(context)


def gather_word_reps(hidden: torch.Tensor, first_subword: torch.Tensor) -> torch.Tensor:
    safe = first_subword.clamp(min=0, max=hidden.shape[1] - 1)
    expanded = safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return torch.gather(hidden, 1, expanded)


def load_joint(path: Path) -> LoadedJoint:
    from joint_lib import instantiate_from_checkpoint

    if not path.exists():
        raise FileNotFoundError(f"joint checkpoint not found: {path}")
    model, tokenizer, checkpoint = instantiate_from_checkpoint(path, device="cpu")
    model.eval()
    char_vocab = {str(key): int(value) for key, value in checkpoint["char_vocab"].items()}
    labels = [str(label) for label in checkpoint["labels"]]
    encoder_id = str(
        checkpoint.get("encoder_source")
        or checkpoint.get("base_model")
        or ENCODER
    )
    return LoadedJoint(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        char_vocab=char_vocab,
        labels=labels,
        encoder_id=encoder_id,
    )


def make_collator(loaded: LoadedJoint) -> JointCollator:
    return JointCollator(
        loaded.tokenizer,
        loaded.labels,
        loaded.char_vocab,
        max_chars=loaded.model.max_chars,
    )


def dummy_inputs(loaded: LoadedJoint) -> tuple[torch.Tensor, ...]:
    rows = rows_from_plain_sentences(
        [
            "Vilnius yra gražus miestas.",
            "Lietuvos rinktinė laimėjo rungtynes.",
        ]
    )
    batch = make_collator(loaded)(rows)
    return (
        batch["input_ids"].to(torch.long),
        batch["attention_mask"].to(torch.long),
        batch["char_ids"].to(torch.long),
        batch["first_subword"].to(torch.long),
        batch["last_subword"].to(torch.long),
    )


def export_fp32(loaded: LoadedJoint, output_path: Path, opset: int) -> None:
    wrapper = JointExportWrapper(loaded.model).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = dummy_inputs(loaded)
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "subwords"},
        "attention_mask": {0: "batch", 1: "subwords"},
        "char_ids": {0: "batch", 1: "tokens", 2: "chars"},
        "first_subword": {0: "batch", 1: "tokens"},
        "last_subword": {0: "batch", 1: "tokens"},
        "pos_logits": {0: "batch", 1: "tokens"},
        "stress_logits": {0: "batch", 1: "tokens", 2: "chars"},
        "no_stress_logits": {0: "batch", 1: "tokens"},
    }
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            output_path,
            input_names=[
                "input_ids",
                "attention_mask",
                "char_ids",
                "first_subword",
                "last_subword",
            ],
            output_names=["pos_logits", "stress_logits", "no_stress_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            external_data=False,
            dynamo=False,
        )
    stale_external_data = output_path.with_name(f"{output_path.name}.data")
    if stale_external_data.exists():
        stale_external_data.unlink()


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


def load_parity_rows(corpus: Path, sentence_count: int) -> list[dict]:
    if not corpus.exists():
        raise FileNotFoundError(f"parity corpus not found: {corpus}")
    sentences = split_sentences(corpus.read_text(encoding="utf-8"))
    if len(sentences) < sentence_count:
        raise RuntimeError(
            f"only found {len(sentences)} sentences in {corpus}; need {sentence_count}"
        )
    return rows_from_plain_sentences(sentences[:sentence_count])


def numpy_stress_argmax(
    stress_logits: np.ndarray,
    no_stress_logits: np.ndarray,
    char_valid: np.ndarray,
) -> np.ndarray:
    masked = np.where(char_valid, stress_logits, -1e9)
    flat = masked.reshape(masked.shape[0], masked.shape[1], -1)
    full = np.concatenate([flat, no_stress_logits[..., None].astype(np.float32)], axis=2)
    return full.argmax(axis=2)


def torch_decisions(
    wrapper: JointExportWrapper,
    batch: dict,
) -> tuple[list[int], list[int]]:
    inputs = (
        batch["input_ids"].to(torch.long),
        batch["attention_mask"].to(torch.long),
        batch["char_ids"].to(torch.long),
        batch["first_subword"].to(torch.long),
        batch["last_subword"].to(torch.long),
    )
    with torch.no_grad():
        pos_logits, stress_logits, no_stress_logits = wrapper(*inputs)
    char_valid = batch["char_valid"].to(torch.bool)
    masked = stress_logits.masked_fill(~char_valid, -1e9)
    flat = masked.flatten(2)
    stress_full = torch.cat([flat, no_stress_logits[..., None].float()], dim=2)
    word_mask = batch["word_mask"].to(torch.bool)
    pos = pos_logits.argmax(-1)[word_mask].detach().cpu().tolist()
    stress = stress_full.argmax(-1)[word_mask].detach().cpu().tolist()
    return [int(item) for item in pos], [int(item) for item in stress]


def onnx_decisions(
    session: object,
    batch: dict,
) -> tuple[list[int], list[int]]:
    output_names = [item.name for item in session.get_outputs()]
    input_names = {item.name for item in session.get_inputs()}
    ort_inputs = {
        "input_ids": batch["input_ids"].detach().cpu().numpy().astype(np.int64),
        "attention_mask": batch["attention_mask"].detach().cpu().numpy().astype(np.int64),
        "char_ids": batch["char_ids"].detach().cpu().numpy().astype(np.int64),
        "first_subword": batch["first_subword"].detach().cpu().numpy().astype(np.int64),
        "last_subword": batch["last_subword"].detach().cpu().numpy().astype(np.int64),
    }
    ort_inputs = {key: value for key, value in ort_inputs.items() if key in input_names}
    values = session.run(output_names, ort_inputs)
    outputs = dict(zip(output_names, values))
    word_mask = batch["word_mask"].detach().cpu().numpy().astype(bool)
    pos_ids = outputs["pos_logits"].argmax(axis=2)
    stress_ids = numpy_stress_argmax(
        outputs["stress_logits"],
        outputs["no_stress_logits"],
        batch["char_valid"].detach().cpu().numpy().astype(bool),
    )
    return (
        [int(item) for item in pos_ids[word_mask].tolist()],
        [int(item) for item in stress_ids[word_mask].tolist()],
    )


def make_onnx_session(path: Path) -> object:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    return ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def append_decisions(
    target: dict[str, list[int]],
    prefix: str,
    decisions: tuple[list[int], list[int]],
) -> None:
    pos, stress = decisions
    target[f"{prefix}_pos"].extend(pos)
    target[f"{prefix}_stress"].extend(stress)


def agreement(reference: list[int], candidate: list[int]) -> tuple[int, int, float]:
    if len(reference) != len(candidate):
        raise ValueError("decision lists have different lengths")
    matches = sum(1 for left, right in zip(reference, candidate) if left == right)
    total = len(reference)
    return matches, total, matches / (total or 1)


def print_metric(label: str, matches: int, total: int, rate: float, gate: float) -> None:
    print(f"parity {label}: {matches}/{total} = {rate:.2%} (threshold {gate:.2%})")


def run_parity(
    loaded: LoadedJoint,
    fp32_path: Path,
    int8_path: Path,
    corpus: Path,
    sentence_count: int,
    batch_size: int,
) -> dict[str, float | int]:
    rows = load_parity_rows(corpus, sentence_count)
    collator = make_collator(loaded)
    wrapper = JointExportWrapper(loaded.model).eval()
    fp32_session = make_onnx_session(fp32_path)
    int8_session = make_onnx_session(int8_path)
    decisions: dict[str, list[int]] = {
        "torch_pos": [],
        "torch_stress": [],
        "fp32_pos": [],
        "fp32_stress": [],
        "int8_pos": [],
        "int8_stress": [],
    }

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = collator(batch_rows)
        append_decisions(decisions, "torch", torch_decisions(wrapper, batch))
        append_decisions(decisions, "fp32", onnx_decisions(fp32_session, batch))
        append_decisions(decisions, "int8", onnx_decisions(int8_session, batch))

    fp32_pos_matches, pos_total, fp32_pos_rate = agreement(
        decisions["torch_pos"],
        decisions["fp32_pos"],
    )
    fp32_stress_matches, stress_total, fp32_stress_rate = agreement(
        decisions["torch_stress"],
        decisions["fp32_stress"],
    )
    int8_pos_matches, _pos_total, int8_pos_rate = agreement(
        decisions["torch_pos"],
        decisions["int8_pos"],
    )
    int8_stress_matches, _stress_total, int8_stress_rate = agreement(
        decisions["torch_stress"],
        decisions["int8_stress"],
    )
    print_metric("fp32 POS", fp32_pos_matches, pos_total, fp32_pos_rate, FP32_GATE)
    print_metric(
        "fp32 stress",
        fp32_stress_matches,
        stress_total,
        fp32_stress_rate,
        FP32_GATE,
    )
    print_metric("int8 POS", int8_pos_matches, pos_total, int8_pos_rate, INT8_GATE)
    print_metric(
        "int8 stress",
        int8_stress_matches,
        stress_total,
        int8_stress_rate,
        INT8_GATE,
    )
    if fp32_pos_rate < FP32_GATE:
        raise RuntimeError(f"FP32 POS parity below gate: {fp32_pos_rate:.2%}")
    if fp32_stress_rate < FP32_GATE:
        raise RuntimeError(f"FP32 stress parity below gate: {fp32_stress_rate:.2%}")
    if int8_pos_rate < INT8_GATE:
        raise RuntimeError(f"INT8 POS parity below gate: {int8_pos_rate:.2%}")
    if int8_stress_rate < INT8_GATE:
        raise RuntimeError(f"INT8 stress parity below gate: {int8_stress_rate:.2%}")
    return {
        "sentences": len(rows),
        "tokens": pos_total,
        "fp32_pos_matches": fp32_pos_matches,
        "fp32_pos_agreement": fp32_pos_rate,
        "fp32_stress_matches": fp32_stress_matches,
        "fp32_stress_agreement": fp32_stress_rate,
        "int8_pos_matches": int8_pos_matches,
        "int8_pos_agreement": int8_pos_rate,
        "int8_stress_matches": int8_stress_matches,
        "int8_stress_agreement": int8_stress_rate,
    }


def write_meta(
    path: Path,
    loaded: LoadedJoint,
    checkpoint: Path,
    fp32_path: Path,
    int8_path: Path,
    opset: int,
    parity: dict[str, float | int] | None,
    quantization: dict[str, object],
) -> None:
    meta = {
        "checkpoint": safe_relative(checkpoint),
        "encoder": loaded.encoder_id,
        "fp32_onnx": fp32_path.name,
        "int8_onnx": int8_path.name,
        "inputs": {
            "input_ids": "int64[batch, subwords]",
            "attention_mask": "int64[batch, subwords]",
            "char_ids": "int64[batch, tokens, chars]",
            "first_subword": "int64[batch, tokens]",
            "last_subword": "int64[batch, tokens]",
        },
        "outputs": {
            "pos_logits": "float32[batch, tokens, labels]",
            "stress_logits": "float32[batch, tokens, chars, marks]",
            "no_stress_logits": "float32[batch, tokens]",
        },
        "char_vocab": loaded.char_vocab,
        "labels": loaded.labels,
        "marks": MARKS,
        "max_chars": int(loaded.model.max_chars),
        "max_subwords": MAX_SUBWORDS,
        "no_stress": True,
        "opset": opset,
        "parity": parity,
        "quantization": quantization,
    }
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "joint_v2_literary.best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--parity-sentences", type=int, default=DEFAULT_PARITY_SENTENCES)
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
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / DEFAULT_FP32
    int8_path = args.output_dir / DEFAULT_INT8
    meta_path = args.output_dir / DEFAULT_META

    print(f"loading checkpoint on CPU: {safe_relative(args.checkpoint)}")
    loaded = load_joint(args.checkpoint)
    print(
        "checkpoint shape: "
        f"labels={len(loaded.labels)} chars={len(loaded.char_vocab)} "
        f"max_chars={loaded.model.max_chars} encoder={loaded.encoder_id}"
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
            fp32_path=fp32_path,
            int8_path=int8_path,
            corpus=args.corpus,
            sentence_count=args.parity_sentences,
            batch_size=args.batch_size,
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
