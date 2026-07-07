# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "onnx",
#   "onnxruntime",
#   "onnxscript",
#   "torch",
#   "transformers<5",
# ]
# ///
"""Run the SPEC48 pruned-vocabulary gauntlet in order."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ACCENTUATOR_DIR))

from joint_lib import (  # noqa: E402
    WORD_RE,
    JointCollator,
    JointDataset,
    SimpleToken,
    batch_to_device,
    has_letter,
    instantiate_from_checkpoint,
    predict_batches,
    rows_from_plain_sentences,
    safe_relative,
    tokens_from_text_sentence,
)
import eval_chrestomatija as chrest  # noqa: E402
import eval_joint as joint_eval  # noqa: E402
import eval_nodict_pipeline as nodict  # noqa: E402


PRUNED_DIR = SCRIPT_DIR / "pruned"
DEFAULT_ORIGINAL_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "joint_v3.best.pt"
DEFAULT_PRUNED_CHECKPOINT = PRUNED_DIR / "joint_v3.pruned.pt"
DEFAULT_ONNX_DIR = PRUNED_DIR / "onnx"
DEFAULT_RESEGMENTED = PRUNED_DIR / "resegmented.txt"
DEFAULT_REPORT = PRUNED_DIR / "gauntlet_report.json"
LRT_SMOKE = ACCENTUATOR_DIR / "data" / "eval" / "lrt-smoke.txt"
LRT_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "lrt-corpus.txt"
LRT_SILVER = ACCENTUATOR_DIR / "data" / "eval" / "lrt-silver.jsonl"
LRT_AUDIT = ACCENTUATOR_DIR / "data" / "eval" / "lrt-silver-audit.json"
CHREST_PLAIN = ACCENTUATOR_DIR / "data" / "eval" / "chrestomatija-plain.txt"
WIKI_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "wikipedia-corpus.txt"
HF_RELEASE = SCRIPT_DIR / "hf_release"
OLD_BUNDLE = REPO_ROOT / "local-model"
FP32_NAME = "joint.onnx"
PARTIAL_INT8_NAME = "joint.int8.partial.onnx"
FULL_INT8_NAME = "joint.int8.full.onnx"
META_NAME = "joint.meta.json"
FP32_GATE = 0.995
INT8_GATE = 0.98
SEGMENTATION_GATE = 0.999
CHREST_GATE = 0.905
LRT_GATE = 0.916
ALKSNIS_GATE = 0.887
FOREIGN_GATE_DELTA = 0.01


def pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def compact_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): compact_dataclass(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_dataclass(item) for item in value]
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(compact_dataclass(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def choose_device() -> torch.device:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_allowed = cuda_visible is None or cuda_visible.strip() not in {"", "-1"}
    return torch.device("cuda" if cuda_allowed and torch.cuda.is_available() else "cpu")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenizer_segments(tokenizer: Any, words: list[str]) -> list[list[str]]:
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
    )
    ids = list(encoded["input_ids"])
    word_ids = list(encoded.word_ids())
    segments: list[list[str]] = [[] for _word in words]
    for token_id, word_id in zip(ids, word_ids):
        if word_id is None or word_id < 0 or word_id >= len(segments):
            continue
        segments[word_id].append(str(tokenizer.convert_ids_to_tokens(int(token_id))))
    return segments


def decision_rows(
    model: Any,
    tokenizer: Any,
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[list[dict[str, int | str | None]]]:
    collator = JointCollator(tokenizer, model.labels, checkpoint["char_vocab"])
    loader = DataLoader(
        JointDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    all_rows: list[list[dict[str, int | str | None]]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_rows = batch["rows"]
            moved = batch_to_device(batch, device)
            out = model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                token_type_ids=moved.get("token_type_ids"),
                first_subword=moved["first_subword"],
                last_subword=moved["last_subword"],
                word_mask=moved["word_mask"],
                char_ids=moved["char_ids"],
                char_valid=moved["char_valid"],
                char_mask=moved["char_mask"],
            )
            pos_ids = out["pos_logits"].argmax(-1).detach().cpu()
            stress_ids = out["stress_logits"].argmax(-1).detach().cpu()
            stress_positions = out["stress_word_positions"].detach().cpu().tolist()
            stress_by_position = {
                (int(batch_index), int(word_index)): int(stress_id)
                for (batch_index, word_index), stress_id in zip(
                    stress_positions,
                    stress_ids.tolist(),
                )
            }
            word_mask = moved["word_mask"].detach().cpu()
            for batch_index, row in enumerate(batch_rows):
                row_out = []
                for word_index, token in enumerate(row.get("tokens", [])):
                    predicted = (
                        word_index < word_mask.shape[1]
                        and bool(word_mask[batch_index, word_index])
                    )
                    row_out.append(
                        {
                            "word": str(token.get("word") or ""),
                            "pos": int(pos_ids[batch_index, word_index].item())
                            if predicted
                            else None,
                            "stress": stress_by_position.get((batch_index, word_index))
                            if predicted
                            else None,
                        }
                    )
                all_rows.append(row_out)
    return all_rows


def remap_sanity(
    original_checkpoint: Path,
    pruned_checkpoint: Path,
    batch_size: int,
) -> dict[str, Any]:
    print("GATE 1: remap sanity")
    device = choose_device()
    sentences = nodict.split_sentences(LRT_SMOKE.read_text(encoding="utf-8"))[:50]
    rows = rows_from_plain_sentences(sentences)

    original_model, original_tokenizer, original_payload = instantiate_from_checkpoint(
        original_checkpoint,
        device=device,
    )
    pruned_model, pruned_tokenizer, pruned_payload = instantiate_from_checkpoint(
        pruned_checkpoint,
        device=device,
    )
    original_decisions = decision_rows(
        original_model,
        original_tokenizer,
        original_payload,
        rows,
        device,
        batch_size,
    )
    pruned_decisions = decision_rows(
        pruned_model,
        pruned_tokenizer,
        pruned_payload,
        rows,
        device,
        batch_size,
    )

    total = pos_matches = stress_matches = 0
    mismatches: list[dict[str, Any]] = []
    identical_segment_tokens = 0
    for row, orig_row, pruned_row in zip(rows, original_decisions, pruned_decisions):
        words = [str(token["word"]) for token in row.get("tokens", [])]
        orig_segments = tokenizer_segments(original_tokenizer, words)
        pruned_segments = tokenizer_segments(pruned_tokenizer, words)
        for index, word in enumerate(words):
            if orig_segments[index] != pruned_segments[index]:
                continue
            identical_segment_tokens += 1
            left = orig_row[index]
            right = pruned_row[index]
            if left["pos"] is None or right["pos"] is None:
                continue
            total += 1
            pos_ok = left["pos"] == right["pos"]
            stress_ok = left["stress"] == right["stress"]
            pos_matches += int(pos_ok)
            stress_matches += int(stress_ok)
            if not pos_ok or not stress_ok:
                mismatches.append(
                    {
                        "row": row.get("id"),
                        "word": word,
                        "original": left,
                        "pruned": right,
                        "segments": orig_segments[index],
                    }
                )

    cleanup_cuda()
    if mismatches:
        raise RuntimeError(f"remap sanity mismatch: {mismatches[:5]}")
    if total == 0:
        raise RuntimeError("remap sanity had no comparable tokens")
    result = {
        "sentences": len(rows),
        "identical_segment_tokens": identical_segment_tokens,
        "compared_tokens": total,
        "pos_matches": pos_matches,
        "stress_matches": stress_matches,
        "pos_rate": pos_matches / total,
        "stress_rate": stress_matches / total,
    }
    print(
        "  remap sanity: "
        f"POS {pos_matches}/{total}={pct(result['pos_rate'])}; "
        f"stress {stress_matches}/{total}={pct(result['stress_rate'])}"
    )
    return result


def is_census_word(token: str) -> bool:
    return any(ch.isalnum() or unicodedata.category(ch).startswith("L") for ch in token)


def word_tokens_from_text(path: Path, limit: int | None = None) -> Iterable[str]:
    count = 0
    for match in WORD_RE.finditer(path.read_text(encoding="utf-8")):
        token = match.group(0)
        if not is_census_word(token):
            continue
        yield token
        count += 1
        if limit is not None and count >= limit:
            return


def token_piece_segment(tokenizer: Any, token: str) -> tuple[str, ...]:
    ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    return tuple(str(item) for item in tokenizer.convert_ids_to_tokens(ids))


def segmentation_census(
    original_checkpoint: Path,
    pruned_checkpoint: Path,
    resegmented_path: Path,
) -> tuple[dict[str, Any], Counter[str]]:
    print("GATE 2: segmentation census")
    _original_model, original_tokenizer, _original_payload = instantiate_from_checkpoint(
        original_checkpoint,
        device="cpu",
    )
    _pruned_model, pruned_tokenizer, _pruned_payload = instantiate_from_checkpoint(
        pruned_checkpoint,
        device="cpu",
    )
    original_cache: dict[str, tuple[str, ...]] = {}
    pruned_cache: dict[str, tuple[str, ...]] = {}
    resegmented: Counter[str] = Counter()
    examples: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    total = identical = 0
    sources = [
        ("lrt", word_tokens_from_text(LRT_CORPUS)),
        ("chrestomatija", word_tokens_from_text(CHREST_PLAIN)),
        ("wiki-20k", word_tokens_from_text(WIKI_CORPUS, limit=20_000)),
    ]
    source_totals: dict[str, dict[str, int]] = {}
    for source_name, tokens in sources:
        source_total = source_identical = 0
        for token in tokens:
            total += 1
            source_total += 1
            original = original_cache.setdefault(
                token,
                token_piece_segment(original_tokenizer, token),
            )
            pruned = pruned_cache.setdefault(
                token,
                token_piece_segment(pruned_tokenizer, token),
            )
            if original == pruned:
                identical += 1
                source_identical += 1
            else:
                resegmented[token] += 1
                examples.setdefault(token, (original, pruned))
        source_totals[source_name] = {
            "tokens": source_total,
            "identical": source_identical,
            "resegmented": source_total - source_identical,
        }

    resegmented_path.parent.mkdir(parents=True, exist_ok=True)
    with resegmented_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"# total={total} identical={identical} "
            f"rate={identical / (total or 1):.8f}\n"
        )
        handle.write("token\tcount\toriginal_pieces\tpruned_pieces\n")
        for token, count in resegmented.most_common():
            original, pruned = examples[token]
            handle.write(
                f"{token}\t{count}\t"
                f"{json.dumps(original, ensure_ascii=False)}\t"
                f"{json.dumps(pruned, ensure_ascii=False)}\n"
            )

    rate = identical / (total or 1)
    if rate < SEGMENTATION_GATE:
        raise RuntimeError(f"segmentation identity below gate: {pct(rate)}")
    result = {
        "total_tokens": total,
        "identical_tokens": identical,
        "resegmented_tokens": total - identical,
        "identity_rate": rate,
        "unique_resegmented": len(resegmented),
        "sources": source_totals,
        "path": safe_relative(resegmented_path),
    }
    print(
        "  segmentation: "
        f"{identical}/{total}={pct(rate)}; "
        f"resegmented={total - identical:,} tokens, {len(resegmented):,} types"
    )
    return result, resegmented


def chrest_exact_rate(metrics: Any) -> float:
    return metrics.token_exact / (metrics.total_tokens or 1)


def lrt_exact_rate(metrics: Any) -> float:
    return metrics.token_exact / (metrics.answered_tokens or 1)


def foreign_rate(summary: Any) -> float:
    return summary.foreign_unmarked_ok / (summary.foreign_unmarked_tokens or 1)


def run_chrestomatija(checkpoint: Path, batch_size: int) -> dict[str, Any]:
    sentences = chrest.load_gold(chrest.DEFAULT_GOLD, limit=None)
    result = chrest.run_joint(
        sentences=sentences,
        checkpoint=checkpoint,
        batch_size=batch_size,
        force_cpu=False,
        cuda_memory_threshold_mib=999_999,
    )
    if result.status != "ok":
        raise RuntimeError(f"chrestomatija joint failed: {result.error}")
    metrics = chrest.score_result(sentences, result)
    return {
        "sentences": len(sentences),
        "tokens": metrics.total_tokens,
        "answered": metrics.answered_tokens,
        "token_exact": metrics.token_exact,
        "token_exact_rate": chrest_exact_rate(metrics),
        "token_position": metrics.token_position,
        "sentence_exact": metrics.sentence_exact,
        "skipped_gold_tokens": metrics.skipped_gold_tokens,
        "skipped_model_tokens": metrics.skipped_model_tokens,
        "elapsed_seconds": metrics.elapsed_seconds,
    }


def run_joint_benchmarks(checkpoint: Path, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    device = choose_device()
    model, tokenizer, payload = instantiate_from_checkpoint(checkpoint, device=device)
    char_vocab = payload["char_vocab"]
    alksnis = joint_eval.eval_pos_split(
        "alksnis-test",
        SCRIPT_DIR / "data" / "alksnis_test.jsonl",
        model,
        tokenizer,
        char_vocab,
        device,
        limit=None,
        batch_size=batch_size,
    )
    lrt = joint_eval.eval_stress_lrt(
        model,
        tokenizer,
        char_vocab,
        device,
        corpus_path=LRT_CORPUS,
        silver_path=LRT_SILVER,
        audit_path=LRT_AUDIT,
        limit=None,
        batch_size=batch_size,
    )
    cleanup_cuda()
    audited = lrt["audited"]
    summary = lrt["audit_summary"]
    return (
        {
            "alksnis": alksnis,
        },
        {
            "silver_tokens": lrt["silver_tokens"],
            "aligned": lrt["aligned"],
            "skipped_silver": lrt["skipped_silver"],
            "skipped_model": lrt["skipped_model"],
            "raw": lrt["raw"],
            "audited": audited,
            "audit_summary": summary,
            "audited_token_exact_rate": lrt_exact_rate(audited),
            "foreign_unmarked_rate": foreign_rate(summary),
        },
    )


def benchmark_gauntlet(checkpoint: Path, batch_size: int) -> dict[str, Any]:
    print("GATE 3: benchmark gauntlet")
    chrest_result = run_chrestomatija(checkpoint, batch_size)
    joint_result, lrt_result = run_joint_benchmarks(checkpoint, batch_size)
    alksnis = joint_result["alksnis"]
    chrest_rate = float(chrest_result["token_exact_rate"])
    lrt_rate = float(lrt_result["audited_token_exact_rate"])
    alksnis_rate = float(alksnis["label_accuracy"])
    print(
        "  benchmarks: "
        f"chrestomatija={pct(chrest_rate)}; "
        f"LRT audited={pct(lrt_rate)}; "
        f"ALKSNIS label={pct(alksnis_rate)}"
    )
    failures = []
    if chrest_rate < CHREST_GATE:
        failures.append(f"chrestomatija {pct(chrest_rate)} < {pct(CHREST_GATE)}")
    if lrt_rate < LRT_GATE:
        failures.append(f"LRT audited {pct(lrt_rate)} < {pct(LRT_GATE)}")
    if alksnis_rate < ALKSNIS_GATE:
        failures.append(f"ALKSNIS {pct(alksnis_rate)} < {pct(ALKSNIS_GATE)}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "chrestomatija": chrest_result,
        "lrt": lrt_result,
        "alksnis": alksnis,
    }


def run_lrt_only(checkpoint: Path, batch_size: int) -> dict[str, Any]:
    _joint, lrt = run_joint_benchmarks(checkpoint, batch_size)
    return lrt


def predict_token_outputs(
    checkpoint: Path,
    tokens: list[str],
    batch_size: int,
) -> dict[str, dict[str, str | None]]:
    if not tokens:
        return {}
    device = choose_device()
    model, tokenizer, payload = instantiate_from_checkpoint(checkpoint, device=device)
    rows = [
        {
            "id": f"reseg-{index}",
            "text": token,
            "tokens": [{"word": token, "pos_label": "X|_", "stress": None}],
        }
        for index, token in enumerate(tokens)
    ]
    collator = JointCollator(tokenizer, model.labels, payload["char_vocab"])
    loader = DataLoader(
        JointDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    predictions = predict_batches(model, loader, device)
    out: dict[str, dict[str, str | None]] = {}
    for token, row in zip(tokens, predictions):
        pred = row["tokens"][0] if row.get("tokens") else {}
        out[token] = {
            "pos": str(pred.get("pos") or ""),
            "stress": pred.get("stress"),
        }
    cleanup_cuda()
    return out


def foreign_torture(
    original_checkpoint: Path,
    pruned_checkpoint: Path,
    pruned_lrt: dict[str, Any],
    resegmented: Counter[str],
    batch_size: int,
) -> dict[str, Any]:
    print("GATE 4: foreign torture")
    original_lrt = run_lrt_only(original_checkpoint, batch_size)
    original_rate = float(original_lrt["foreign_unmarked_rate"])
    pruned_rate = float(pruned_lrt["foreign_unmarked_rate"])
    delta = abs(pruned_rate - original_rate)
    if delta > FOREIGN_GATE_DELTA:
        raise RuntimeError(
            f"foreign abstention drift {pct(delta)} > {pct(FOREIGN_GATE_DELTA)}"
        )

    tokens = list(resegmented.keys())
    original_outputs = predict_token_outputs(original_checkpoint, tokens, batch_size)
    pruned_outputs = predict_token_outputs(pruned_checkpoint, tokens, batch_size)
    changed = [
        token
        for token in tokens
        if original_outputs.get(token) != pruned_outputs.get(token)
    ]
    changed_occurrences = sum(resegmented[token] for token in changed)
    result = {
        "original_foreign_unmarked_rate": original_rate,
        "pruned_foreign_unmarked_rate": pruned_rate,
        "delta": delta,
        "foreign_tokens_original": original_lrt["audit_summary"].foreign_unmarked_tokens,
        "foreign_tokens_pruned": pruned_lrt["audit_summary"].foreign_unmarked_tokens,
        "resegmented_unique_tokens": len(tokens),
        "changed_output_unique_tokens": len(changed),
        "changed_output_occurrences": changed_occurrences,
        "changed_output_examples": [
            {
                "token": token,
                "count": resegmented[token],
                "original": original_outputs.get(token),
                "pruned": pruned_outputs.get(token),
            }
            for token in changed[:25]
        ],
    }
    print(
        "  foreign: "
        f"original={pct(original_rate)}; pruned={pct(pruned_rate)}; "
        f"delta={pct(delta)}; reseg-output-changed="
        f"{len(changed):,}/{len(tokens):,} types, {changed_occurrences:,} occurrences"
    )
    return result


def append_decisions(
    target: dict[str, list[int]],
    prefix: str,
    decisions: tuple[list[int], list[int]],
) -> None:
    pos, stress = decisions
    target[f"{prefix}_pos"].extend(pos)
    target[f"{prefix}_stress"].extend(stress)


def agreement_stats(
    torch_pos: list[int],
    torch_stress: list[int],
    onnx_pos: list[int],
    onnx_stress: list[int],
) -> dict[str, Any]:
    if not (len(torch_pos) == len(torch_stress) == len(onnx_pos) == len(onnx_stress)):
        raise ValueError("agreement lists have different lengths")
    total = len(torch_pos)
    pos_matches = sum(1 for left, right in zip(torch_pos, onnx_pos) if left == right)
    stress_matches = sum(1 for left, right in zip(torch_stress, onnx_stress) if left == right)
    combined_matches = sum(
        1
        for tp, ts, op, os_ in zip(torch_pos, torch_stress, onnx_pos, onnx_stress)
        if tp == op and ts == os_
    )
    return {
        "tokens": total,
        "pos_matches": pos_matches,
        "stress_matches": stress_matches,
        "combined_matches": combined_matches,
        "pos_rate": pos_matches / (total or 1),
        "stress_rate": stress_matches / (total or 1),
        "combined_rate": combined_matches / (total or 1),
    }


def onnx_parity_for_paths(
    loaded: Any,
    paths: dict[str, Path],
    sentence_count: int,
    batch_size: int,
) -> dict[str, Any]:
    from export_joint_onnx import (  # noqa: PLC0415
        JointExportWrapper,
        load_parity_rows,
        make_collator,
        make_onnx_session,
        onnx_decisions,
        torch_decisions,
    )

    rows = load_parity_rows(LRT_SMOKE, sentence_count)
    collator = make_collator(loaded)
    wrapper = JointExportWrapper(loaded.model).eval()
    decisions: dict[str, list[int]] = {
        "torch_pos": [],
        "torch_stress": [],
    }
    for name in paths:
        decisions[f"{name}_pos"] = []
        decisions[f"{name}_stress"] = []
    sessions = {name: make_onnx_session(path) for name, path in paths.items()}

    for start in range(0, len(rows), batch_size):
        batch = collator(rows[start : start + batch_size])
        append_decisions(decisions, "torch", torch_decisions(wrapper, batch))
        for name, session in sessions.items():
            append_decisions(decisions, name, onnx_decisions(session, batch))

    return {
        name: agreement_stats(
            decisions["torch_pos"],
            decisions["torch_stress"],
            decisions[f"{name}_pos"],
            decisions[f"{name}_stress"],
        )
        for name in paths
    }


def onnx_gates(
    checkpoint: Path,
    output_dir: Path,
    parity_sentences: int,
    batch_size: int,
) -> dict[str, Any]:
    print("GATE 5: ONNX gates")
    from export_joint_onnx import (  # noqa: PLC0415
        DEFAULT_OPSET,
        export_fp32,
        load_joint,
        quantize_int8,
        write_meta,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / FP32_NAME
    partial_path = output_dir / PARTIAL_INT8_NAME
    full_path = output_dir / FULL_INT8_NAME
    meta_path = output_dir / META_NAME

    loaded = load_joint(checkpoint)
    print(f"  exporting fp32: {safe_relative(fp32_path)}")
    export_fp32(loaded, fp32_path, DEFAULT_OPSET)
    print(f"  quantizing partial int8: {safe_relative(partial_path)}")
    partial_quant = quantize_int8(
        fp32_path,
        partial_path,
        per_channel=False,
        encoder_layers=4,
    )
    print(f"  quantizing full dynamic int8: {safe_relative(full_path)}")
    full_quant = quantize_int8(
        fp32_path,
        full_path,
        per_channel=False,
        encoder_layers=-1,
    )

    parity = onnx_parity_for_paths(
        loaded,
        {"fp32": fp32_path, "partial_int8": partial_path, "full_int8": full_path},
        parity_sentences,
        batch_size,
    )
    fp32_rate = float(parity["fp32"]["combined_rate"])
    partial_rate = float(parity["partial_int8"]["combined_rate"])
    full_rate = float(parity["full_int8"]["combined_rate"])
    passing_int8 = []
    if partial_rate >= INT8_GATE:
        passing_int8.append(("partial_int8", partial_path, partial_quant))
    if full_rate >= INT8_GATE:
        passing_int8.append(("full_int8", full_path, full_quant))

    print(
        "  parity combined: "
        f"fp32={pct(fp32_rate)}; partial_int8={pct(partial_rate)}; "
        f"full_int8={pct(full_rate)}"
    )
    if fp32_rate < FP32_GATE:
        raise RuntimeError(f"FP32 ONNX combined parity below gate: {pct(fp32_rate)}")
    if not passing_int8:
        raise RuntimeError(
            "no INT8 recipe passed gate: "
            f"partial={pct(partial_rate)}, full={pct(full_rate)}"
        )

    shipped_name, shipped_path, shipped_quant = min(
        passing_int8,
        key=lambda item: item[1].stat().st_size,
    )
    selected_parity = {
        "sentences": parity_sentences,
        "tokens": parity[shipped_name]["tokens"],
        "fp32_pos_matches": parity["fp32"]["pos_matches"],
        "fp32_pos_agreement": parity["fp32"]["pos_rate"],
        "fp32_stress_matches": parity["fp32"]["stress_matches"],
        "fp32_stress_agreement": parity["fp32"]["stress_rate"],
        "fp32_combined_matches": parity["fp32"]["combined_matches"],
        "fp32_combined_agreement": parity["fp32"]["combined_rate"],
        "int8_pos_matches": parity[shipped_name]["pos_matches"],
        "int8_pos_agreement": parity[shipped_name]["pos_rate"],
        "int8_stress_matches": parity[shipped_name]["stress_matches"],
        "int8_stress_agreement": parity[shipped_name]["stress_rate"],
        "int8_combined_matches": parity[shipped_name]["combined_matches"],
        "int8_combined_agreement": parity[shipped_name]["combined_rate"],
    }
    write_meta(
        meta_path,
        loaded=loaded,
        checkpoint=checkpoint,
        fp32_path=fp32_path,
        int8_path=shipped_path,
        opset=DEFAULT_OPSET,
        parity=selected_parity,
        quantization=shipped_quant,
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["all_parity"] = parity
    meta["all_quantization"] = {
        "partial_int8": partial_quant,
        "full_int8": full_quant,
    }
    meta["shipped_int8_recipe"] = shipped_name
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "fp32": parity["fp32"],
        "partial_int8": parity["partial_int8"],
        "full_int8": parity["full_int8"],
        "shipped_int8_recipe": shipped_name,
        "shipped_int8_file": shipped_path.name,
        "meta": safe_relative(meta_path),
    }


def file_size(path: Path | None) -> int | None:
    return path.stat().st_size if path and path.exists() else None


def tokenizer_size(path: Path) -> int:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "config.json",
    )
    return sum((path / name).stat().st_size for name in names if (path / name).exists())


def size_row(label: str, before: int | None, after: int | None) -> dict[str, Any]:
    reduction = None
    if before and after is not None:
        reduction = 1 - (after / before)
    return {
        "artifact": label,
        "before_bytes": before,
        "after_bytes": after,
        "reduction": reduction,
    }


def print_size_table(rows: list[dict[str, Any]]) -> None:
    print("GATE 6: sizes")
    print("| artifact | before | after | reduction |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        before = row["before_bytes"]
        after = row["after_bytes"]
        reduction = row["reduction"]
        before_text = "-" if before is None else f"{before:,}"
        after_text = "-" if after is None else f"{after:,}"
        reduction_text = "-" if reduction is None else pct(float(reduction))
        print(f"| {row['artifact']} | {before_text} | {after_text} | {reduction_text} |")


def size_table(
    original_checkpoint: Path,
    pruned_checkpoint: Path,
    onnx_dir: Path,
) -> list[dict[str, Any]]:
    before_full = OLD_BUNDLE / "joint.full-int8.onnx"
    rows = [
        size_row("torch checkpoint", file_size(original_checkpoint), file_size(pruned_checkpoint)),
        size_row("onnx fp32", file_size(HF_RELEASE / "joint.onnx"), file_size(onnx_dir / FP32_NAME)),
        size_row(
            "int8 partial",
            file_size(HF_RELEASE / "joint.int8.onnx"),
            file_size(onnx_dir / PARTIAL_INT8_NAME),
        ),
        size_row(
            "int8 full dynamic",
            file_size(before_full),
            file_size(onnx_dir / FULL_INT8_NAME),
        ),
        size_row(
            "tokenizer",
            tokenizer_size(HF_RELEASE),
            tokenizer_size(PRUNED_DIR / "tokenizer"),
        ),
    ]
    print_size_table(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, default=DEFAULT_ORIGINAL_CHECKPOINT)
    parser.add_argument("--pruned-checkpoint", type=Path, default=DEFAULT_PRUNED_CHECKPOINT)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--resegmented", type=Path, default=DEFAULT_RESEGMENTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--onnx-batch-size", type=int, default=8)
    parser.add_argument("--parity-sentences", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    for path in (args.original_checkpoint, args.pruned_checkpoint):
        require_file(path)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "original_checkpoint": safe_relative(args.original_checkpoint),
        "pruned_checkpoint": safe_relative(args.pruned_checkpoint),
    }

    report["remap_sanity"] = remap_sanity(
        args.original_checkpoint,
        args.pruned_checkpoint,
        args.batch_size,
    )
    segmentation, resegmented = segmentation_census(
        args.original_checkpoint,
        args.pruned_checkpoint,
        args.resegmented,
    )
    report["segmentation_census"] = segmentation
    benchmark = benchmark_gauntlet(args.pruned_checkpoint, args.batch_size)
    report["benchmarks"] = benchmark
    report["foreign_torture"] = foreign_torture(
        args.original_checkpoint,
        args.pruned_checkpoint,
        benchmark["lrt"],
        resegmented,
        args.batch_size,
    )
    report["onnx_gates"] = onnx_gates(
        args.pruned_checkpoint,
        args.onnx_dir,
        args.parity_sentences,
        args.onnx_batch_size,
    )
    report["sizes"] = size_table(
        args.original_checkpoint,
        args.pruned_checkpoint,
        args.onnx_dir,
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "passed"
    write_json(args.report, report)
    print(f"gauntlet report written: {safe_relative(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
