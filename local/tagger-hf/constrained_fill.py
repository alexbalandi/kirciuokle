"""Constrained-decoding self-fill for MATAS gap features (NC-free).

Plain self-training fails here: the self-teacher learns to OMIT Number/Person
exactly where the gold corpus omits them, so its argmax label rarely carries
the missing key. Constrained decoding fixes that: for each gap token we
compare ONLY candidate labels that (a) belong to the token's UPOS family,
(b) agree with every feature the gold analysis already specifies, and
(c) carry the missing key — then take the key's value from the best-scoring
candidate. Bare labels never compete, so the omission prior is bypassed.

Gap tokens (same definition as teacher_fill.py):
  - UPOS VERB/AUX missing Number  -> fill Number
  - UPOS PRON missing Person      -> fill Person

Usage (GPU strongly recommended):
    .venv-train/Scripts/python.exe local/tagger-hf/constrained_fill.py \
        --model-dir local/tagger-hf/runs/litlat__selfteacher__stage2/best \
        --input local/tagger-hf/data/raw/MATAS3.conllu \
        --output local/tagger-hf/data/raw/MATAS3.selftrain2.conllu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

GAP_RULES = {
    "VERB": ("Number", ("VERB", "AUX")),
    "AUX": ("Number", ("VERB", "AUX")),
    "PRON": ("Person", ("PRON", "DET")),
}
MAX_LENGTH = 256


def parse_feats(raw: str) -> dict[str, str]:
    if raw in ("_", ""):
        return {}
    out = {}
    for kv in raw.split("|"):
        if "=" in kv:
            key, value = kv.split("=", 1)
            out[key] = value
    return out


def label_parts(label: str) -> tuple[str, dict[str, str]]:
    if "|" not in label:
        return label, {}
    upos, feats = label.split("|", 1)
    return upos, parse_feats(feats)


def build_candidates(labels: list[str]) -> dict[str, list[tuple[int, dict[str, str]]]]:
    """gap-upos -> [(label_id, label_feats)] for labels carrying the gap key."""
    table: dict[str, list[tuple[int, dict[str, str]]]] = {u: [] for u in GAP_RULES}
    for gap_upos, (key, families) in GAP_RULES.items():
        for index, label in enumerate(labels):
            upos, feats = label_parts(label)
            if upos in families and key in feats:
                table[gap_upos].append((index, feats))
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-sentences", type=int, default=64)
    ap.add_argument("--limit", type=int, help="first N sentences (trial)")
    ap.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="fill only when the best value's logit beats the best competing "
        "value by this margin; 0 fills everything. Gold treebank convention "
        "marks these features only where context determines them - the gate "
        "mimics that annotator silence on genuinely ambiguous tokens.",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device).eval()
    if device == "cuda":
        model.to(dtype=torch.bfloat16)

    id2label = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    candidates = build_candidates(id2label)
    for upos, cands in candidates.items():
        print(f"candidate labels for {upos} gap: {len(cands)}", file=sys.stderr)

    # ---- read sentences (token line lists between blank lines) ----------
    sentences: list[list[str]] = []
    current: list[str] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sentences.append(current)
    if args.limit:
        sentences = sentences[: args.limit]

    fills = {"Number": 0, "Person": 0}
    processed = 0
    out_lines: list[str] = []

    def flush_batch(batch: list[tuple[list[str], list[int], list[int]]]) -> None:
        """batch items: (sentence_lines, word_line_idx list, gap positions)."""
        nonlocal fills
        forms = [
            [ln.split("\t")[1] for ln in lines if ln.split("\t")[0].isdigit()]
            for lines, _, _ in batch
        ]
        encoded = tokenizer(
            forms,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits.float()

        for batch_index, (lines, word_line_idx, gaps) in enumerate(batch):
            word_ids = encoded.word_ids(batch_index=batch_index)
            first_subword = {}
            for position, word_id in enumerate(word_ids):
                if word_id is not None and word_id not in first_subword:
                    first_subword[word_id] = position
            for word_index in gaps:
                if word_index not in first_subword:
                    continue
                line_i = word_line_idx[word_index]
                cols = lines[line_i].split("\t")
                gold_feats = parse_feats(cols[5])
                key, _ = GAP_RULES[cols[3]][0], None
                token_logits = logits[batch_index, first_subword[word_index]]
                best_per_value: dict[str, float] = {}
                for label_id, label_feats in candidates[cols[3]]:
                    # candidate must agree with everything gold specifies
                    if any(label_feats.get(k) != v for k, v in gold_feats.items()):
                        continue
                    score = token_logits[label_id].item()
                    value = label_feats[key]
                    if value not in best_per_value or score > best_per_value[value]:
                        best_per_value[value] = score
                best_value = None
                if best_per_value:
                    ranked = sorted(best_per_value.items(), key=lambda kv: -kv[1])
                    margin = (
                        ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
                    )
                    if margin >= args.min_margin:
                        best_value = ranked[0][0]
                if best_value is not None:
                    gold_feats[key] = best_value
                    cols[5] = "|".join(f"{k}={v}" for k, v in sorted(gold_feats.items()))
                    lines[line_i] = "\t".join(cols)
                    fills[key] += 1

    batch: list[tuple[list[str], list[int], list[int]]] = []
    for lines in sentences:
        processed += 1
        word_line_idx = [i for i, ln in enumerate(lines) if ln.split("\t")[0].isdigit()]
        gaps = []
        for word_index, line_i in enumerate(word_line_idx):
            cols = lines[line_i].split("\t")
            if len(cols) < 6 or cols[3] not in GAP_RULES:
                continue
            key = GAP_RULES[cols[3]][0]
            if key not in parse_feats(cols[5]):
                gaps.append(word_index)
        if gaps:
            batch.append((lines, word_line_idx, gaps))
            if len(batch) >= args.batch_sentences:
                flush_batch(batch)
                batch = []
        if processed % 20000 == 0:
            print(f"  {processed:,} sentences, fills so far: {fills}", file=sys.stderr)

    if batch:
        flush_batch(batch)

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for lines in sentences:
            handle.write("\n".join(lines) + "\n\n")

    print(f"processed {processed:,} sentences; fills: {fills}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
