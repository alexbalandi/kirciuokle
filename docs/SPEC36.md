# SPEC36 — Rehearsal-mixture builder + literary fine-tune wiring

## Context

The literary fine-tune loop: teacher-labeled literary sentences (accent
supervision at ~99.5% purity, POS masked) mixed with a MATAS rehearsal
slice (gold POS + projected stress) so the shared encoder can't drift;
fine-tune from `joint/checkpoints/joint_v1_polish.best.pt` at polish LR;
judged later on the untouched chrestomatija. This spec builds the
dataset plumbing.

One new script `local/accentuator/joint/build_finetune_mixture.py`,
plus (ONLY IF NEEDED) a minimal patch to the dataset loader in
`local/accentuator/joint/joint_lib.py` so a token row with
`pos_label: null` contributes NO POS loss (LABEL_PAD_ID) while its
stress target still trains — verify whether the loader already handles
null; patch minimally if not; nothing else changes.

## Mixture builder

Inputs:
- `--literary` : a teacher `label.py` output jsonl (joint-training
  token format; stress accepted per calibrated strata; POS null).
- `--matas-dir`: the existing full joint dataset dir (joint/data) —
  sample sentences from its train.jsonl.
- `--rehearsal-ratio` (default 0.25): MATAS tokens ≈ ratio × literary
  tokens (sample whole sentences, seed 20260705).
- `--dev-share` (default 0.02): hold out literary SENTENCES as the
  mixture's dev.jsonl (dev must be literary — MATAS dev is saturated
  and would not rank literary fit).

Output: a data dir (`--out`, default joint/data-literary) with
train.jsonl / dev.jsonl / labels.json (copy the MATAS labels.json —
the label set must stay IDENTICAL to the checkpoint's, assert it) and
stats: literary tokens (stress-supervised count), MATAS tokens, ratio,
dev size.

GUARD: refuse any `--literary` path whose name contains "chrestomatija"
unless `--allow-benchmark-smoke` is passed (the benchmark must never
become training data; the flag exists only for plumbing smoke tests).

## Fine-tune invocation (document in the module docstring; do not run
the full training — the human launches it)

```
.venv-train/Scripts/python.exe local/accentuator/joint/train_joint.py \
  --init-checkpoint local/accentuator/joint/checkpoints/joint_v1_polish.best.pt \
  --data-dir local/accentuator/joint/data-literary \
  --epochs 2 --lr-scale 0.1 --schedule constant \
  --checkpoint local/accentuator/joint/checkpoints/joint_v2_literary.pt
```

## Pass criteria

1. Loader null-POS handling verified (state whether a patch was needed;
   if patched, show a 3-line smoke proving a null-POS row trains stress
   and contributes zero POS loss).
2. Mixture smoke: build with
   `--literary data/teacher/chrestomatija-plain.labeled.jsonl` (or the
   actual SPEC34 output name on disk) `--allow-benchmark-smoke
   --max-literary-sentences 200` → dir written, stats printed, labels
   assertion passes.
3. Guard check: same command WITHOUT the flag → refusal, nonzero exit.
4. A 30-step CPU training smoke from the init checkpoint on the smoke
   mixture runs (CUDA_VISIBLE_DEVICES empty, tiny batch) — proves
   end-to-end format compatibility. Delete smoke outputs.

Do not commit.
