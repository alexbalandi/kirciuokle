# SPEC34 — Teacher-labeler pipeline (etalon annotations for new corpora)

## Goal

A pipeline that produces the BEST-ACHIEVABLE accentuation + POS labels for
arbitrary Lithuanian text, by combining every system in this repo and
accepting a label only when an empirically-calibrated agreement stratum
vouches for it. Purpose: manufacture high-purity training data from new
corpora (public-domain literary classics, more LRT) — coverage is
sacrificial, purity is the product (training masks unlabeled tokens).

Measured single-system quality on the chrestomatija gold (accents,
token exact): vdu-udpipe 95.8%, joint 86.9%, liepa 76.7% (98.1%
position of answered), dict-default 67.8% (91.5% exact of answered).
Known pairwise result: nn∧liepa agreement = 99.5% exact. The teacher
should beat every row via stratified consensus.

New directory `local/accentuator/teacher/` for all scripts; do not modify
existing files. Run with `.venv-train/Scripts/python.exe` (torch needed).

## 1. `collect_layers.py` — multi-system annotation of a corpus

Input: `--corpus` (plain text, one sentence per line) and optionally
`--vdu-silver` (existing build_silver_truth output for that corpus, to
avoid re-calling external services when present — REQUIRED present for
the calibration corpora; the script must NOT call external services
itself; if no silver is given, the vdu layer is simply absent).

Per word token, collect into one jsonl (token stream, sentence ids):
- `vdu`: accented form + mi label from the silver jsonl (aligned by the
  two-cursor stripped-surface method used everywhere).
- `joint`: the joint model's accented form + its POS label
  (checkpoint `joint/checkpoints/joint_v1_polish.best.pt`; reuse
  joint/eval_joint.py inference by import).
- `liepa`: phonology_engine form (guess_uncovered.engine_accent).
- `dict`: dictionary variant matched against the JOINT model's predicted
  label via score_tags (reuse pick_dict_form logic from
  eval_nodict_pipeline.py), plus the dictionary's default form.
- `tagger`: the released -vdu tagger's label for the token (serve via
  the tagger subprocess machinery from eval_nodict_pipeline.py) — this
  is a second POS opinion independent of the joint model.

Output: `data/teacher/<corpus-name>.layers.jsonl`. Resumable per
sentence. GPU for joint; check nvidia-smi and fall back to CPU.

## 2. `calibrate.py` — measure every stratum against gold

Two calibration tracks:

ACCENTS — corpus: chrestomatija. Gold from
`data/eval/chrestomatija-gold.jsonl`; layers file from step 1 (the vdu
silver already exists at `data/eval/chrestomatija-vdu-silver.jsonl`).
For every token compute its agreement PATTERN over the accent layers
(which nonempty layers produce the identical NFC form: e.g.
`vdu+joint+liepa`, `vdu+joint`, `vdu-only`, `joint+dict vs vdu`, ...).
Report per-pattern: token count, share, and accuracy vs gold. Emit
`data/teacher/accent-strata.json`: for each pattern, accuracy + count.

POS — corpus: the ALKSNIS gold test set (prepared jsonl on disk under
local/tagger-hf/data/; it has gold labels). Layers here: joint POS
label, tagger label, and (no vdu layer needed) — report agreement
strata (joint=tagger vs disagree) accuracy against gold labels, both
full-label and the slot projection (reuse the slot machinery from
kirciuokle.disambiguate via parse-and-compare or the metrics module in
local/tagger-hf if importable). Emit `data/teacher/pos-strata.json`.

Also print a human table of both tracks sorted by accuracy.

## 3. `label.py` — apply the policy

Input: layers jsonl + strata jsons + `--min-accent-accuracy` (default
0.98) + `--min-pos-accuracy` (default 0.95).
Per token: accept the accent iff its agreement pattern's calibrated
accuracy ≥ threshold (accepted form = the consensus form); accept the
POS label iff its pattern qualifies (accepted label = joint's, which
the calibration will likely show is fine when it matches the tagger).
Output: `data/teacher/<corpus>.labeled.jsonl` in the EXACT format the
joint dataset builder consumes (study joint/build_joint_dataset.py
output: tokens with word, pos_label, stress target or null), plus
stats: coverage per layer-track (accent coverage, POS coverage, both),
purity estimate (weighted average of accepted strata accuracies).

## Pass criteria

1. `collect_layers.py --corpus data/eval/chrestomatija-plain.txt --vdu-silver data/eval/chrestomatija-vdu-silver.jsonl` completes (GPU ok),
   layers jsonl written, per-layer nonempty counts printed.
2. `calibrate.py` prints both strata tables; accent table MUST show the
   full-agreement stratum (all of vdu/joint/liepa nonempty and equal)
   with accuracy ≥ 0.98 — paste the full table.
3. `label.py` on the chrestomatija layers with default thresholds:
   report accent coverage + purity estimate (expect purity ≥ 0.98 at
   coverage well above 50%).
4. Paste both strata tables and the final coverage/purity numbers.

Do not commit.
