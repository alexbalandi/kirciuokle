# SPEC30 — Polish pass + dev-based best checkpoint for the joint trainer

## Context

The joint model trains with warmup+cosine (joint_lib.step_schedule), batch
16 sentences — the loss oscillates downward and the final weights land
wherever the last step happens to be. Two upgrades so we can (a) keep the
best point of the main run and (b) run a low-LR polish pass afterwards.

Modify ONLY `local/accentuator/joint/train_joint.py` and (if helpers are
needed) `local/accentuator/joint/joint_lib.py`. A training run may be
executing while you work — editing files is safe (its code is loaded), but
run NOTHING on the GPU; all pass criteria are CPU smoke runs with
CUDA_VISIBLE_DEVICES empty.

## 1. Per-epoch dev evaluation + best-checkpoint selection (main loop)

- After each epoch: evaluate on the prepared dev split (`dev.jsonl` in the
  data dir) capped at `--dev-eval-sentences` (default 1000): POS label
  accuracy + stress row exact (only rows with stress supervision;
  a no-stress target counts correct when the model predicts no-stress).
  Print both plus a combined score = (pos_acc + stress_acc) / 2.
- Save every epoch to the checkpoint path atomically (crash insurance,
  same .tmp/replace pattern as train_stress_nn.py), AND track the best
  combined score: when an epoch beats it, additionally copy to
  `<checkpoint stem>.best.pt`. Final print names the best epoch and both
  files.

## 2. Polish mode

- `--init-checkpoint PATH`: load model weights from an existing joint
  checkpoint before training (bypasses the encoder/stress-head warm-start
  logic; label set and char vocab come from the checkpoint and must match
  the dataset — assert and fail clearly otherwise).
- `--lr-scale FLOAT` (default 1.0): multiplies both encoder-lr and
  head-lr.
- `--schedule {cosine,constant}` (default cosine): constant = flat LR
  after a 50-step warmup (use for polish).
- Polish invocation shape (document it in the module docstring):
  `train_joint.py --init-checkpoint checkpoints/joint_v1.pt --epochs 1
  --lr-scale 0.1 --schedule constant --checkpoint checkpoints/joint_v1_polish.pt`
  The per-epoch dev eval from part 1 then reports whether polish beat the
  init point (evaluate the init checkpoint on dev BEFORE training starts
  when --init-checkpoint is set, as the baseline to beat, and say so).

## Pass criteria (CPU only, smoke dataset already on disk is fine — but
do NOT overwrite the real dataset or checkpoints; use --checkpoint under
a smoke- prefix and delete after)

1. `CUDA_VISIBLE_DEVICES= .venv-train/Scripts/python.exe local/accentuator/joint/train_joint.py --max-sentences 300 --epochs 2 --batch-size 8 --checkpoint local/accentuator/joint/checkpoints/smoke-a.pt`
   → per-epoch dev metrics print; smoke-a.pt and smoke-a.best.pt written;
   best epoch named.
2. `CUDA_VISIBLE_DEVICES= .venv-train/Scripts/python.exe local/accentuator/joint/train_joint.py --init-checkpoint local/accentuator/joint/checkpoints/smoke-a.best.pt --epochs 1 --lr-scale 0.1 --schedule constant --max-sentences 300 --batch-size 8 --checkpoint local/accentuator/joint/checkpoints/smoke-b.pt`
   → baseline dev eval of the init checkpoint printed first, then polish
   epoch, then comparison line.
3. Delete both smoke checkpoints. Report the printed numbers.

Do not commit.
