# Retraining & evaluating the local accentuation model

The end-to-end recipe for producing a new `joint_vN` (the in-browser POS+stress
model), gating it, and shipping it everywhere. Written for both humans and
agents; every command was exercised for the v4 release (2026-07, SPEC59).
Deploy etiquette (dev auto / prod owner-gated) is in [AGENTS.md](../AGENTS.md).

## 0. Prerequisites

- **GPU** for training (v4 took ~2 h/epoch on an RTX 3080 Ti Laptop, 16 GB).
  Export/quantization run on CPU.
- **Training venv** (not committed; recreate as needed):

  ```sh
  uv venv local/accentuator/joint/.venv --python 3.12
  uv pip install --python local/accentuator/joint/.venv/Scripts/python.exe \
      torch --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python local/accentuator/joint/.venv/Scripts/python.exe \
      "transformers==4.51.3" "tokenizers<0.22" onnx onnxruntime numpy \
      safetensors sentencepiece huggingface_hub
  ```

  **transformers must stay <5** — 5.x can no longer load the saved XLM-R
  tokenizer (`Unigram(vocab=dict)` TypeError).
- **Gitignored data** that must exist locally:
  `local/accentuator/data/{lexicon.sqlite, generated.sqlite}` (the open
  dictionary), `local/tagger-hf/data/gen2/` (MATAS+ALKSNIS source rows),
  `local/accentuator/data/teacher/round2-combined.labeled.jsonl` (teacher
  labels). Checkpoint lineage lives in `local/accentuator/joint/checkpoints/`.
- **.env**: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` (R2 upload),
  `HF_TOKEN` (Hugging Face).
- All commands below run from `local/accentuator/joint/` with
  `PY=.venv/Scripts/python.exe` unless noted.

## 1. Rebuild the datasets

Stress labels are *projected* from the dictionary onto corpus tokens by
scoring VDU morphology strings against the corpus POS — so parser fixes in
`local/app/kirciuokle/disambiguate.py` (e.g. the SPEC59 `jstk.` fix) only take
effect after a rebuild:

```sh
$PY build_joint_dataset.py                       # -> data/ (MATAS projection)
$PY build_finetune_mixture.py \
    --literary ../data/teacher/round2-combined.labeled.jsonl \
    --out data-round2-r50 --rehearsal-ratio 0.5 \
    --init-checkpoint checkpoints/joint_v1_polish.best.pt   # -> mixture
```

Sanity-check `data/stats.json` (supervision share ~52–55%) and, if your change
targets specific words, grep `data/train.jsonl` to confirm they now carry
`"stress": [pos, mark]` instead of `null`.

## 2. Train

The v4 recipe (same as v3, but 0.5 rehearsal + slightly more head plasticity):

```sh
$PY train_joint.py --data-dir data-round2-r50 \
    --init-checkpoint checkpoints/joint_v2_literary.best.pt \
    --epochs 2 --lr-scale 0.2 --schedule constant \
    --checkpoint checkpoints/joint_vN.pt
```

Logs print `loss / pos / stress` per 10 steps and a dev eval per epoch;
`joint_vN.best.pt` is written whenever dev combined improves. Checkpoints save
only at epoch ends — a killed run loses the in-progress epoch. Agents: run it
as a harness background task (headless); never spawn console windows.

## 3. Evaluate (the gates)

```sh
$PY panel_intj.py --checkpoint checkpoints/joint_vN.best.pt   # must be 11/11
$PY eval_joint.py --checkpoint checkpoints/joint_vN.best.pt
$PY eval_joint.py --checkpoint checkpoints/joint_v4.best.pt   # baseline, same protocol
```

Ship criteria: the interjection panel at 11/11; LRT stress (raw + audited) and
the literary-dev stress at or above the previous release; POS within noise
(±0.2 pp). v4 reference numbers are tabled in [SPEC59.md](SPEC59.md).

## 4. Prune, gauntlet, export

```sh
mv pruned pruned-vOLD-archive          # prune_vocab refuses to overwrite
$PY prune_vocab.py --checkpoint checkpoints/joint_vN.best.pt --output-dir pruned
mv pruned/joint_v3.pruned.pt pruned/joint_vN.pruned.pt   # output name is hardcoded
$PY spec48_gauntlet.py --original-checkpoint checkpoints/joint_vN.best.pt \
    --pruned-checkpoint pruned/joint_vN.pruned.pt --onnx-dir pruned/onnx
```

The gauntlet is the artifact factory: it runs remap sanity, segmentation
census, benchmarks, foreign torture, exports **fp32 + both int8 recipes** into
`pruned/onnx/`, measures parity, and writes `gauntlet_report.json`. Gates:
fp32 parity 100%, partial-int8 ≥98%, benchmarks within ~0.3 pp of the
previous release's gauntlet numbers.

## 5. Build the browser bundle

From the repo root:

```sh
uv run scripts/prepare_local_model.py \
  --model-onnx local/accentuator/joint/pruned/onnx/joint.int8.partial.onnx \
  --model-name joint.int8.partial.onnx \
  --light-onnx local/accentuator/joint/pruned/onnx/joint.int8.full.onnx \
  --light-name joint.int8.full.onnx \
  --meta-json local/accentuator/joint/pruned/onnx/joint.meta.json \
  --tokenizer-dir local/accentuator/joint/pruned/tokenizer \
  --skip-parity --skip-full
```

`--skip-parity` is required: the script's parity reference is hardcoded to
`joint_v2_literary` and produces garbage numbers against newer models (the
truthful parity lives in `joint.meta.json` from the gauntlet). `--skip-full`
avoids building a stray third variant from stale `hf_release/` artifacts. If a
previous run wrote bogus parity values into `local-model/manifest.json`, null
them — the generator carries "existing" values forward.

**Verify in the browser before shipping**: `npm run dev` serves `local-model/`;
clear the `main-local-accent-model-v1` cache, switch to Local mode, and check a
sentence with interjections + controls, e.g. `Prašom užeiti. Ačiū labai.` →
`Prãšom užeĩti. Ãčiū labaĩ.`

## 6. Ship

```sh
uv run scripts/upload_local_model_r2_multipart.py   # R2 (manifest uploaded last)
npm run deploy:dev                                  # then verify on the dev URL
npm run deploy:prod                                 # OWNER-GATED — see AGENTS.md
uv run scripts/upload_pruned_to_hf.py               # HF pruned/ set
local/accentuator/joint/.venv/Scripts/python.exe \
  local/accentuator/joint/package_hf.py \
  --checkpoint local/accentuator/joint/checkpoints/joint_vN.best.pt --upload   # HF root
```

- **Do not use `scripts/upload_local_model_to_r2.sh` for the full bundle** —
  `wrangler r2 object put` caps at 300 MiB and the heavy tier (~450 MB) fails;
  the multipart script derives S3 credentials from `CLOUDFLARE_API_TOKEN`.
- `package_hf.py` **regenerates the HF model card**, dropping the
  pruned-variant section and any changelog — re-append them after (see
  `scripts/upload_pruned_to_hf.py`'s card logic, and add a dated changelog
  entry describing what changed).
- R2 is shared by dev and prod: once the bucket is updated, **both** serve the
  new model. Deploy order stays dev → verify → prod for the site code.

## Quick evaluation only (no retrain)

```sh
cd local/accentuator/joint
.venv/Scripts/python.exe panel_intj.py --checkpoint checkpoints/joint_v4.best.pt
.venv/Scripts/python.exe eval_joint.py --checkpoint checkpoints/joint_v4.best.pt
```

## Lineage

`joint_v1` (MATAS, `data/`) → `v1_polish` → `v2_literary` (literary mixture,
`data-literary/`) → `v3` (round-2 teacher mixture, `data-round2/`) → **`v4`**
(same recipe on jstk.-fixed data, `data-round2-r50/`) — see
[SPEC59.md](SPEC59.md) for why and the before/after numbers.
