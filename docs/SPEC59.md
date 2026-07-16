# SPEC59 — The `jstk.` bug: why the local model mis-stressed prašom (and every interjection)

## Symptom

The shipped joint model (all variants: fp32, int8-partial "heavy", int8-full
"light") consistently output **prašõm** (tilde on the final *o*) instead of the
correct **prãšom**, with the correct answer ranked ~12 logits below the wrong one.
Same for **ačiū** → *Ačiū̃* instead of **ãčiū**. Web mode was unaffected
(dictionary-first; every dictionary source has the right forms).

## Root cause — one missing letter in an abbreviation table

VDU morphology strings abbreviate *jaustukas* (interjection) as **`jstk.`**, but
`MI_TAGS` in `local/app/kirciuokle/disambiguate.py` (and its TS twin
`src/shared/tags.ts`, and the copy in `scripts/accent_text.py`) only mapped
**`jst.`** — which never substring-matches `"jstk.,"`. Consequences, in order:

1. `parse_mi("jstk., pagr.")` → `{}` → `score_tags({}, {pos: INTJ})` → **0**.
2. In the joint-dataset stress projector (`joint_lib.project_stress`), the
   interjection variant scores 0 and the competing verb variant scores −3 against
   an INTJ context, so `best_score <= 0` → **stress target = None = masked from
   the loss**. The dictionary knew `prãšom` all along; the projector just couldn't
   match it to the corpus token.
3. Blast radius (measured on the v3 training data): **2,450 masked INTJ tokens,
   220 distinct interjections, only 11 supervised (0.4%)** — `na`, `deja`,
   `prašom` (229×), `ačiū` (155×), `labas`, `dėkui`, `sudie`, …
4. The model, never supervised on these words but seeing them constantly,
   generalized from supervised `-om` endings (end-stressed instrumental plurals:
   `gerõm`, `tõm`) → confidently wrong `prašõm`.

## The self-training echo (round 2)

In the round-2 teacher, the layer votes for `prašom` were dict=`prãšom`,
LIEPA=`prãšom`, VDU=`prãšom` vs joint(v1)=`prašõm`. That pattern
(`vdu+liepa+dict vs joint`) has calibrated accuracy 0.930 < the 0.98
`min_accent_accuracy` gate in `teacher/label.py` → the token was dropped to null
**again**. The model's own round-1 error vetoed its correction. No gate change is
needed: once the joint layer stops dissenting, the all-agree pattern (0.998)
passes.

## What was ruled out (investigation 2026-07-08)

- **Architecture** — sound. Single flat argmax over validity-masked (char, mark)
  pairs + a no-stress class; no separate syllable/letter stages to disagree.
  Supervised siblings `prãšome`/`prãšomas` decode perfectly.
- **ONNX export / quantization** — fp32, int8-partial and int8-full all make the
  identical error; fp32-vs-checkpoint parity is 1.000 on 1,106 tokens.
- **Browser inference/decode** — a bit-exact Python replication of
  `engine.ts`/`decode.ts` reproduces the browser output on all test words.
- **The morphology taggers on HF** (`litlat-bert-lithuanian-morphology-{full,ud,vdu}`)
  — **not affected**: their pipeline (`local/tagger-hf/`) never touches the MI
  parser; labels come from MATAS/ALKSNIS gold + XPOS reconstruction + UDPipe
  teacher fill. No stress head, no dictionary-variant scoring.

## Fix

1. Add `"jstk.": INTJ` to all three MI maps (keep `"jst."` defensively; it is
   attested nowhere in the current dictionary — 218 `jstk.` strings, 0 bare
   `jst.`). Regression tests in `local/app/tests/test_disambiguate.py` and
   `test/disambiguation.test.ts`.
2. Rebuild the joint datasets (`build_joint_dataset.py`, then
   `build_finetune_mixture.py`) — INTJ tokens become supervised via the fixed
   MATAS projection; the round-2 mixture carries them in through the 25% MATAS
   rehearsal even though the stored teacher labels keep their nulls.
3. Polish-retrain from `joint_v3.best.pt` (low-LR constant schedule, the
   documented train_joint.py polish pattern) on the rebuilt mixture.
4. Re-run the SPEC48 chain: vocab prune → gauntlet → ONNX export → int8
   quantization → parity gate — then ship: `local-model/` bundle, R2
   (`kirciuokle-models`), dev deploy → verify → prod, and the HF accentuator repo.

Acceptance: prãšom/ãčiū correct in Local mode on the deployed site; ALKSNIS
dev/test stress accuracy within noise of v3; parity/gauntlet gates pass.

## Outcome (joint_v4, shipped 2026-07-16)

Retrained with the exact v3 recipe (init `joint_v2_literary`, 2 epochs, constant
schedule) on the corrected data with a 0.5 rehearsal ratio and lr_scale 0.2.
INTJ supervision went from 11 to 806 tokens (`prašom` 216×, `ačiū` 135×, …).

| metric | v3 | v4 |
|---|---|---|
| interjection panel | 6/11 | **11/11** |
| literary dev stress | 96.67% | **97.31%** |
| LRT token exact (raw / audited) | 90.8 / 92.2 | **91.7 / 93.0** |
| stress type exact | 88.2% | **89.4%** |
| chrestomatija (gauntlet) | 90.7% | **90.96%** |
| POS (MATAS dev / ALKSNIS test) | 98.79 / 88.95 | 98.82 / 88.85 (noise) |
| fp32 / int8-partial ONNX parity | 1.000 / 99.2% | 1.000 / 99.2% |

Every stress metric improved; POS unchanged. Verified end-to-end in the real
browser on the deployed site: `Prãšom užeĩti. Ãčiū labaĩ. Lãbas rýtas, dė̃kui ùž
vìską.` Shipped to R2 (`c891ef9765`), dev + prod, and both HF sets (`pruned/` +
root release).

Found along the way: `await nextFrame()` (bare rAF) between inference batches
hung the whole Local run in throttled/background tabs — fixed by racing a 250ms
timeout. And `wrangler r2 object put` caps at 300 MiB; the heavy tier ships via
`scripts/upload_local_model_r2_multipart.py` (S3 multipart, manifest last).
