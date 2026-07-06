# SPEC29 — Joint POS + accentuation model (single pass, shared loss)

## Goal (experimental architecture — the user explicitly wants this shape)

One litlat-bert encoder, one forward pass over a SENTENCE, two heads:

- **POS head**: per word-token UPOS|FEATS classification — same shape as
  the existing tagger head (see local/tagger-hf/head_modeling.py /
  head_config.py; single combined-label head is fine for v1 of this
  experiment).
- **Stress head**: per word-token, hierarchical "choose the letter within
  the token, choose the mark": character queries for THAT word
  (char embedding + within-word position embedding) cross-attend into the
  word's OWN subword hidden states (mask attention to the word's subword
  span), masked softmax over (char × 3 marks) + a no-stress cell — the
  same head design as train_stress_nn.py's StressHead + v3 no-stress,
  applied per token inside a sentence.

Loss = POS loss (all word tokens) + stress loss (tokens with stress
supervision only), simple sum (flag for a weight, default 1.0).

New directory `local/accentuator/joint/` for everything; do not modify
existing files. Scripts must run with `.venv-train/Scripts/python.exe`.

## 1. `build_joint_dataset.py` (CPU, can run while the GPU trains)

Training corpus: MATAS v3.0 with gold morphology (CC BY 4.0; fetch/prep
machinery exists — see local/tagger-hf/fetch_corpora.py + prep_corpus.py
and their cached outputs under local/tagger-hf/data/; reuse the prepared
train/dev jsonl if present rather than re-preparing).

Stress supervision by projection from OUR dictionary (provenance-clean:
no external accent source):

- for each MATAS token: word key = lowercase form (alpha only);
  look up generated.sqlite; convert the token's gold UPOS+FEATS to slots
  via kirciuokle.disambiguate.token_tags-equivalent logic (MATAS jsonl
  stores label strings `UPOS|FEATS` — parse them into the same slot
  space; study how prep_corpus.py formats labels);
- score the word's variants' mi labels against those slots (score_tags);
  if a UNIQUE best variant wins with positive score → its form is the
  stress target (extract (char index, mark) via train_guesser.stress_of);
  ties with different stress or negative score → NO stress supervision
  for that token (stress-loss mask 0);
- foreign-letter tokens (outside the LT alphabet) → no-stress target;
- write train/dev jsonl: {tokens: [{word, pos_label, stress: [idx, mark] | "none" | null}]}
  plus dataset stats: token counts, share with stress supervision
  (expect ~60-80%), homograph-resolved share, label-set size.

## 2. `train_joint.py`

- Warm start: encoder from a released tagger torch checkpoint if one
  exists on disk (search local/tagger-hf artifacts/ and release/ for
  pytorch_model.bin/safetensors; if only ONNX exists, start from
  EMBEDDIA/litlat-bert); stress-head weights from
  data/stress_nn3/stress_nn3.pt when the file exists (else stress_nn2,
  else fresh) — load what matches by shape, report what was warm-started.
- POS head fresh (label set comes from the joint dataset).
- Batching: sentences padded to max 128 subwords; char grids per word
  token (cap word length 30 as elsewhere). bf16 autocast, AdamW with
  encoder/head LR split like train_stress_nn.py.
- Defaults sized for a first experiment, not a campaign: --max-sentences
  120000, --epochs 2, --batch-size 16 sentences. Print step losses for
  both heads separately.
- Checkpoint to local/accentuator/joint/checkpoints/joint_v1.pt.

## 3. `eval_joint.py`

- POS: accuracy of the combined label on the MATAS dev split + on the
  ALKSNIS gold test if the prepared files are on disk (they are, under
  local/tagger-hf/data/) — report next to the released tagger's known
  86-89% slot numbers (local/README.md has the reference table).
- Stress: token exact/position on the LRT audited silver — reuse the
  scoring/alignment/audit machinery from eval_nodict_pipeline.py by
  import (its tagger is NOT needed — the joint model produces its own
  labels; you need corpus sentence iteration + silver alignment + audit
  application). Report next to the nodict pipeline's 81.1% audited exact.
- Also report: single-pass tokens/s on GPU, parameter count vs the
  two-model stack.

## Pass criteria

1. `build_joint_dataset.py --max-sentences 2000` writes train/dev jsonl
   with stats printed; stress-supervision share printed.
2. `train_joint.py --max-sentences 2000 --epochs 1 --batch-size 8`
   completes on GPU OR CPU (if the GPU is busy training v3, use
   CUDA_VISIBLE_DEVICES= and a tiny run — this is a smoke test only).
3. `eval_joint.py --checkpoint ... --limit 300` runs both eval legs on
   smoke data.
4. Report all smoke numbers + dataset stats. Do NOT launch the full
   training run (the human schedules GPU time).

Do not commit.
