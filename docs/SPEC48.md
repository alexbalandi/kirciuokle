# SPEC48 — Tender vocabulary prune of joint_v3 (for the site bundle)

## Goal

Prune unused litlat-bert vocabulary rows from
`joint/checkpoints/joint_v3.best.pt` and produce a smaller site bundle —
with a gauntlet of guards; ANY gate failing means we ship nothing and
keep the unpruned model. New scripts under `local/accentuator/joint/`:
`prune_vocab.py`, plus reuse of export/eval tooling. Outputs land in
`joint/pruned/` (never overwrite existing artifacts).

## Retention set (generous by design)

Union of pieces observed when tokenizing:
- all 574k dictionary words (plain forms of every variant),
- every corpus on disk: LRT, wikipedia, literary 1+2, chrestomatija
  plain, MATAS train/dev text (reuse the prepared jsonl),
plus unconditionally: all special tokens, ALL pieces of character
length ≤2, and the `▁` word-boundary piece family for those.
Report: retained count, dropped count, retained share of embedding
params, and the 20 highest-SP-score dropped pieces (eyeball check).

## Mechanics

- HF fast tokenizer (tokenizer.json, Unigram): filter the vocab list
  to retained pieces PRESERVING relative order and scores; specials
  keep their roles. Save as a new tokenizer dir.
- Embedding slice: new_row[new_id] = old_row[old_id] built from the
  same mapping; patch config vocab_size; save pruned torch checkpoint
  (same dict format as joint_v3.best.pt + `"pruned_vocab": mapping
  metadata`).
- Then: ONNX export via export_joint_onnx.py --checkpoint <pruned>
  (fp32 + BOTH int8 recipes: the parity-stable partial scope AND full
  dynamic quantization — gates below decide which int8 ships).

## The gauntlet (in order; abort on first failure)

1. REMAP SANITY: 50 LRT-smoke sentences through original vs pruned
   torch — POS and stress argmax must match on 100% of tokens whose
   segmentation is identical. Any mismatch = abort loudly.
2. SEGMENTATION CENSUS: tokenize held-out text (full LRT corpus +
   chrestomatija plain + a 20k-token wiki slice) with both tokenizers:
   ≥99.9% of word tokens identical segmentation; write every
   re-segmented token to joint/pruned/resegmented.txt with counts.
3. BENCHMARK GAUNTLET (pruned torch, GPU fine): run
   eval_chrestomatija (joint row only) and eval_joint (ALKSNIS + LRT)
   with the pruned checkpoint — REQUIRED: chrestomatija token exact
   ≥ 90.5%, audited LRT ≥ 91.6%, ALKSNIS label ≥ 88.7% (i.e. within
   ~0.2pp of joint_v3's 90.7/91.8/88.95).
4. FOREIGN TORTURE: from the LRT eval's audit diagnostics, the
   foreign-unmarked desired-behavior rate must be within 1pp of the
   unpruned run on the same command; additionally diff the model
   outputs on every token from resegmented.txt and report
   changed-output count.
5. ONNX GATES: fp32 ≥99.5% / int8 ≥98% token agreement vs pruned
   torch (both int8 recipes measured; ship the smallest one that
   passes).
6. SIZES: report a table — torch, onnx fp32, both int8s, tokenizer —
   before vs after.

## Site bundle (only if all gates pass)

Regenerate the site's local-model bundle from the pruned artifacts
(bundled_weights_pilot/prepare_model.py is the current generator —
point it at the pruned ONNX + pruned tokenizer via flags; add them if
missing). The label bridge is tokenizer-independent (label ids
unchanged) — assert label count 804 unchanged. Update manifest sha256s
and the size shown in the consent string (it reads from the manifest).
Do NOT delete the old bundle; keep both dirs, site config constant
decides.

## Pass criteria

Every gauntlet gate's numbers pasted in order; the final size table;
statement of which int8 recipe shipped; the site smoke (Playwright:
Local mode loads the pruned bundle, accentuates the repro paragraph,
consent string shows the new size). Do not commit, do not deploy.
