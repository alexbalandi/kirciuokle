# SPEC24 — No-dictionary accentuation pipeline + unseen-text eval

## Context (read first)

The project's target architecture is a dictionary-FREE accentuator: our
own tagger provides contextual morphology, and the POS-conditioned stress
model (SPEC22, checkpoint `data/stress_nn2/stress_nn2.pt`) accents every
word from (word, label) alone. The dictionary pipeline remains only as a
comparison baseline. This spec builds the end-to-end no-dict pipeline and
evaluates it on the unseen LRT corpus against the silver truth from
SPEC21 (`build_silver_truth.py` output).

New script: `local/accentuator/eval_nodict_pipeline.py`. Do not modify
existing files.

## The label bridge (verified contracts)

The stress model was trained on dictionary label strings (VDU-style, e.g.
`dkt., mot. g., vns. įn.`). The tagger emits CoNLL-U. The bridge selects,
per token, the best-matching label from the CLOSED vocabulary of
dictionary labels:

- `local/app/kirciuokle/disambiguate.py` provides everything:
  `parse_mi(label) -> dict[Slot, str]` (label string → slots),
  `token_tags(token) -> dict[Slot, str]` (CoNLL-U token → slots, incl.
  the DET→PRON / VerbForm=Part conventions), and
  `score_tags(variant_slots, context_slots) -> int`. Import them
  (sys.path append `local/app`; check how kirciuokle's own modules import
  `Token` — reuse its CoNLL-U parsing if importable, else parse CoNLL-U
  minimally yourself: columns FORM, UPOS, FEATS).
- Label vocabulary: distinct label strings over all variants in
  `generated.sqlite` (`mi` lists, fall back to `info`), with their
  `parse_mi` slots precomputed once. Expect a few hundred distinct labels.
- Per token: `label = argmax over vocab of score_tags(parse_mi(label),
  token_tags(token))`; on ties prefer the label with MORE filled slots;
  if the best score is <= 0, use the empty label "" (unconditioned).

## The tagger (local, ours)

- Serve with `local/tagger-hf/server.py` (UDPipe-compatible
  `POST /process`, form fields per local/README.md). Use the -vdu flavor
  model: try `local/tagger-hf/release/hf-vdu` as `--model-dir` (inspect
  the dir for the ONNX file and pass `--onnx-file` if needed; if that dir
  lacks ONNX, use `local/tagger-hf/artifacts/litlat-gen2-onnx/int8`).
  Verify empirically by tagging one Lithuanian sentence before the run.
- The eval script must START the server itself as a subprocess (pick a
  free port, wait for readiness, ALWAYS terminate it on exit — including
  on exceptions). No orphan servers. `--tagger-url` flag can override to
  skip the subprocess (for reuse).

## Pipelines to score (all against the SPEC21 silver JSONL)

Alignment: the silver JSONL is in corpus token order (accent_text
tokenization). Tag the SAME corpus text sentence-by-sentence; align
tagger tokens to silver tokens by exact surface match on the
accent-stripped lowercase word, advancing two cursors; count and skip
unalignable tokens (report the skip rate — it must stay under 5%).

1. `nodict` (HEADLINE): stress_nn2 with the bridged tagger label, for
   EVERY word token. Load the checkpoint exactly as bench_guessers.py's
   candidate_nn does, but with labels threaded through `batch_predict`'s
   labels argument (SPEC22 added it). Report at confidence 0 and 0.9.
2. `nodict-uncond`: stress_nn2 with empty labels (ablation: what the
   tagger contributes).
3. `liepa`: phonology_engine per distinct word (comparison).
4. `dict`: dictionary pipeline — word in generated.sqlite → variant whose
   mi best matches the bridged label (score_tags), else default form;
   words not covered → unanswered. (Pure dict: NO guess tiers here.)

Metrics per pipeline: token-level answered / exact / position vs the
silver accented form (NFC compare; stress_of from train_guesser for
position). Also type-level exact. Write `reports/nodict-eval.md` with the
table plus 25 sample nodict disagreements (word, silver, nodict, label
used) and the alignment skip rate.

## Pass criteria

1. `.venv-train/Scripts/python.exe local/accentuator/eval_nodict_pipeline.py --corpus local/accentuator/data/eval/lrt-smoke.txt --silver local/accentuator/data/eval/lrt-smoke-silver.jsonl`
   completes: tagger subprocess starts and is terminated (verify no
   python/uvicorn orphans after exit), all four pipeline rows print,
   report written. The v2 checkpoint on disk may be smoke-quality — code
   path correctness is what is being tested, numbers may be low.
2. Alignment skip rate printed and < 5%.
3. Rerun with `--tagger-url` pointing at a manually started server also
   works (subprocess skipped). Then stop that server.
4. Report the four pipeline rows in your final message.

Do not commit. Do not run the full 40-article corpus.
