# SPEC33 — Score the VDU+UDPipe pipeline on the chrestomatija gold

## Context

`build_silver_truth.py` is being run over the accent-stripped
chrestomatija text (`data/eval/chrestomatija-plain.txt`, one gold
sentence per line) producing
`data/eval/chrestomatija-vdu-silver.jsonl` (token stream in corpus
order: `{"word","accented","mi","ambiguous"}`). Score that pipeline
(our production online solution: VDU kirčiuoklė + UDPipe + morphology
disambiguation) against the gold with EXACTLY the metrics of
eval_chrestomatija.py, so the row is comparable to the joint/liepa/dict
rows already in reports/chrestomatija-eval.md.

One new script `local/accentuator/score_silver_vs_gold.py`; modify
nothing else. It may need to run before the silver file is complete —
handle a missing/partial input with a clear message and nonzero exit.

## Implementation

- Load gold sentences from `data/eval/chrestomatija-gold.jsonl`;
  tokenize each gold sentence with the SAME tokenization
  eval_chrestomatija.py uses (import from it — do not reimplement).
- Load the silver token stream; align to the gold token stream by
  accent-stripped lowercase surface with two cursors (the pattern used
  in eval_nodict_pipeline.py); count and report unaligned tokens
  (must be <1%; abort with a diagnostic if worse).
- Metrics (import scoring helpers from eval_chrestomatija.py):
  token answered (silver token carries any stress mark OR gold is also
  unmarked), token exact, token position, and sentence-level SEQUENCE
  accuracy over gold sentences (every token exact).
- Update `reports/chrestomatija-eval.md`: add a `vdu-udpipe (online)`
  row to the existing table (rewrite the file preserving the other
  rows — read the current report and regenerate, or edit the table in
  place; keep the thesis-reference and samples sections).
- Print the row to stdout too.

## Pass criteria

1. Running before the silver is ready → clear message, nonzero exit.
2. Once `data/eval/chrestomatija-vdu-silver.jsonl` is complete (wait
   for it: poll every 60s until the file line count stops growing for
   3 consecutive polls AND the build process is gone — or up to 90
   minutes), run the scorer: alignment skip <1%, row printed, report
   updated with all four systems.
3. Paste the final four-row table.

Do not commit.
