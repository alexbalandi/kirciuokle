# SPEC32 — Chrestomatija gold benchmark (extraction + eval)

## Context

`local/accentuator/data/eval/chrestomatija.pdf` (on disk, 200 pages,
gitignored) is "Kirčiuotų tekstų chrestomatija" (V. Kavaliauskas, 2014) —
50 hand-stressed Lithuanian literary texts, the de-facto community gold
benchmark: the 2026 VU thesis reports sequence-accuracy 0.711 (their
transformer) and 0.702 (VDU Kirčiuoklis) on 2,303 samples from it.
Recovered from the Wayback Machine
(`http://web.archive.org/web/20220120104613id_/http://www.esparama.lt/documents/10157/490675/2014_Kirciuotu_tekstu_chrestomatija_mok_knyga.pdf`)
— record this URL in the extractor docstring. The book is copyrighted
teaching material: the extracted text stays in the gitignored data dir,
only aggregate numbers are committed.

Two new scripts under `local/accentuator/`; modify nothing else.

## 1. `extract_chrestomatija.py`

PDF → `data/eval/chrestomatija-gold.jsonl` (one sentence per line:
`{"text": "<accented sentence>", "page": N}`).

Extraction rules (pypdf; run under uv with `--with pypdf`):
- Skip front matter and apparatus: keep only pages whose ACCENT DENSITY
  is high — compute combining-mark count (U+0300/0301/0303 after NFD)
  per 100 letters per page; keep pages above a threshold you calibrate
  (print the per-page density histogram so the human can review the
  cutoff; front matter/intro pages discuss accentuation and contain
  SOME marks — the threshold plus a minimum-page-number floor should
  drop them; the intro ends within the first ~10 pages).
- Fix hyphenation across line breaks: `trupu-\ntį` → `truputį`
  (join when a line ends with `-` and the next starts lowercase;
  preserve the accent marks through the join).
- Drop lines that are ALL-CAPS titles, bare page numbers, author
  names/dates lines (heuristic: very short lines with no stress marks),
  and footnote-like lines starting with digits.
- Sentence-split on [.!?…] + whitespace + capital/quote; keep sentences
  with ≥3 word tokens and ≥1 stress mark.
- NFC-normalize; collapse whitespace.
- Print: kept pages, dropped pages, sentence count, token count, total
  stress marks, and 5 random sample sentences for human QA.
- Target: expect a few thousand sentences (the thesis got 2,303 samples
  — same order of magnitude; report what you get).

## 2. `eval_chrestomatija.py`

Score our systems against the gold. Reuse existing machinery by import —
alignment/scoring/tokenization from `eval_nodict_pipeline.py` (silver
loader replaced by the gold jsonl: gold accented form per token comes
from the gold sentence itself; plain word = strip_accents of it).

Systems to score (each optional/graceful-skip like bench_guessers):
1. `joint` — the joint model checkpoint (default
   `joint/checkpoints/joint_v1_polish.best.pt`), inference per sentence
   exactly as `joint/eval_joint.py` does (import from there).
2. `dict` — dictionary + joint model's OWN predicted labels for variant
   selection is too entangled; instead use the dictionary with default
   forms only (report it as `dict-default`) — a floor baseline.
3. `liepa` — phonology_engine per distinct word.

Metrics per system:
- token-level: answered / exact / position (same definitions as
  everywhere; foreign-unmarked convention: a gold token WITHOUT any
  stress mark counts exact when the system also leaves it unmarked or
  abstains).
- sentence-level SEQUENCE accuracy: a sentence counts correct iff EVERY
  word token's form matches gold exactly — this is the number comparable
  to the thesis's 0.711/0.702 (note in the report: tokenization and
  normalization protocols may differ from theirs; ours is a
  reimplementation, so treat cross-paper comparison as indicative).
- Write `reports/chrestomatija-eval.md` with the table, the thesis
  reference numbers for context, sample disagreements (15), and the
  extraction stats.

## Pass criteria

1. Extraction runs; per-page density histogram printed; ≥1,500 sentences
   kept; 5 samples visibly clean accented literary text (paste them).
2. `eval_chrestomatija.py --limit 200` (sentence cap) runs the full
   pipeline on GPU or CPU and prints all three system rows.
3. Full run: launch WITHOUT --limit, report the final table.
4. `git status` — only the two new scripts + report are new; the jsonl
   and pdf stay untracked (data dir is gitignored).

GPU note: a training run may or may not be active — check
`nvidia-smi` before using CUDA; if >6GB is in use, run on CPU.

Do not commit.
