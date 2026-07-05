# SPEC21 — Live-guess evaluation on unseen text (LRT corpus)

## Goal

We are moving the guess tier from precomputed-only toward LIVE guessing of
out-of-dictionary words. Before wiring anything into serving we need an
evaluation pipeline on text we have never seen: fresh LRT (lrt.lt) news
articles, with silver ground truth produced by the validated production
pipeline (`scripts/accent_text.py` = VDU kirčiuoklė + UDPipe tagging +
morphology-scored disambiguation).

Create THREE new scripts under `local/accentuator/`; do not modify existing
files. All corpus/derived data goes under `local/accentuator/data/eval/`
(the data dir is gitignored).

## 1. `fetch_lrt_corpus.py`

- Fetch recent Lithuanian-language articles from lrt.lt. Discover the best
  machine access yourself and verify it works before settling: candidates
  are the RSS feeds (https://www.lrt.lt/rss and section feeds) and the JSON
  API used by their site search (try
  `https://www.lrt.lt/api/search?...` / `https://www.lrt.lt/api/json/...`
  endpoints; inspect what the site actually calls if needed). Use a normal
  browser User-Agent; throttle to ≥1s between requests.
- CLI: `--articles N` (default 40), `--output` (default
  `data/eval/lrt-corpus.txt`). Extract clean paragraph text only (no HTML,
  no captions/bylines); skip non-Lithuanian items. Write one paragraph per
  line, blank line between articles, plus a sidecar
  `lrt-corpus.meta.json` listing article URLs + fetch date (attribution).
- Target size: 40 articles ≈ 25–40k tokens. Print token/word-type counts.

## 2. `build_silver_truth.py`

- Input: the corpus txt. Output: `data/eval/lrt-silver.jsonl`, one JSON per
  token: `{"word": plain_lower, "accented": form, "mi": winning_label_or_null,
  "ambiguous": bool}` in corpus order.
- Reuse `scripts/accent_text.py` MACHINERY BY IMPORT (sys.path append the
  scripts dir) — do not reimplement the VDU client, chunking, or the
  disambiguation scoring. Look at how eval_accenter.py imports and drives
  accent_text for the pattern. The pipeline is async httpx; keep its
  chunking (CHUNK_LIMIT) and add: resumability (skip chunks already in the
  output; flush after each chunk) and throttling (≥1s between VDU calls).
  External endpoints: kalbu.vdu.lt (nonce + ajax-call) and LINDAT UDPipe —
  both are called by accent_text already; do not add new endpoints.
- Only word tokens (the WORD_RE / tokenize approach in eval_accenter.py) go
  to the JSONL; punctuation/numbers are skipped.

## 3. `eval_live_guess.py`

- Inputs: `data/eval/lrt-silver.jsonl`, `data/generated.sqlite`,
  `data/guesses.sqlite`, plus the live backends from `guess_uncovered.py`
  (import LiepaBackend / NNBackend / AgreementBackend and build the
  `nn&liepa+liepa` cascade via build_backends; requires the .venv-train
  interpreter for the nn part — degrade to liepa-only with a warning if
  torch is missing).
- For each silver token bucket the word into TIERS:
  1. `dict` — word key in generated.sqlite,
  2. `precomputed-guess` — in guesses.sqlite,
  3. `live-guess` — neither: run the live cascade on the word (deduplicate:
     run each distinct word once, reuse per token),
  4. `unanswered` — live cascade abstains.
- Score each answered tier against silver: token-level and type-level
  `exact` (NFC form match vs silver accented) and `position` (same
  stressed-letter index; reuse the stress_of approach from
  train_guesser.py). For the `dict` tier, when a word has multiple variants
  pick the variant whose mi label best matches the silver `mi` (simple
  exact-label match, else default form) — this mirrors serving.
- Report to stdout AND `reports/live-guess-eval.md`: corpus size, per-tier
  token/type counts (= real-world OOV rate of the dictionary on fresh
  news), per-tier exact/position, and 20 sample live-guess disagreements
  (`word: live=X silver=Y`).

## Pass criteria (repo root, in order)

1. `uv run local/accentuator/fetch_lrt_corpus.py --articles 6 --output local/accentuator/data/eval/lrt-smoke.txt`
   → ≥ 3,000 tokens of clean Lithuanian text; meta sidecar written.
2. `uv run local/accentuator/build_silver_truth.py --input local/accentuator/data/eval/lrt-smoke.txt --output local/accentuator/data/eval/lrt-smoke-silver.jsonl`
   → JSONL lines ≥ 2,500; spot-print 5 lines showing accented forms with
   combining marks and mi labels. Interrupt+rerun must resume, not restart.
3. `.venv-train/Scripts/python.exe local/accentuator/eval_live_guess.py --silver local/accentuator/data/eval/lrt-smoke-silver.jsonl`
   → report written with all four tiers present (live-guess tier may be
   small on a smoke corpus; that is fine).
4. Do NOT run the full 40-article fetch — the human will run it after
   review to keep external load deliberate.

Constraints: throttle everything ≥1s; no parallel hammering of VDU/LINDAT;
no new external endpoints beyond lrt.lt + the two accent_text already uses.
Do not commit.
