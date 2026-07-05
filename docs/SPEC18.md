# SPEC18 — Multi-backend guess-tier artifact builder

## Goal

`local/accentuator/guess_uncovered.py` currently builds the lowest-confidence
guess tier (`data/guesses.sqlite`) exclusively with the LIEPA
`phonology_engine`. We now have two additional home-grown guessers trained on
our own dictionary — Anbinderis-style letter rules and a litlat-bert neural
stresser — and a benchmark deciding which becomes the preferred backend.
Upgrade `guess_uncovered.py` so the artifact can be built with any backend or
a cascade, with per-word provenance recording which backend answered.

Modify ONLY `local/accentuator/guess_uncovered.py`. Do not reformat existing
code, do not touch other files.

## CLI (new arguments; existing behavior with no args must stay identical)

- `--backend NAME`, default `liepa`. Accepted values:
  - `liepa` — current behavior (phonology_engine).
  - `anbinderis` — end-bgn letter rules trained at startup from the
    generated dictionary.
  - `nn` — the litlat-bert checkpoint.
  - Cascades `anbinderis+liepa`, `nn+liepa`, `anbinderis+nn+liepa`: try each
    stage in order; the first stage that answers a word provides its form.
- `--min-confidence FLOAT`, default `0.0` — nn answers below this softmax
  confidence count as "no answer" (so cascades fall through).
- `--limit N`, default none — cap the candidate list (after sorting) for
  smoke runs.

## Verified API contracts (all modules are siblings in local/accentuator/)

- Anbinderis backend:
  ```python
  from train_guesser import load_training
  from anbinderis_rules import AnbinderisModel
  from _common import DEFAULT_GENERATED
  model = AnbinderisModel(load_training(DEFAULT_GENERATED))  # ~60s build
  form_or_none = model.predict_form(word)  # accented str | None
  ```
- NN backend (torch only exists in the training venv, see Runtime):
  ```python
  import torch
  from transformers import AutoModel, AutoTokenizer
  from train_stress_nn import ENCODER, MAX_CHARS, OUT_DIR, StressModel, batch_predict
  ckpt = torch.load(OUT_DIR / "stress_nn.pt", map_location="cpu", weights_only=False)
  tokenizer = AutoTokenizer.from_pretrained(ckpt.get("encoder", ENCODER))
  model = StressModel(AutoModel.from_pretrained(ckpt.get("encoder", ENCODER)),
                      len(ckpt["char_vocab"]) + 2)
  model.load_state_dict(ckpt["state_dict"])
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  preds = batch_predict(model, tokenizer, ckpt["char_vocab"], words, device)
  # -> list aligned with words: (accented_form, confidence) | None
  # words longer than MAX_CHARS must be answered None (batch_predict handles it,
  # but guard len(word) <= MAX_CHARS anyway)
  ```
  The checkpoint `local/accentuator/data/stress_nn/stress_nn.pt` already
  exists (smoke-quality; a full training run may overwrite it later — do not
  care about its quality, only the code path).
- LIEPA backend: keep the existing `engine_accent(pe, word)` code.

## Provenance and schema

Output schema (9-column `words` table) and the `info: "spėjimas"` label stay
exactly as they are. Provenance encodes the ANSWERING backend:

- liepa answer:      `open-accentuator:liepa-guess:{word}` (unchanged)
- anbinderis answer: `open-accentuator:anbinderis-guess:{word}`
- nn answer:         `open-accentuator:nn-guess:{word}:conf={conf:.3f}`

Candidate-set logic (lt_50k ∪ VDU keys, minus generated-dictionary words,
`.isalpha()`) stays unchanged. The final print must also report per-backend
answer counts when a cascade runs, e.g.
`guessed 24,092 (anbinderis 14,000 + liepa 10,092) of 24,461 ...`.

## Runtime constraints

- Keep the PEP 723 header exactly as-is (uv script mode, dependency
  `phonology_engine` only). torch/transformers must be imported lazily inside
  the nn code path only; when the nn backend is requested and torch is
  missing, exit with a one-line actionable error naming
  `.venv-train/Scripts/python.exe`.
- The `liepa` import (`from phonology_engine import PhonologyEngine`) must
  also become lazy so `--backend anbinderis` works in an environment without
  phonology_engine.

## Pass criteria (run all, from the repo root)

1. `uv run local/accentuator/guess_uncovered.py --backend anbinderis --limit 500 --output local/accentuator/data/guesses-smoke.sqlite`
   → writes an artifact; every provenance starts with
   `open-accentuator:anbinderis-guess:`; some candidates may be unanswered
   (abstention) — answered count < limit is expected and fine.
2. `uv run local/accentuator/guess_uncovered.py --limit 300 --output local/accentuator/data/guesses-smoke.sqlite`
   → default liepa path still works, provenance unchanged
   (`open-accentuator:liepa-guess:`).
3. `set CUDA_VISIBLE_DEVICES= && .venv-train/Scripts/python.exe local/accentuator/guess_uncovered.py --backend nn+liepa --limit 200 --min-confidence 0.5 --output local/accentuator/data/guesses-smoke.sqlite`
   (CPU inference; a GPU training run is in progress — do NOT use CUDA)
   → mixed provenances possible; per-backend counts printed.
4. `uv run local/accentuator/guess_uncovered.py --backend nn --limit 10 --output local/accentuator/data/guesses-smoke.sqlite`
   → clean one-line error mentioning `.venv-train` (torch missing under uv),
   nonzero exit.
5. Delete `local/accentuator/data/guesses-smoke.sqlite` afterwards.
6. `uv run local/accentuator/selfcheck.py` still passes (nothing else changed).

Do not start servers. Do not run the full (no --limit) generation. Do not
commit; leave the working tree for review.
