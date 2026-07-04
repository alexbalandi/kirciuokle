# Self-hosted local replica

This directory contains a local, self-hosted replica of the production
kirčiuoklė: the same frontend, the same `/api/*` contract, a SQLite copy of
the D1 `words` table, and a swappable UDPipe-compatible tagger.

## Run

From the repo root:

```sh
docker compose -f local/docker-compose.yml up --build
```

The app listens on <http://127.0.0.1:8765/> and stores the dictionary at
`local/data/words.sqlite`. Docker is intentionally only an artifact here;
the Python app tests run without Docker.

## Dictionary

`DICT_PATH` defaults to `/data/words.sqlite` in the container. On startup the
app creates the `words` schema from the root `migrations/*.sql` if needed.

Provisioning options:

- Warm naturally: keep `FALLBACK=vdu` and first-time misses are fetched from
  VDU and written to SQLite.
- Preload from production: run the orchestrator-provided
  `scripts/export_dictionary.py` once it exists; it writes
  `local/data/words.sqlite`.
- Stay fully offline: set `FALLBACK=none`. Cache misses become
  `unknown: true`, the miss budget is disabled, and VDU is never called.

The dictionary data is not committed.

## Taggers

The app calls `POST {TAGGER_URL}/process` with the UDPipe REST form fields
`tokenizer`, `tagger`, `model`, and `data`. The default stack builds
`local/tagger-stanza`, a CPU-only Stanza wrapper for Lithuanian ALKSNIS:

```sh
TAGGER_URL=http://tagger:8001
```

Stanza-lt tags slightly below UDPipe 2 mBERT, which is the production tagger.
For closer production parity, use the UDPipe 2 compose file, based on
[`ufal/udpipe2-docker`](https://github.com/ufal/udpipe2-docker):

```sh
docker compose -f local/docker-compose.udpipe2.yml up --build
```

You can also point `TAGGER_URL` at LINDAT or any future service that returns
`{"result": "<conllu>"}` from `/process`.

## Tagger backends & benchmarking

Use `scripts/bench_taggers.py` to compare UDPipe-compatible taggers against
the UD_Lithuanian-ALKSNIS test set:

```sh
uv run scripts/bench_taggers.py --backends lindat --limit 400
uv run --with stanza scripts/bench_taggers.py --backends stanza --limit 400
```

Measured on the 400-sentence gold test set (2026-07). `slots` is the
metric that matters for accentuation: exact match of the scoring
projection (POS family + case/gender/number/tense/person/voice/degree)
that drives homograph disambiguation. `aux/v` is the AUX-vs-VERB
distinction that powers the yra→yrà lemma exception.

Note: the `slots` metric merges DET into PRON per VDU conventions (see
docs/SPEC13.md); UDPipe's number is unchanged by the merge. The `lemma`
and `feats` columns do not apply to the fine-tuned model (it emits
form-as-lemma and slots-only features by design).

| backend | upos | lemma | feats | **slots** | aux/v | tok/s |
|---|---|---|---|---|---|---|
| **fine-tuned litlat-bert (ONNX INT8, local CPU)** | 90.5% | n/a | n/a | **92.1%** | **97.4%** | **882** |
| `lindat` (UDPipe 2 mBERT — prod) | 94.3% | 91.7% | 88.4% | **89.0%** | 96.4% | 637 (network) |
| `stanza` (lt, local CPU) | 90.6% | 90.3% | 84.3% | **84.7%** | 94.4% | 425 |
| `trankit` (XLM-R) | — | — | — | — | — | not viable |

The fine-tuned model (recipe in `local/tagger-hf/`, trained on
XPOS-repaired + teacher-filled MATAS v3.0 + ALKSNIS with VDU-normalized
labels) beats the production tagger by 3.1pp on the accentuation metric
and resolves copular *yra → būti* natively. Its UPOS number is lower
against raw treebank gold mostly due to residual convention differences
that the slots projection is designed to absorb.

Trankit verdict: the 2021 codebase no longer installs against current
Python/transformers (needs `--python 3.10 --with "transformers==4.30.2"
--with "numpy<2" --with six` just to import), and its model host
(`nlp.uoregon.edu`) was unreachable when tested — treat it as abandoned.

The path to beating UDPipe 2 on CPU is `local/tagger-hf/`: fine-tune
`VSSA-SDSA/LT-MLKM-modernBERT` (Apache-2.0 Lithuanian ModernBERT, native
LT tokenizer) on ALKSNIS joint UPOS+FEATS labels, export to ONNX INT8 via
optimum, and serve it through the same UDPipe REST contract. Start with
`local/tagger-hf/README.md` for the prep, fine-tune, export, and sidecar
recipe; gate any switch on this benchmark plus the repo's accentuation
parity eval.

## Local checks

```sh
uv run --project local/app pytest local/app/tests
npm run check
```
