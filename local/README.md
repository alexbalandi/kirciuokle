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
- Preload from production: `uv run scripts/export_dictionary.py` writes
  `local/data/words.sqlite` from the production D1.
- Stay fully offline: set `FALLBACK=none`. Cache misses become
  `unknown: true`, the miss budget is disabled, and VDU is never called.

The dictionary data is not committed.

## Taggers

The app calls `POST {TAGGER_URL}/process` with the UDPipe REST form fields
`tokenizer`, `tagger`, `model`, and `data`. Options, best first:

1. **Our released models** (see below) served by the sidecar in
   `local/tagger-hf/` — highest quality, fully local, permissive license.
2. `local/tagger-stanza` — the default compose stack; CPU-only Stanza
   wrapper, simplest to build, measurably weaker.
3. `docker compose -f local/docker-compose.udpipe2.yml up --build` — the
   official [`ufal/udpipe2-docker`](https://github.com/ufal/udpipe2-docker)
   (exact production-tagger parity; models CC BY-NC-SA).
4. Any URL returning `{"result": "<conllu>"}` from `/process` (e.g. the
   public LINDAT service).

## Released models

Three fine-tuned litlat-bert taggers, CC BY-SA 4.0, fully NC-free lineage
(gold MATAS v3.0 + ALKSNIS + constrained-decoding self-training; recipe in
`local/tagger-hf/README.md`):

| model | best for | emits |
|---|---|---|
| [litlat-bert-lithuanian-morphology](https://huggingface.co/alexbalandi/litlat-bert-lithuanian-morphology) | highest UPOS/slots accuracy, UD categories | UPOS (incl. DET/AUX) + core FEATS |
| [litlat-bert-lithuanian-morphology-full](https://huggingface.co/alexbalandi/litlat-bert-lithuanian-morphology-full) | complete annotations | full UD FEATS + lemmas |
| [litlat-bert-lithuanian-morphology-vdu](https://huggingface.co/alexbalandi/litlat-bert-lithuanian-morphology-vdu) | accentuation pipelines | traditional-grammar categories (DET→PRON, AUX→VERB) |

## Benchmarks

All numbers one-shot on the **full 684-sentence** UD_Lithuanian-ALKSNIS
gold test (2026-07). `slots` = exact match of the agreement-feature
projection (POS family + Case/Gender/Number/Tense/Person/Voice/Degree)
that drives homograph disambiguation; DET/PRON and AUX/VERB merge inside
the projection (docs/SPEC13.md). `lemma`/`feats` are n/a where a model
does not emit them by design.

| backend | upos | lemma | feats | **slots** | tok/s |
|---|---|---|---|---|---|
| `-ud` (ONNX INT8, local CPU) | 92.5% | n/a | n/a | **89.1%** | 874 |
| `-full` (ONNX INT8, local CPU) | 91.4% | **94.2%** | 80.3% | 86.3% | 921 |
| `-vdu` (accentuation flavor) | 90.6% | n/a | n/a | 86.4% | 880 |
| UDPipe 2 (prod, via network) | 95.1% | 92.5% | 88.7% | **89.2%** | 605 |
| Stanza-lt (local CPU)¹ | 90.6% | 90.3% | 84.3% | 84.7% | 425 |
| Trankit (XLM-R) | — | — | — | — | not viable² |

Official CoNLL-18 (gold tokenization): `-ud` UPOS 94.0; `-full` UPOS 93.2 /
UFeats 84.1 / **Lemmas 94.7 (vs UDPipe 92.9)**; UDPipe reference UPOS 95.2 /
UFeats 89.1. Accentuation quality (`-vdu`, agreement of homograph stress
choices with production): 20/20 on the sample text, 323/370 on the
Wikipedia corpus — equal to the internal UDPipe-taught reference.

¹ Stanza measured on the 400-sentence subset.
² The 2021 Trankit codebase no longer installs against current
Python/transformers and its model host (`nlp.uoregon.edu`) was unreachable
when tested — treat it as abandoned.

Reproduce with `uv run scripts/bench_taggers.py --backends lindat
[--lindat-url http://127.0.0.1:8001/process] --limit 684` and
`uv run local/tagger-hf/eval_conll18.py --tagger-url …`. Encoder verdict
from the bake-off: EMBEDDIA/litlat-bert beat the Lithuanian ModernBERT by
5.4pp on identical data — measure, don't assume. Gate any tagger change on
this benchmark plus the repo's accentuation parity eval.

## Open accentuator (experimental, not wired into serving)

`local/accentuator/` builds an accentuation dictionary from open sources
(Wiktionary accent classes + a paradigm engine implementing published
accentology; plan in `docs/PLAN-open-accentuator.md`). Standalone
artifacts only — the replica's dictionary chain is unchanged. Full
architecture, module inventory, rule sources, and QA methodology:
`docs/ACCENTUATOR.md`. 566k-word dictionary; parity vs the VDU cache
(10,015 words): 73.6% covered, 81.6% of covered exact, **zero
unadjudicated disagreements** —
every divergence is repaired by a published rule (notation
normalization; future-tense metatony; verb stress retraction,
participle constraints and declension per Kushnir 2019; self-accented
suffix rules per the VDU 2010 monograph, see `ATTRIBUTIONS/`), excluded
with a reason in `local/accentuator/parity_vetoes.json`, or recorded as
a NORM-DELTA where the VLKK names database backs our form against the
cache.

## Local checks

```sh
uv run --project local/app pytest local/app/tests
npm run check
```
