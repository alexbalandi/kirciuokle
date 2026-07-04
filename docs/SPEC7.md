# Phase 7 — self-hosted local replica (`local/`)

A docker-compose artifact that replicates the production functionality
fully locally: same UI, same API contract, same accentuation pipeline,
no Cloudflare, no calls to VDU or LINDAT required once the dictionary is
warm. Not deployed anywhere — it is a reviewable, runnable artifact.

## Architecture

```
local/
  docker-compose.yml            # app + tagger-stanza (default stack)
  docker-compose.udpipe2.yml    # optional override: ufal/udpipe2-docker for
                                # exact prod tagger parity (documented, thin)
  app/
    Dockerfile                  # multi-stage: node builds ../dist frontend →
                                # python:3.12-slim runtime
    pyproject.toml              # fastapi, uvicorn, httpx (uv-compatible)
    kirciuokle/
      server.py                 # FastAPI app: /api/accent, /api/word, static
      accent.py                 # tokenization + local-first pipeline
      disambiguate.py           # MI tags, scoring, alignment, lemma exceptions
      dictionary.py             # sqlite store, read-through, optional VDU fallback
      vdu.py                    # nonce + text_accents/word_accent client
      tagger.py                 # UDPipe-REST client + CoNLL-U parsing
    tests/                      # pytest, no network, no docker needed
  tagger-stanza/
    Dockerfile
    server.py                   # Stanza(lt) exposing the UDPipe REST contract
  README.md                     # run, configure, provision dictionary, taggers
scripts/export_dictionary.py    # production D1 → local/data/words.sqlite
```

## The app service (Python 3.12, FastAPI)

**Port the pipeline from the TypeScript worker — it is the
parity-authoritative implementation.** Sources to port faithfully:
`src/worker/localAccent.ts` (tokenization: `[\p{L}\p{M}]+` tokens,
ABBREVIATIONS set + dotted-abbreviation rule, uppercase-Roman-numeral rule,
LT-alphabet NON_LT check, case-sensitive lower/title dictionary sides,
matchCase, miss budget), `src/worker/disambiguation.ts` (MI_TAGS, parseMi
longest-first, tokenTags incl. PART_VERB/PROPN/SCONJ merges and Degree=Pos
drop, scoring weights +4/−3/+2/−2 with strict-winner rule, LEMMA_EXCEPTIONS,
alignment with letter-token filtering), `src/worker/vdu.ts` (nonce scrape +
retry, `message: false` negative, chunking) — `scripts/accent_text.py`
already contains Python ports of several of these to crib from.

- **API contract**: identical to production —
  `POST /api/accent` `{text}` → `{parts, tagger, source}` (same Part shape:
  text/accented/type/ambiguous/unknown/variants/chosen/resolvedBy),
  `GET /api/word?w=`, same 400/413/502 error mapping. `source` is `"local"`
  (or `"vdu"` when the miss-budget fallback ran). The existing frontend is
  served as-is from `dist/client` (env `STATIC_DIR`, default `./static`);
  it must work against this server unchanged.
- **Dictionary**: SQLite file (env `DICT_PATH`, default
  `/data/words.sqlite`), exact same `words` schema as the D1 migrations
  (apply `migrations/*.sql` on startup if the file/tables are missing).
  Read-through behavior controlled by env `FALLBACK`:
  - `vdu` (default): misses fetched from VDU (3-call fetchWordEntry port),
    stored, self-warming — same semantics and MISS_BUDGET=10 as prod,
    including full fallback to the VDU text_accents path when over budget;
  - `none`: pure offline — misses become `unknown: true` parts, never over
    budget, no network to VDU at all.
- **Tagger**: env `TAGGER_URL` (default `http://tagger:8001`). The client
  speaks the UDPipe REST protocol: `POST {TAGGER_URL}/process` with form
  fields `tokenizer`, `tagger`, `model`, `data` → `{"result": "<conllu>"}`.
  10 s timeout; on failure degrade exactly like prod
  (`tagger: "unavailable"`, defaults kept). This makes the tagger swappable:
  the stanza sidecar, ufal's udpipe2-docker, or LINDAT's public endpoint are
  all drop-in values of `TAGGER_URL`.

## The tagger-stanza service

Small FastAPI wrapper around **Stanza** with the `lt` (ALKSNIS) model,
exposing the same `/process` contract (tokenize+tag, output CoNLL-U with
lemma, UPOS, XPOS, FEATS — columns the app actually consumes). Model is
downloaded at image build time so the container runs offline. CPU-only.

Honest quality note for the README: Stanza-lt tags slightly below UDPipe 2
mBERT (the prod tagger). For exact prod parity use
`docker-compose.udpipe2.yml` (official ufal/udpipe2-docker, heavier), or
point `TAGGER_URL` at LINDAT. Better-than-prod candidates (Trankit XLM-R, a
LitLat-BERT fine-tune) plug into the same contract later — the repo's eval
harness is the gate for choosing.

## Dictionary provisioning

`scripts/export_dictionary.py` (repo root, uv script like the others):
reads `.env` credentials, pages through the production D1 (`SELECT * FROM
words LIMIT/OFFSET` via the CF REST API), writes `local/data/words.sqlite`
with the same schema. Idempotent (recreates the file). The local dictionary
is intentionally NOT committed to the repo (data licensing).

## Quality bar

- `local/app` tests (pytest, offline, temp sqlite, tagger mocked or
  `FALLBACK=none`): port the parity-critical fixtures from the worker
  tests — alyta/Alyta case sides, vilnius MULTIPLE_VARIANT vs Vilnius
  MULTIPLE_MEANING, kas suppressed-reading ONE, abbreviations (`m.`,
  `rus.`, initial `V.`, `XX a.` Roman), pre-accented `mė́nuo` untouched,
  yra→yrà lemma exception with a mocked CoNLL-U tagger response, miss with
  `FALLBACK=none` → unknown, NFC output, `/api/accent` error mapping.
- Tests must be runnable on Windows via
  `uv run --project local/app pytest local/app/tests` (no docker).
- Dockerfiles and compose files must be syntactically complete and coherent
  (multi-stage app build compiles the frontend with npm), but docker is NOT
  available in this environment — do not attempt to build or run images.
- Root README: short "Self-hosting locally" section pointing at
  `local/README.md`. Do not modify `scripts/` (except you may NOT — the
  orchestrator writes export_dictionary.py) or `docs/`.
- `npm run check` must still pass (untouched TS should be unaffected).
