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

## Local checks

```sh
uv run --project local/app pytest local/app/tests
npm run check
```
