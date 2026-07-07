# Web app runtime

How the deployed site behaves at request time (as opposed to how the models were
built — that's [ARCHITECTURES.md](ARCHITECTURES.md)).

Two accentuation modes, user-chosen:

- **Web mode** (`POST /api/accent`) — the Cloudflare Worker accentuates from the D1
  dictionary, hitting external services only on cache misses. Fast, no download.
- **Local mode** — an in-browser ONNX model; nothing leaves the device.

Both deployments (`kirciuokle`, `kirciuokle-dev`) run `ACCENT_SOURCE: "local"`, i.e.
Web mode is dictionary-first (`src/worker/localAccent.ts`), not a raw VDU passthrough.

## Web mode: VDU vs UDPipe

Both external services are called **server-side from the Worker**
([vdu.ts](../src/worker/vdu.ts), [udpipe.ts](../src/worker/udpipe.ts)) — so from
Cloudflare's egress IP, **shared across all users**. There is no per-user API key or
quota anywhere; the only exposure is the Worker's shared origin being rate-limited if
aggregate traffic gets high. The two services differ sharply:

| | VDU kirčiuoklė (`kalbu.vdu.lt`) | UDPipe (`lindat.mff.cuni.cz`) |
|---|---|---|
| Purpose | word accentuation | context POS tagging (homograph disambiguation) |
| Cached? | **Yes** — per word in D1 (`kirciuokle-words`), self-warming | **No** — tagging is sentence-contextual |
| Hit frequency | only novel words (≤10 misses/request, `MISS_BUDGET`); → ~0 as D1 warms | ~once per Web-mode request |
| On failure | upstream error surfaced (502) | graceful: `tagger:"unavailable"`, default reading |

So **UDPipe is the shared hot path**; VDU is well-insulated by the cache.

**CORS finding (measured):** UDPipe returns `Access-Control-Allow-Origin: *` and the
call is a CORS *simple request* (form-encoded, safelisted headers) → it can be called
**directly from the browser**. VDU sends no CORS headers and its nonce is
session-bound → it stays server-only. **Planned (flow A, not yet shipped):** move the
UDPipe call to the browser so tagging egresses from each user's own IP; the Worker
keeps orchestrating VDU-cache + disambiguation and falls back to a server-side UDPipe
call when the client doesn't supply tags.

## Spellcheck (fully client-side)

Underlines likely misspellings in the result pane and offers one-click fixes. **No
model, no server round-trip** — it runs in a browser **Web Worker** (a standard
on-device worker, *not* a Cloudflare Worker). The only server involvement is serving
two static files. Full design + rationale: [SPEC56.md](SPEC56.md).

- **Data are generated build artifacts, not source.** `public/spellcheck-lt.txt`
  and `public/spellcheck-bigrams.txt` are gitignored (like the model) and produced
  by `uv run scripts/regenerate_spellcheck_dicts.py` from the local
  `lexicon.sqlite` + `generated.sqlite` + the hermitdave frequency list + the local
  corpora. They must exist in `public/` before a build — `npm run build` fails fast
  (via `scripts/check_spellcheck_assets.mjs`) if they're missing.
- **Two-tier dictionary** — `spellcheck-lt.txt` (`form\tfreq`, ~580k forms = the
  union of `lexicon.sqlite` ∪ `generated.sqlite`):
  - *Accept* tier = **all** ~580k forms → "is this a real word?" (this is what stops
    rare-but-valid inflected forms from being false-flagged).
  - *Correct* tier = the **`freq>0` subset** (~77k) → the fold/delete indexes that
    generate suggestions (keeps browser memory bounded).
- **Ranking**: edit-distance band → context bigram (`spellcheck-bigrams.txt`) →
  frequency → deterministic tie-breaks. Statuses: `restore` (ASCII→diacritics,
  `as`→`aš`), `typo` (edit distance ≤2, Damerau, transposition = 1 edit), `ok`,
  `unknown`.
- **Web Worker + Cache API** ([spellcheck.worker.ts](../src/client/spellcheck.worker.ts),
  [spellcheckClient.ts](../src/client/spellcheckClient.ts)): builds the engine off the
  main thread (verified 0 main-thread long-tasks during the first ~5 s build), batches
  a text's words into one message, and falls back to in-thread if a worker can't
  spawn. The two asset files are stored via the Cache API (`spellcheck-assets-v*`) —
  downloaded once, reused every session, offline-capable (same idea as the model
  cache; bump the cache-name suffix when the assets change).
- **Live check** fires on paste + typing-pause (preview underlines *before*
  accentuation). **Fixes re-accentuate only the edited sentence** — an offset re-tile
  + reconstruction check guarantees left/right character alignment, with a full-text
  re-accentuation fallback if it can't.
