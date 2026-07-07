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

**VDU** is called server-side from the Worker ([vdu.ts](../src/worker/vdu.ts)) — from
Cloudflare's shared egress IP. **UDPipe** is now called from the browser (from each
user's own IP), with a server-side fallback ([udpipe.ts](../src/worker/udpipe.ts)).
There is no per-user API key or quota anywhere. The two services differ sharply:

| | VDU kirčiuoklė (`kalbu.vdu.lt`) | UDPipe (`lindat.mff.cuni.cz`) |
|---|---|---|
| Purpose | word accentuation | context POS tagging (homograph disambiguation) |
| Called from | Worker (shared IP) | **browser (user's IP)**, server fallback |
| Cached? | **Yes** — per word in D1 (`kirciuokle-words`), self-warming | **No** — tagging is sentence-contextual |
| Hit frequency | only novel words (≤10 misses/request, `MISS_BUDGET`); → ~0 as D1 warms | ~once per Web-mode request |
| On failure | upstream error surfaced (502) | graceful: `tagger:"unavailable"`, default reading |

VDU is well-insulated by the D1 cache; UDPipe used to be the shared hot path but now
egresses from the user's own IP (see below), leaving no per-request shared-IP load.

**UDPipe runs from the browser (shipped).** UDPipe returns
`Access-Control-Allow-Origin: *` and the call is a CORS *simple request*
(form-encoded, safelisted headers), so the client calls it **directly from the user's
own IP** ([`fetchUdpipeTags`](../src/client/main.ts)) and posts the raw CoNLL-U to
`/api/accent` as `tags`. The Worker uses those tags for alignment/disambiguation
([`getTaggerResult`](../src/worker/vdu.ts) → `parseConllu`) and **falls back to a
server-side UDPipe call** when `tags` is absent or unparseable (old clients, blocked
networks, a failed browser call). So UDPipe no longer routes through the Worker's
shared IP in the common case. VDU stays server-only — it sends no CORS headers and its
nonce is session-bound.

## Spellcheck (fully client-side)

Underlines likely misspellings in the result pane and offers one-click fixes. **No
model, no server round-trip** — it runs in a browser **Web Worker** (a standard
on-device worker, *not* a Cloudflare Worker). The only server involvement is serving
two static files. Full design + rationale: [SPEC56.md](SPEC56.md).

- **Data are generated build artifacts, not source.** `public/spellcheck-lt.txt`,
  `public/spellcheck-bigrams.txt`, `public/lt.dic` and `public/lt.aff` are gitignored
  (like the model) and produced by `uv run scripts/regenerate_spellcheck_dicts.py`
  (the wordlist/bigrams from the local `lexicon.sqlite` + `generated.sqlite` + the
  hermitdave frequency list + the local corpora; the `.dic`/`.aff` fetched from the
  BSD-3 Lithuanian hunspell). All four must exist in `public/` before a build —
  `npm run build` fails fast (via `scripts/check_spellcheck_assets.mjs`) if any is
  missing.
- **Accept = real Hunspell, not a wordlist.** "Is this a valid Lithuanian word?" is
  answered by the actual Hunspell engine (compiled to WebAssembly via `hunspell-asm`)
  running the BSD-3 ispell-lt `.dic`/`.aff` in the worker. Hunspell applies the full
  affix morphology, so *every* valid inflected form is recognised — this is what
  stopped the rare-inflection false positives (`sąjungininkių`, `priteisė`,
  `įslaptintame`, …) a finite corpus wordlist produced. If the dictionary fails to
  load, the engine falls back to the wordlist's own `valid` set.
- **Correction wordlist** — `spellcheck-lt.txt` (`form\tfreq`, ~162k freq-bearing
  forms) drives *suggestion generation and ranking* only: the fold/delete indexes for
  candidates, plus the frequencies that rank them. It does **not** decide acceptance
  (Hunspell does), so it can stay small and browser-memory-bounded.
- **Suggestions come from wordlist + Hunspell, and are validity-filtered.** Typo
  candidates are the wordlist's fold/delete matches **plus `hunspell.suggest()`** —
  the latter covers valid forms too rare to be in the slimmed wordlist (the correct
  `kalbeti`→`kalbėti` even when `kalbėti` isn't indexed). Then every candidate is run
  back through Hunspell and **dropped if it isn't itself a valid word**: the corpus
  wordlist carries diacritic-less noise like `pakalbeti`, and we must never offer a
  "fix" that is itself misspelled.
- **Fold-neighbour probe for double errors.** A real letter typo *plus* dropped
  diacritics — `pokalbeti` → `pakalbėti` (`o→a` **and** the ė) — is invisible to a
  plain fold lookup (`fold(pokalbeti)` ≠ `fold(pakalbėti)`), and Hunspell's own
  suggester won't cross two edits. So for rejected words we also look the fold index
  up over the **edit-1 neighbours of the query's fold** (`pokalbeti` → `pakalbeti` =
  `fold(pakalbėti)` → the real word). Pure Map lookups, still validity-filtered. The
  only correct form now out of reach is one that isn't in the ~162k wordlist at all.
- **Ranking**: edit-distance band → context bigram (`spellcheck-bigrams.txt`) →
  frequency → deterministic tie-breaks. Statuses: `restore` (ASCII→diacritics,
  `as`→`aš`, fired even for Hunspell-accepted ASCII words when the diacritic form
  dominates by frequency), `typo` (edit distance ≤2, Damerau, transposition = 1 edit),
  `ok`, `unknown`. Plain-`e` words that are genuinely valid (`gera`, `tema`, `erdvė`'s
  base) are accepted by Hunspell and never flagged, so `e`→`ę`/`ė` false positives
  don't happen at the accept layer; the only `e`→diacritic *restore* on an accepted
  word requires the diacritic sibling to dominate by frequency (≥8×, ≥100).
- **Web Worker + Cache API** ([spellcheck.worker.ts](../src/client/spellcheck.worker.ts),
  [spellcheckClient.ts](../src/client/spellcheckClient.ts)): builds the engine + the
  Hunspell instance off the main thread (Hunspell builds the ~85k-lemma dict in tens of
  ms; 0 main-thread long-tasks), batches a text's words into one message, and falls
  back to in-thread if a worker can't spawn. All four asset files are stored via the
  Cache API (`spellcheck-assets-v*`) — downloaded once, reused every session,
  offline-capable (same idea as the model cache; bump the cache-name suffix when the
  assets change).
- **Bundler note.** `hunspell-asm`'s published ESM build is broken under bundlers (it
  calls namespace imports as functions). `vite.config.ts` forces its CJS build (and
  `emscripten-wasm-loader`'s) for both dev and the worker sub-bundle — do not "simplify"
  back to the bare `import ... from "hunspell-asm"` or the prod worker silently ships
  the broken ESM path and every rare inflection gets false-flagged again.
- **Live check** fires on paste + typing-pause (preview underlines *before*
  accentuation). **Fixes re-accentuate only the edited sentence** — an offset re-tile
  + reconstruction check guarantees left/right character alignment, with a full-text
  re-accentuation fallback if it can't.
- **Fixes are undoable.** A single fix and **fix-all** are each applied through
  `document.execCommand("insertText")` over the minimal changed span, so each is one
  native **Ctrl+Z** step (and redo re-applies). Assigning `textarea.value` directly —
  the obvious approach — would wipe the undo stack, so don't
  ([`applyUndoableTextareaEdit`](../src/client/main.ts)).

## Installable app (PWA)

The site is an installable Progressive Web App — "Add to Home Screen" / the desktop
install icon. Beyond the app window, **installing is what makes the local model
durable**: Chrome only grants `navigator.storage.persist()` to engaged/installed
origins, and until it does, the Cache-API-stored model (~130–450 MB) sits in the
best-effort eviction pool and can vanish under disk pressure. We request persistence
whenever the model cache opens ([`ensurePersistentStorage`](../src/client/local/assets.ts));
installed users get it granted and stop re-downloading the model.

- **Manifest + icons** — `public/manifest.webmanifest` (standalone, teal `#09706c`
  theme, `any` + `maskable` icons) linked from `index.html` with the apple-touch and
  favicon tags. Icons are **committed** (small, stable) and regenerated from a single
  master by `uv run scripts/generate_pwa_icons.py`
  ([`design/icon-master.png`](../design/icon-master.png) → `public/icon-*.png`,
  `apple-touch-icon.png`, `favicon-*`). The `*.png` gitignore rule has explicit
  exceptions for these. Re-run the script if the master changes.
- **Service worker** ([public/sw.js](../public/sw.js)) — a *browser* service worker
  (not a Cloudflare Worker) that caches the app shell for offline use.
  **Registered in production only** ([main.ts](../src/client/main.ts)); dev unregisters
  any stale one so it can't fight Vite HMR.
  - It **returns responses verbatim** — never rebuilds them — so the COOP/COEP headers
    that give the page cross-origin isolation (required for the threaded ONNX runtime /
    `SharedArrayBuffer`) survive being served from the SW. This is the one invariant to
    preserve: a SW that constructs its own `Response` for a navigation would silently
    break Local mode. Verified on deploy that `crossOriginIsolated` stays `true` while
    the SW controls the page.
  - It **bypasses the large self-managed assets** (`/local-model/`, `.onnx`, `.wasm`,
    the spellcheck dictionaries, `/api/`) — those have their own Cache API storage;
    double-caching them in the shell cache would waste quota and cause eviction.
  - Navigations are **network-first** (updates land on the next load), assets are
    cache-first. To force every returning visitor onto a fresh shell, bump the
    `app-shell-v*` cache name in `sw.js`.
