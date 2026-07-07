# SPEC56 — Spellcheck upgrades: frequency + context ranking, live preview, sentence-scoped fixes

Status: ready for implementation
Owner: codex (engine + UI), Claude (data assets — DONE, review + verify)

## Goal

Make the existing lightweight client-side spellcheck (SPEC55) noticeably smarter
and more responsive, without adding a model or any network round-trip at check
time. Four changes:

1. **Frequency-ranked candidates** — the most common real word wins (`as` → `aš`,
   not some rare homograph).
2. **Edit-distance-2 typos** — catch two-error typos, not just one.
3. **Context (bigram) tie-break** — when a flagged word has several candidates,
   prefer the one that forms a known collocation with its neighbour.
4. **Live check on paste + typing-pause** — flag misspellings in a lightweight
   preview *before* the (expensive) accentuation runs, and when the user fixes a
   word, re-accentuate **only the affected sentence**, not the whole text.

All four are additive: the current `suggest()` behaviour and the underline /
click-to-fix / fix-all UX stay exactly as they are; we improve ranking, coverage,
and *when* the check fires.

## Data assets (already built — do not regenerate)

Both are produced by scripts under `scripts/` and committed under `public/`.
The worker already serves `public/*` as static assets.

### `public/spellcheck-lt.txt` — frequency-augmented wordlist

- **New format**: one line per form, `"<form>\t<freq>"`.
  - `form` = un-stressed surface form, LT diacritics kept (unchanged from before).
  - `freq` = integer corpus frequency (hermitdave 2018 `lt_full`), `0` when the
    form is not in the frequency list.
- 130,215 forms; 42,186 have a non-zero frequency. ~1.79 MB raw / ~399 KB gzipped.
- Built (with the bigrams) by `scripts/regenerate_spellcheck_dicts.py`. Gitignored
  build artifact, not tracked.
- Example lines: `aš\t116732`, `ačiū\t14413`, `abadai\t0`.

### `public/spellcheck-bigrams.txt` — context table

- One line per adjacent word pair, `"<w1>\t<w2>\t<count>"`, both words lowercased,
  both are real surface forms (in the wordlist), `count >= 2`.
- 17,119 pairs. ~236 KB raw / ~83 KB gzipped.
- Built (with the wordlist) by `scripts/regenerate_spellcheck_dicts.py` from the
  local corpora. Gitignored build artifact, not tracked.
- Example lines: `taip\tpat\t205`, `ir\taš\t197`, `dėl\tto\t146`.
- **Coverage is deliberately thin** (small corpus). Use bigrams ONLY to re-rank the
  candidates of an *already-flagged* word. **Never** use them to flag a valid word
  as a real-word error — 17k pairs is far too sparse and would produce false
  positives. This is a ranking signal, not a detector.

## Part 1 — `src/client/spellcheck.ts` engine changes

### 1a. Parse the new wordlist format + store frequency

**Keep `createSpellcheckEngine(forms: Iterable<string>)` accepting plain strings**,
but have the engine parse each string on `\t`: field 0 is the form, field 1
(optional) is the integer frequency (a line with no `\t` → freq 0). This is
backward-compatible: the existing test's `createSpellcheckEngine(["ačiū", ...])`
still works (all freq 0), and the real loader just passes the raw file lines
(`"aš\t116732"`) straight through — `loadSpellcheckEngine()` splits on `\n`, trims,
filters empty, and hands the lines to the constructor unchanged.

In `SpellcheckEngine`, add `readonly freq = new Map<string, number>()` keyed by the
normalized form. Populate it in the constructor alongside `valid`.

### 1a-bis. Two-tier vocabulary (accept vs correct) — REQUIRED for memory

The shipped wordlist is now ~580k forms (the full own-pipeline union). Building the
fold/delete indexes over all 580k would create millions of entries and hang/OOM the
browser. Split responsibilities by the frequency field:

- **Accept tier = ALL forms** (~580k). `valid` (the membership Set) gets every form,
  regardless of frequency. This is what decides `ok` vs flagged, so it must be
  complete — that's the whole point of the expansion (a valid inflected word must
  never be flagged as a mistake).
- **Correction tier = forms with `freq > 0`** (~77k). Populate `foldIndex` (restore)
  and `deleteIndex` (typo) **only** for forms whose parsed frequency is `> 0`. So in
  the constructor loop: always add to `valid` and `freq`; guard the `foldIndex` and
  `deleteIndex` population with `if (freq > 0)`.

Rationale: suggestions (restore/typo candidates) only matter for words common enough
that someone would type and want them fixed; rare inflected forms just need to be
*accepted*, not *suggested*. This keeps both heavy indexes at ~77k entries — snappy
construction, bounded memory — while the accept set stays complete. Do not build a
delete-2 dictionary index (query-side deletes only, per 1c).

### 1b. Lazy-load the bigram table

Add a separate lazy fetch of `/spellcheck-bigrams.txt`, **only triggered the first
time context ranking is actually requested** (i.e. when `suggest()` is called with
a non-empty context). Do NOT fetch it eagerly on engine load — many sessions never
need it, and it should not delay the first check.

Store bigram counts on the engine as `bigrams: Map<string, number>` keyed by
`` `${w1}\t${w2}` ``. The engine method `suggest` STAYS SYNCHRONOUS and simply uses
whatever bigrams are currently loaded (empty map → context factor is a no-op). The
async fetch lives at the **module level**, not inside the sync method:

- `createSpellcheckEngine(forms, bigrams?)` gains an optional second param —
  `Iterable<string>` of `"w1\tw2\tcount"` lines, or a `Map<string, number>` — so
  tests can inject a bigram table directly. Also add
  `engine.setBigrams(lines: Iterable<string>): void` for the loader to populate it
  after a lazy fetch.
- Module-level `ensureBigrams(): Promise<void>` fetches `/spellcheck-bigrams.txt`
  once (memoized promise, same pattern as `sharedEnginePromise`), parses it, and
  calls `engine.setBigrams(...)`. A fetch failure must be swallowed — context
  ranking degrades to a no-op; never throw out of `suggest`.
- Trigger `ensureBigrams()` **only** the first time the async module-level
  `suggest(word, context)` wrapper is called with a non-empty context — never
  eagerly on engine load.

### 1c. Edit-distance-2 typo candidates (browser-safe, bounded memory)

Keep the current delete-1 dictionary index (`deleteIndex`). Do **not** build a
delete-2 dictionary index (memory blow-up in the browser). Instead reach edit
distance 2 from the **query** side:

- Generate `deletes1(query)` and `deletes2(query)` (deletes-2 = deletes-1 applied
  to each deletes-1 string; dedupe). For a single word this is cheap (O(len²)).
- Candidate pool = union of:
  - `deleteIndex.get(query)`,
  - `deleteIndex.get(d)` for every `d` in `deletes1(query)` and `deletes2(query)`,
  - any `d` in `deletes1(query) ∪ deletes2(query)` that is itself a valid form.
- **Verify** every pooled candidate with a real bounded edit distance and keep only
  those with `distance <= 2`. Over-generation from the delete lookups is fine
  because this verify step is the gate.

Replace the current ad-hoc `editDistance` (which conflates everything ≥2) with a
proper bounded Damerau–Levenshtein capped at 2 — it must return exact 0/1/2 and
`>2` as 3, and count a single adjacent transposition as distance 1 (catches the
common `ie`↔`ei` / swapped-letter typos). Keep the early-exit on length diff > 2.

`typo` candidates should still be surfaced only where the current logic surfaces
them (post-accentuation: word was `unknown`; preview: always — see Part 2), so
edit-distance-2 widening does not create new false positives on valid text.

### 1d. Ranking with frequency + context

Change the internal ranking so both `restore` and `typo` candidate lists are
ordered by, in priority order:

1. **Edit-distance band** — strictly lower true edit distance first. An ed-1 typo
   candidate always ranks above an ed-2 one; never let frequency or context promote
   an ed-2 candidate above an ed-1 candidate. (For `restore`, all candidates share
   the same folded form; rank the band by number of diacritic substitutions.)
2. **Context score** — if context neighbours are available, higher combined bigram
   count `bigram(prev, candidate) + bigram(candidate, next)` first. Skip this factor
   entirely when no context was supplied or no bigram matched (score 0 for all).
3. **Frequency** — higher `freq.get(candidate)` first.
4. **Existing deterministic tie-breaks** — diacritic count, length, `localeCompare`.

Then `slice(0, MAX_CANDIDATES)`.

### 1e. Public API

```ts
export type SpellcheckContext = { prev?: string; next?: string };

// SYNCHRONOUS engine method — uses whatever bigrams are already loaded:
class SpellcheckEngine {
  suggest(word: string, context?: SpellcheckContext): SpellcheckSuggestion
  setBigrams(lines: Iterable<string>): void
}

// ASYNC module-level wrapper — loads engine, lazily loads bigrams when context is
// supplied, then delegates to the sync engine method:
export async function suggest(
  word: string,
  context?: SpellcheckContext,
): Promise<SpellcheckSuggestion>
```

- `SpellcheckEngine.suggest` stays sync so the existing sync tests keep working. When
  `context` has a `prev`/`next` and bigrams are loaded, rank with the context factor;
  otherwise the factor is a no-op. `prev`/`next` are raw neighbour word strings;
  normalize them (NFC, lowercase — NO folding: bigrams store un-folded LT forms).
- The async module-level `suggest`: `await loadSpellcheckEngine()`; if `context` is
  non-empty, `await ensureBigrams()` before delegating. When `context` is
  omitted/empty, it resolves immediately with no bigram fetch — behaviour identical
  to today.
- `main.ts` already imports `suggest as suggestSpelling` (async) — signature grows
  an optional 2nd arg, callers updated per 2d.

Keep `SpellcheckStatus`, `SpellcheckSuggestion`, `createSpellcheckEngine`,
`foldAscii`, `resetSpellcheckForTests` exports intact (createSpellcheckEngine gains
the optional 2nd `bigrams` param).

## Part 2 — `src/client/main.ts` live preview + sentence-scoped fix

### 2a. Client-side tokenizer for preview

Add `tokenizeForPreview(text: string): RenderedPart[]` that partitions `text` into
`Part`-shaped tokens that exactly tile the string:

- A **word** token (`type: "word"`) is a maximal run of Lithuanian/Latin letters
  with internal hyphens/apostrophes allowed (mirror the wordlist's letter class:
  `a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ`, plus internal `-`). Everything else is a **sep**
  token (`type: "sep"`) — whitespace, punctuation, digits.
- Each token carries `sourceStart`, `sourceEnd`, and `current: text`. No `accented`,
  no `variants`, no `ambiguous`, no `unknown`.
- The concatenation of all `token.text` MUST equal the input exactly.
- Mark every token `preview: true` (new optional flag on `RenderedPart`).

### 2b. Preview render + correction gating

Add `preview?: true` to the `RenderedPart` type. Update `shouldShowCorrection`:

```ts
function shouldShowCorrection(part: RenderedPart): boolean {
  const spelling = part.spelling;
  if (!spelling || spelling.candidates.length === 0) return false;
  if (spelling.status === "restore") return true;
  if (spelling.status === "typo") return part.preview === true || Boolean(part.unknown);
  return false;
}
```

`renderResult()` needs no other change — preview word tokens with no accent/variants
fall through to the plain-text-node branch, and flagged ones render as the existing
`token-correctable` button. (A preview shows the raw text on the right with just the
misspelling underlines — no accent marks yet.)

### 2c. Live check trigger (paste + typing pause)

Add a debounced live-check that runs the preview spellcheck **without** accentuation:

- New `schedulePreviewSpellcheck()` — debounce ~600 ms.
- Fire it from the `textarea` `input` handler (covers typing-pause) **and** from a
  new `paste` handler (fire immediately on paste, i.e. after the pasted text lands —
  use a microtask/`requestAnimationFrame` so `textarea.value` is updated).
- `runPreviewSpellcheck()`:
  1. Bump `spellcheckRequestId` (shares the existing staleness guard so a later
     accentuation or a newer keystroke supersedes it).
  2. If `textarea.value.trim()` is empty → clear preview and return.
  3. **Guard against clobbering a real accentuation**: only render a preview when
     the current result is not already an up-to-date accentuation of this exact
     text. Concretely: skip if `canRewriteRenderedSource()` is true AND
     `renderedParts` are non-preview (i.e. the right pane already shows the accented
     result for the current text). Otherwise the user edited the text after
     accentuating (or never accentuated) → a preview is appropriate.
  4. `renderedParts = tokenizeForPreview(textarea.value)`; `renderedSourceText =
     textarea.value`.
  5. Run spellcheck over the word tokens with neighbour context (2e), respecting the
     `spellcheckRequestId` staleness guard, then `renderResult()`.
- When the user hits accentuate (`submitText`) the normal flow overwrites the
  preview with the real accented parts — no special-casing needed beyond the guard
  in step 3.
- `copyButton` should stay disabled while showing a preview (there's nothing
  accented to copy yet) — gate `copyButton.disabled` on "parts exist AND not
  preview".

### 2d. Neighbour context for `suggest`

`annotateUnknownWordsWithSpellcheck()` and the preview both currently call
`suggestSpelling(part.text)`. Change both to pass context: for the target word at
`renderedParts[index]`, `prev` = the nearest preceding `type: "word"` part's text,
`next` = the nearest following `type: "word"` part's text (skip `sep` parts; may be
undefined at text edges). This is the only wiring the bigram ranking needs.

### 2e. Sentence-scoped re-accentuation on fix

Today `applySpellingCorrection()` and `fixAllRestores()` end with `submitText()`,
which re-accentuates the **whole** text (≈26 s in local mode). Replace that final
step with a sentence-scoped re-accentuation that is provably alignment-safe.

Add `reaccentuateEdits(edits: Array<{ start: number; end: number; text: string }>)`
called AFTER `replaceTextareaRanges(edits)` (which has already mutated
`textarea.value`). `edits` are ranges in the **pre-edit** source (`renderedSourceText`
before the rewrite). Algorithm:

1. If `accentMode === "web"`, or `renderedParts` was a preview, or anything below
   fails a safety check → fall back to `submitText()` (correctness over speed).
2. **Affected spans (old text)**: for each edit, find the sentence containing it in
   the OLD text (`renderedSourceText`). Sentence boundary = run bounded by
   `.!?…\n` terminators (scan left to the previous terminator or text start; scan
   right through the next terminator inclusive, or text end). Snap each span outward
   to the nearest enclosing **part** boundary (start = `sourceStart` of the first
   part it intersects, end = `sourceEnd` of the last). Merge overlapping/adjacent
   spans. Result: a sorted, disjoint list of old-text `[start,end)` spans.
3. **New span text**: for each old span, compute its new-text range by shifting for
   the cumulative length delta of edits lying before the span (no edit straddles a
   snapped boundary because spans are part-aligned and edits replace whole words).
   `newSpanText = textarea.value.slice(newStart, newEnd)`.
4. **Accent each new span** with a side-effect-free helper `accentFragment(text):
   Promise<Part[]>` (local mode: call the engine's accent on just the fragment;
   reuse the same engine, ignore/merge stats). Show the existing local run status
   while it works.
5. **Rebuild `renderedParts`** as a concatenation, in source order:
   - kept OLD parts before the first span,
   - new parts for span 1 (from `accentFragment`),
   - kept OLD parts between spans,
   - … new parts for the last span,
   - kept OLD parts after the last span.
6. **Re-tile offsets**: walk the concatenated parts in order, assigning
   `sourceStart`/`sourceEnd` sequentially from 0 by each `part.text.length`, and set
   `current = accented ?? text` (as `applyAccentResponse` does). Set
   `renderedSourceText = textarea.value`.
7. **Safety check**: the joined `part.text` of the rebuilt list MUST equal
   `textarea.value` exactly. If not → discard and `submitText()` fallback.
8. `renderResult()`; then re-run `annotateUnknownWordsWithSpellcheck()` (with
   context) so the freshly accented sentence gets its spellcheck flags. Enable
   `copyButton`.

This keeps the left/right character alignment exact (offsets are re-tiled from the
real text and verified) while only paying accentuation cost for the edited
sentence(s). `fixAllRestores()` passes all its replacements through the same path;
edits in different sentences produce multiple spans, each accented independently.

## Part 3 — Tests

Extend `test/` (there is an existing spellcheck test; mirror its style, Vitest):

- Frequency ranking: given a folded query with multiple restorations, the
  higher-frequency form ranks first (`as` → `aš` before any rare homograph).
- Edit-distance-2: a two-error typo of a dictionary word yields that word as a
  `typo` candidate; a 3-error string yields `unknown`.
- Transposition counts as distance 1.
- Context: with `bigrams` seeded (use `createSpellcheckEngine` + a test hook to
  inject bigram counts, or expose a small setter), a candidate forming a known pair
  with `prev`/`next` outranks a higher-frequency candidate that doesn't.
- `tokenizeForPreview`: round-trips (joined tokens === input) on text with mixed
  punctuation, newlines, hyphenated words, and digits.
- Sentence-scoped rebuild: a unit test of the offset re-tiling + reconstruction
  check (joined parts === text) for a two-sentence input where the first sentence is
  edited — parts after the edit get correctly shifted offsets.

## Part 4 — Web Worker + Cache API (post-review addition)

Once the wordlist grew to ~580k forms, building the engine on the main thread froze
the UI for ~5 s on first check. Fixed by moving the engine off-thread. This is a
browser **Web Worker** (standard, on-device — NOT a Cloudflare Worker); nothing
about spellcheck depends on the server beyond serving the static `.txt` files.

- `src/client/spellcheck.worker.ts` — instantiates `SpellcheckEngine`, builds it on
  first message, lazily loads bigrams on the first context request, and answers a
  batched `{ id, words: [{word, prev?, next?}] }` request with `{ id, results }`.
  Fetches `/spellcheck-lt.txt` and `/spellcheck-bigrams.txt` through the **Cache
  API** (`spellcheck-assets-v<n>`), so they download once and are reused every
  session (offline-capable). Bump the cache-name suffix when the shipped assets
  change; stale suffixes are pruned on worker start. Typed against the DOM lib via a
  minimal `WorkerScope` cast (no WebWorker-lib/DOM-lib conflict).
- `src/client/spellcheckClient.ts` — main-thread wrapper. Lazily constructs the
  worker, batches a whole text's words into one message, and **falls back to the
  in-thread `suggest()`** if the worker can't be created (correctness never depends
  on the worker; it's a "don't block the UI" optimisation).
- `annotateUnknownWordsWithSpellcheck()` calls `suggestBatch(words)` once instead of
  per-word `suggest()`.
- Verified: `PerformanceObserver('longtask')` shows **0 main-thread tasks > 50 ms**
  during the first build; both asset files land in the Cache API; flags remain
  correct (restore/typo/accept). Production `vite build` emits the worker as its own
  same-origin chunk, which loads fine under the site's COOP/COEP isolation headers.

Two-tier vocabulary (§1a-bis) still applies inside the worker: `valid` = all ~580k
forms (accept), fold/delete indexes = the `freq > 0` subset (suggest).

## Constraints / notes

- No new dependencies. Pure TS, runs in the browser and in Vitest.
- Never throw out of `suggest()` — a missing/failed bigram fetch degrades to
  no-context ranking.
- Keep `MAX_CANDIDATES = 8`.
- Preserve the existing casing re-application (`reapplyCase`) on candidates.
- Do not touch the worker, wrangler config, or the local model pipeline.
- `npm run build` (tsc + vite) and the Vitest suite must pass.
```
