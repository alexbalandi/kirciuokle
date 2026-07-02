# Phase 5 — D1 dictionary and local-first accentuation

Goal: stop depending on VDU's `text_accents` for every request. Store the
word dictionary in D1 (batch-queryable), accent text locally from it, fall
back to the legacy VDU whole-text path when local coverage is insufficient,
and let the dictionary warm itself with use. Quality rule: the local path
must reproduce the VDU path's output exactly — it is a reconstruction from
VDU's own per-word data, never a new engine. It ships OFF by default behind
a flag until a differential eval proves parity.

## D1 database (already created)

Add to `wrangler.jsonc`:
```jsonc
"d1_databases": [
  {
    "binding": "DICT",
    "database_name": "kirciuokle-words",
    "database_id": "09f3ad62-f4b7-4869-bf69-b941f4316bd1"
  }
]
```

Schema — put it in `migrations/0001_words.sql` (wrangler d1 migrations
layout) and note the apply commands in the README:
```sql
CREATE TABLE IF NOT EXISTS words (
  word TEXT PRIMARY KEY,          -- NFC, lowercased
  variants TEXT NOT NULL,         -- JSON: [{form,info,mi}], [] for negatives
  fetched_at TEXT NOT NULL,       -- ISO timestamp
  negative_until TEXT             -- ISO timestamp; non-NULL marks a negative
                                  -- entry valid until then (30 days)
);
```
The orchestrator (not codex) will apply the migration; codex should still
make `npm run check` pass without a live DB (tests mock the binding).

## Storage layer (replaces the KV layer from phase 4)

`src/worker/dictionary.ts`:
- `getWords(env, words: string[]): Promise<Map<string, AccentVariant[] | null>>`
  — batched `SELECT ... WHERE word IN (...)`, chunked at 90 bound
  parameters per query. Expired negatives (negative_until < now) are
  treated as absent. Returns `null` for absent words, `[]` for valid
  negatives.
- `putWords(env, entries: {word, variants}[])` — batched
  `INSERT OR REPLACE` (chunk likewise); entries with empty variants get
  `negative_until = now + 30 days`.
- `lookupWordVariantsD1(word, env, ctx)` — single-word read-through used by
  `/api/word` and by the legacy path's ambiguous-variant fetching: D1 hit →
  return; miss → VDU `word_accent` → `ctx.waitUntil(putWords(...))`.
- Delete the KV code path and the `WORDS` binding from wrangler.jsonc
  (namespace had only test data). Update tests from KV mocks to a D1 mock
  (in-memory Map behind the same `getWords`/`putWords` interface — keep SQL
  behind a thin seam so the mock is trivial).

## Local-first accent path

`src/worker/localAccent.ts`:

1. **Tokenize** exactly like VDU does: words are maximal runs of Unicode
   letters (`/\p{L}+/u`); everything between is separator parts.
   A word containing letters outside the Lithuanian alphabet
   (`A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž`) is `NON_LT` → `{type:"word", unknown:true}`
   without any dictionary lookup.
2. **Batch-read** all distinct remaining words (lowercased) via `getWords`.
3. For dictionary **hits**: default accented form = first variant's `form`,
   case-restored with the existing `matchCase`; `ambiguous` = more than one
   distinct variant `form`; valid negatives (`[]`) → `unknown:true`.
4. **Misses** (absent from D1):
   - if total misses ≤ `MISS_BUDGET` (25): fetch each via VDU `word_accent`
     (existing concurrency-6 pool), use the result, and `waitUntil` a
     `putWords` batch — the dictionary warms itself;
   - if misses > budget: **fall back to the legacy VDU `text_accents` path
     for this request** (correctness first on cold texts), and still
     `waitUntil`-seed up to `MISS_BUDGET` of the missed words via
     `word_accent` so repeat requests get warmer.
5. Disambiguation (UDPipe + scoring + lemma exceptions) runs identically in
   both paths — reuse the existing code; variants for ambiguous words are
   already in hand from the dictionary rows (no extra fetches needed in the
   local path).

## Wiring & observability

- `Env` gains `DICT: D1Database` and optional `ACCENT_SOURCE` var
  (`"vdu"` default | `"local"`).
- `/api/accent` picks the path from `ACCENT_SOURCE`, overridable per-request
  with `?source=local` / `?source=vdu` (for A/B verification).
- Response envelope gains `"source": "local" | "vdu"` (vdu = legacy path,
  including budget fallbacks).
- `/api/accent` response shape is otherwise byte-identical between paths.

## Quality bar

- Unit tests (mocked D1 seam + mocked fetch): batch chunking >90 words;
  negative expiry; local path on a fully-warm dictionary (parts identical to
  the legacy path's normalization for the same fixtures, including
  case restoration, NON_LT, ambiguous flags, chosen variants); miss-budget
  fallback triggers legacy path and still seeds; `?source=` override; NFC.
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/` or `docs/`. Do not create the D1 database or run
  migrations against remote (the orchestrator does that).
