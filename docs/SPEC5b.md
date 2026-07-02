# Phase 5b — exact parity for the local accent path

The A/B parity test (local vs vdu on identical text) found the local
reconstruction diverges because VDU's `text_accents` metadata is NOT
derivable from `word_accent` variant lists:

- `text_accents` has accentType `MULTIPLE_VARIANT` (one meaning, several
  accepted pronunciations, e.g. *pasiekia* → pasíekia|pasiẽkia). The VDU
  path treats these as plain words (only `MULTIPLE_MEANING` becomes
  `ambiguous`); counting distinct variant forms wrongly flags them.
- `word_accent` sometimes returns readings `text_accents` suppresses (e.g.
  *kas* has an unaccented particle reading; `text_accents` still says `ONE`
  → kàs).
- Roman numerals (*II*) come back as type `WITH_NUMBER` from
  `text_accents` — never looked up, no flags. The local tokenizer treated
  them as Lithuanian words → negative lookup → wrongly `unknown`.

Fix: store the canonical default per word and drive the local path off it.

## Migration

`migrations/0002_default_forms.sql`:
```sql
ALTER TABLE words ADD COLUMN default_form TEXT;
ALTER TABLE words ADD COLUMN accent_type TEXT;
```
(Orchestrator applies it; tests keep mocking the seam.)

## Dictionary entry semantics

Entry = `{variants, defaultForm, accentType}`.
- A row is **complete** iff `accent_type IS NOT NULL` OR it is a valid
  negative (`variants = []`, unexpired `negative_until`). Incomplete rows
  (phase-5 leftovers with NULL accent_type) are treated as absent and
  refetched.
- On a runtime miss, fetch BOTH upstreams for the word: `word_accent`
  (variants) and a single-word `text_accents` (default form + accentType);
  add a `fetchWordEntry(word)` helper in `vdu.ts`. If the single-word
  `text_accents` yields no WORD part or no accented form, store a negative.
- Reduce `MISS_BUDGET` to 15 (each miss now costs two upstream calls).

## Local path rules (parity-exact with the VDU path)

1. Tokens matching `/^[IVXLCDM]+$/` (uppercase Roman numerals) are emitted
   as plain word parts — no lookup, no flags (mirrors `WITH_NUMBER`).
2. Dictionary hits:
   - `accentType === "MULTIPLE_MEANING"` → `ambiguous: true`, accented =
     case-restored `defaultForm`, disambiguation + variants exactly as now;
   - `accentType === "ONE"` or `"MULTIPLE_VARIANT"` → plain part, accented
     = case-restored `defaultForm` (no ambiguous flag — same as the VDU
     path);
   - `accentType === "NONE"`, or negative entry → `unknown: true`.
   - The ambiguous flag comes ONLY from `accentType`, never from counting
     distinct variant forms.
3. Everything else (NON_LT charset check, miss-budget fallback to the
   legacy path, seeding via `waitUntil`, UDPipe disambiguation) unchanged.

## Quality bar

- Update/add tests: MULTIPLE_VARIANT hit is plain and uses defaultForm;
  ONE hit with extra suppressed readings (kas-style) is plain → kàs; Roman
  numeral token skips lookup and carries no flags; incomplete row (NULL
  accent_type) is refetched and rewritten complete; miss path stores
  defaultForm + accentType.
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/` or `docs/`; no remote migrations; no deploys;
  no git.
