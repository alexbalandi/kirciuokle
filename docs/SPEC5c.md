# Phase 5c — case-sensitive canonical data and abbreviations

Corpus-scale A/B against the VDU path exposed two remaining local-path gaps:

1. **VDU's canonical data is case-sensitive.** Verified live:
   `alyta` → accentType NONE, but `Alyta` → `Alytà` MULTIPLE_MEANING;
   `vilnius` → MULTIPLE_VARIANT `vìlnius`, but `Vilnius` → MULTIPLE_MEANING
   `Vìlnius`. Lowercase-keyed rows mis-handle capitalized occurrences
   (proper nouns lose accents and their ambiguity flag).
2. **Abbreviations.** VDU leaves `a.`, `m.`, `rus.`, `lot.`, `liet.`, initials
   (`V.`) unaccented (token includes the dot, flagged like unknown). The
   local path accented them (`à.`, `rùs.` — wrong).

## Migration

`migrations/0003_title_case.sql`:
```sql
ALTER TABLE words ADD COLUMN default_form_title TEXT;
ALTER TABLE words ADD COLUMN accent_type_title TEXT;
```

## fetchWordEntry (now 3 upstream calls)

`fetchWordEntry(word)` (word is the lowercase key):
- `word_accent(word)` → variants (unchanged);
- `text_accents(word)` → lower side: `{defaultForm, accentType}`;
- `text_accents(TitleCase(word))` → title side:
  `{defaultFormTitle, accentTypeTitle}`.
- A side with no accented WORD part stores form `null` and type `"NONE"`
  (store the explicit string, never leave NULL — NULL now means
  "legacy row, incomplete").
- Negative (`negative_until`, variants forced `[]`) only when variants are
  empty AND both sides have no form.

## Dictionary

- Read/write the two new columns. **Completeness rule: a row is complete
  iff `accent_type_title IS NOT NULL`.** (Legacy 5b rows — including old
  negatives, which were probed lowercase-only — must be refetched; the
  `alyta` case shows a lowercase negative can hide a valid `Alytà`.)
- Entry shape: `{variants, lower: {form, type}, title: {form, type}}` (or
  equivalent flat fields).

## Local path

- **Abbreviation rule (before Roman-numeral and lookup):** if the token is
  immediately followed by `.` in the original text AND (its length is 1 OR
  its lowercase form is in the exported `ABBREVIATIONS` set), emit it as a
  WORD part with accentType `"NONE"` (→ `unknown: true`, no lookup) — this
  mirrors VDU. Curate the set with the standard Lithuanian abbreviations,
  at least: pvz, kt, kan, doc, prof, dr, habil, akad, gen, vyr, jaun, dir,
  pirm, pan, etc, proc, mln, mlrd, tūkst, egz, pav, sk, str, nr, tel, adr,
  apskr, aps, sav, mstl, gyv, val, min, sek, angl, vok, pranc, rus, lot,
  gr, liet, it, isp, lenk, latv, est, ukr, sen, šnek, tarm, psn, žr, plg,
  dab, pgl, plk, mjr, kpt, šv, pr (feel free to extend with other standard
  VLKK abbreviations).
  Note `XX a.`: `XX` is not followed by a dot → still Roman-numeral rule.
- **Case selection per occurrence:** first character uppercase → use the
  title side; if the title side has no form AND type `"NONE"`, treat as
  unknown (do NOT fall back to the lower side — VDU doesn't). Lowercase
  first char → lower side. `matchCase` still applies for ALL-CAPS.
- `ambiguous` flag = chosen side's type `MULTIPLE_MEANING`; unknown = chosen
  side type `"NONE"`/no form; `MULTIPLE_VARIANT`/`ONE` → plain with the
  side's form.
- `MISS_BUDGET` → 10 (a miss now costs three upstream calls).

## Quality bar

- Tests: alyta/Alyta fixture (lowercase unknown, capitalized ambiguous with
  Alytà); vilnius/Vilnius (variant vs meaning flags); abbreviation rule
  (`m.`, `rus.`, initial `V.`, but plain `kalba` unaffected, `XX` before
  ` a.` stays Roman); legacy 5b row (NULL title columns) counts as a miss
  and is refetched complete; negative requires both sides empty.
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/` or `docs/`; no remote migrations; no deploy; no
  dev server; no git.

## Known accepted deltas vs the VDU path (do not chase these)

- VDU path escapes apostrophes (`\'`) and can swallow a space after a
  sentence dot — the local path is deliberately faithful to the input.
- For MULTIPLE_MEANING words the single-word canonical default can
  occasionally differ from VDU's in-text default (e.g. *verčiama*); both
  paths flag the word ambiguous and disambiguation/popovers apply, so only
  unresolved ties are affected.
