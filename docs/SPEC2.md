# Phase 2 — contextual disambiguation of ambiguous words

Extends the app from SPEC.md. Reference implementation (validated end-to-end):
`scripts/accent_text.py` — port its disambiguation logic to TypeScript
faithfully (mapping tables, scoring weights, alignment, lemma exceptions).

## Problem

VDU marks homographs `MULTIPLE_MEANING` and returns an arbitrary default
(e.g. `yra` → `ỹra` from *irti*, while the copula *būti* → `yrà` is almost
always intended). We disambiguate by tagging the text in context and scoring
each variant's morphology against the contextual tag.

## Tagger: UDPipe 2 REST (LINDAT)

```
POST https://lindat.mff.cuni.cz/services/udpipe/api/process
Content-Type: application/x-www-form-urlencoded
tokenizer=&tagger=&model=lithuanian-alksnis&data=<text>
```
Response `{"result": "<CoNLL-U string>"}`. Parse token lines (skip `#`
comments and range ids like `1-2`): columns `FORM, LEMMA, UPOS, XPOS, FEATS`
(FEATS = `Case=Nom|Gender=Masc|...` or `_`).

- Timeout 10 s; one call per `/api/accent` request (whole text).
- On any failure: degrade gracefully — return VDU defaults, everything else
  still works (this path must be tested).
- Service is free/fair-use (CC BY-NC-SA model) — mention in README.

## Algorithm (port from `scripts/accent_text.py`)

1. In `/api/accent`, run VDU `text_accents` and UDPipe tagging concurrently.
2. **Align**: walk VDU WORD/NON_LT parts in order against UDPipe tokens in
   order; for each part scan forward up to 8 tokens for a case-insensitive
   `form` match; unmatched parts get no token (no disambiguation for them).
3. For each **distinct** ambiguous word, fetch VDU `word_accent` variants
   (server-side, concurrency ~6, `caches.default` 7-day cache per word — this
   replaces/absorbs the phase-1 `/api/word` caching).
4. **Score each variant per occurrence** (occurrences of the same word can
   resolve differently):
   - Parse each variant `mi` label into normalized slots via the `MI_TAGS`
     table in the Python script (copy it verbatim, longest-abbreviation-first
     matching): pos, gender, number, case, tense, person, voice, degree.
   - Build the same slots from the UDPipe token (UPOS + FEATS; participles:
     UPOS VERB/AUX with `VerbForm=Part` → pos `PART_VERB`; NOUN/PROPN merge;
     CCONJ/SCONJ merge; `Degree=Pos` is dropped).
   - Score: pos match +4 / mismatch −3; each other slot present on both
     sides: match +2 / mismatch −2. Variant score = max over its mi labels.
   - Winner needs a strictly higher score than the runner-up; ties → keep
     VDU default, mark unresolved.
5. **Lemma exceptions** (checked before scoring, same table as Python):
   `(word_lower, lemma)` → accented form: `("yra","būti") → "yrà"`,
   `("yra","irti") → "ỹra"`. Keep the table in one obvious place for growth.
6. NFC-normalize all accented forms.

## API changes

`POST /api/accent` response parts gain, for ambiguous words:
```ts
{
  text: "yra", type: "word", accented: "yrà",   // accented = chosen form now
  ambiguous: true,
  resolvedBy?: "lemma" | "context",              // absent = unresolved, VDU default
  variants: [{form: "ỹra", info: "vksm., es. l., 3 asm."},
             {form: "yrà", info: "vksm., es. l., 3 asm."}],
  chosen: 1                                      // index into variants
}
```
Since variants now ship inline, the client no longer needs `/api/word` for
the popover (keep the endpoint — it is still useful and already built).

## UI changes

- Auto-resolved ambiguous words: **green subtle underline** (legend: „parinkta
  pagal kontekstą“); unresolved: amber as before.
- Popover (both kinds): list variants with morphology; mark the currently
  chosen one; user click still overrides per-occurrence.
- If the tagger was down, show a small dismissible notice above the result:
  „Kontekstinė analizė nepasiekiama — dviprasmiškiems žodžiams parinktos
  numatytosios formos.“ (worker adds `"tagger": "ok" | "unavailable"` to the
  response envelope).

## Quality bar

- Unit tests (mocked fetch, no network): CoNLL-U parsing, alignment (incl.
  tokenization mismatch skip-ahead), mi parsing, scoring (masc-nom vs fem-acc
  adjective case), lemma exception, tagger-down degradation.
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` all pass.
- Do not modify `scripts/` or `docs/`.
