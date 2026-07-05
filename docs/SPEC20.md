# SPEC20 — Coverage closure I: attestation expansion + apocope

## Context

2,751 VDU-cache positives remain uncovered by `generated.sqlite`. Residue
analysis found two mechanical classes this spec closes:

- **231 words are prefix + covered-base verb combos** (apiplaukė,
  atpažinti) that `generate_prefixed_verbs` skipped only because combo
  attestation reads the lt_50k frequency list, and these words are not in
  it. The VDU cache keys (`local/data/words.sqlite`, table `words`, column
  `word`) are real words from real user text — using the KEYS as an
  attestation/candidate set is provenance-clean (we never read VDU's
  accents, only which words exist).
- **123 words are standard apocope shortenings** of covered infinitives
  and imperatives (ateiti→ateit, atidaryti→atidaryt, neški→nešk): final
  -i drops, accent stays put (the miegót precedent in parity_vetoes.json:
  standard shortening keeps the acute).

Modify ONLY `local/accentuator/generate_dictionary.py`.

## 1. Attestation/candidate expansion

Add a module-level helper next to `DEFAULT_WORDLIST_NAME`:

```python
def load_candidate_words(wordlist: Path) -> set[str]:
    """lt_50k frequency words plus VDU-cache word keys (keys only —
    provenance-clean: no accent data is read)."""
```

- lt_50k part: exactly the `line.split()[0]` logic used today.
- VDU part: `SELECT word FROM words` from `DEFAULT_VDU_SQLITE` (import it
  from `_common`; it may be missing on other machines — skip silently).
- Use the helper in BOTH `generate_prefixed_verbs` (replace its inline
  `words = {...}` wordlist read) and `generate_derived` (whose `words`
  list feeds `derive_lemmas`; note derive_lemmas lowercases/filters
  itself). Keep each function's `wordlist` parameter working.

## 2. New module `generate_apocope`

Runs AFTER all other modules, BEFORE `write_generated` (call it right
after `generate_derived` in `generate_dictionary`, summary key
`"apocope_forms"` placed after `"derived_lemmas"`).

For every word key in `grouped` ending in `ti` or `ki` (NOT `tis` —
reflexives do not apocopate this way), for each of its variants whose
`info` label marks an infinitive or a 2sg imperative:

- determine the EXACT label strings empirically before coding: query the
  existing artifact, e.g.
  `sqlite3 local/accentuator/data/generated.sqlite "SELECT variants FROM words WHERE word IN ('miegoti','neški','neškite')"`
  and use the label substrings you observe (expect something like
  `bendratis` for infinitive and an imperative label for -ki forms; match
  on those, do not guess).
- the apocope form = variant form minus its final base letter `i` (keep
  all combining marks that belong to earlier letters). Skip the variant if
  its stress mark sits on the final `i` itself or the resulting form would
  carry no stress mark. Skip -kite forms (only -ti and -ki shorten).
- emit via `add_variant` with the same pos/tags context you can carry
  (reuse the source variant's info as the label is NOT possible through
  add_variant — instead call `add_variant` with pos/tags that reproduce an
  equivalent label, or extend the emission to reuse the source label via
  `grouped[word][key]` insertion in the same shape add_variant produces;
  choose whichever is cleaner but keep the (form, label) dedup key
  behavior).
- provenance: `open-accentuator:apocope:{source_word}`.
- apocope must NOT overwrite an existing word key that another module
  already produced (check `grouped` membership first; skip if present).

## Pass criteria (repo root, run in order)

1. `uv run local/accentuator/generate_dictionary.py` completes; summary
   prints `apocope_forms` > 100 and `prefixed_verbs` strictly greater
   than the current 822.
2. `uv run local/accentuator/parity_report.py` — REQUIRED: `covered`
   strictly greater than the current 7,380. If `disjoint` is nonzero, DO
   NOT edit parity_vetoes.json and DO NOT try to fix rules — paste the
   full DISJOINT sample list from reports/parity-vdu.md into your final
   message for human adjudication. disjoint == 0 is the ideal outcome but
   surfacing disagreements honestly beats hiding them.
3. `uv run local/accentuator/selfcheck.py` passes.
4. Report the before/after numbers (covered, exact, overlap, norm_delta,
   disjoint, words) in the final message.

Do not commit. Do not touch parity_vetoes.json, fetch scripts, or other
modules beyond what section 1–2 requires.
