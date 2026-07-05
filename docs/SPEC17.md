# Phase 17 — open accentuator W2: lexicon extraction + paradigm engine core

Per docs/PLAN-open-accentuator.md (W3 serving integration remains
DEFERRED — nothing here touches the site or replica behavior). Everything
lives under `local/accentuator/`.

## 1. `extract_lexicon.py` — the open lexicon from kaikki

Parses `local/tagger-hf/data/raw/kaikki-lt.jsonl` into
`local/accentuator/data/lexicon.sqlite` (gitignored) with tables:

- `nominals(lemma, pos, accented_lemma, stress_class, plural_accented,
  gender, source)` — from entries with `stress pattern N` (nouns, names,
  adjectives, pronouns, numerals). Multi-class lemmas get one row per
  class.
- `verbs(lemma, accented_infinitive, present_3, past_3, source)` — from
  entries with accented principal parts in head templates
  (`darýti / dãro / dãrė` style).
- `forms(stripped, accented, lemma, pos, tags)` — every accented table
  form (the 192k), for cross-validation and direct lookup.
- `closed_draft(lemma, upos, accented_head, frequency, verified)` — the
  450 closed-class lemmas (MATAS UPOS majority in CCONJ/SCONJ/ADP/PART/
  PRON/DET/INTJ/NUM/ADV/AUX, freq ≥ 50), joined with kaikki accented
  headwords where available; `verified=0` everywhere — human review
  happens later against VLKK sources. Words with no kaikki accent get
  accented_head NULL (to be filled during review).
  Also emit `closed_draft.md` — a review-friendly markdown table sorted
  by frequency.

## 2. `paradigm_engine.py` — nominal + verb generation

Pure functions, no I/O. Implements published accentology:

- `accent_nominal(accented_lemma, stress_class, declension_info, cell)
  -> accented_form(s)`:
  - Declensions: (i)as/is/ys, a/ė/i, us/ius, uo/ė-cons; adjectives incl.
    pronominal (definite) forms.
  - The four stress classes: class 1 fixed barytone; class 2 with
    Saussure–Fortunatov retraction cells (acc sg, ins sg, loc sg per
    declension...); class 3/3a/3b mobile (ending-stressed in the defined
    cells, stem otherwise); class 4 mobile with Saussure. Ending accent
    inventory: which endings take grave/circumflex when stressed
    (namù, namè, namaĩ, namų̃...; -ą always unstressed for class 2 via
    Saussure? no — class 2 acc sg IS retracted stem stress; encode from
    the tables below).
  - AUTHORITATIVE SOURCE for the cell tables: derive them empirically
    from the kaikki `forms` table itself — for each (declension, class),
    collect covered lemmas' full paradigms and induce the
    stem-vs-ending-stress pattern + ending accent per cell; store the
    induced tables as data (`paradigm_tables.json`) with per-cell example
    counts. Hand-written expectations exist in accentology books, but the
    induction approach guarantees consistency with the 192k observed
    forms and surfaces any Wiktionary-module quirks as low-count cells.
    Report cells with conflicting inductions (count both, pick majority,
    log).
- `accent_verb(accented_inf, present_3, past_3, cell)`: present/past
  stems take their accent from the principal part (stress position and
  type propagate within the tense across persons, with the known
  1sg/2sg ending-stress cases for ending-stressed presents — induce from
  kaikki verb tables the same way); future/conditional/imperative from
  the infinitive stem per standard rules (also inducible: kaikki verb
  entries with tables provide the evidence).
- Every generated form returns (accented_form, cell_morphology) so
  downstream can build VDU-style variant labels.

## 3. `generate_dictionary.py`

Walks the lexicon, generates full paradigms for nominals (all cases ×
numbers (+ definiteness for adjectives)) and covered verbs (finite
indicative tenses + conditional + imperative + infinitive; participles
EXCLUDED from this phase), plus closed-class forms from `closed_draft`
where accented_head is present. Output:
`local/accentuator/data/generated.sqlite` in the words-table schema used
by the replica (word/variants/default_form/... + `provenance` column) —
STANDALONE ARTIFACT, consumed only by the parity tooling.

## 4. `parity_report.py` (W4 first cut)

For every positive entry in `local/data/words.sqlite` (the VDU cache):
if the generated dictionary covers the stripped form, compare variant
sets and default form. Buckets: EXACT / DEFAULT-MATCH (default agrees,
variant sets differ) / OVERLAP / DISJOINT / UNCOVERED. Writes
`local/accentuator/reports/parity-vdu.md` with counts, percentages, and
up to 40 DISJOINT samples annotated with lemma+class+cell provenance so
divergences can be adjudicated (ENGINE-BUG vs NORM-DELTA vs Wiktionary
data error).

## Quality bar

- `selfcheck` additions in a new `local/accentuator/selfcheck.py`:
  namas/class-4 full paradigm matches the known table (nãmas, nãmo,
  nãmui, nãmą, namù, namè, nãme?, namaĩ, namų̃, namáms?, namùs, namaĩs,
  namuosè — use the kaikki table as the assertion source at runtime);
  one class-1, one class-2 (Saussure cell), one class-3 noun; one
  adjective; one verb (darýti: dãro, dãrė, darýs, darýtų, darýk).
- py_compile + --help everywhere; scripts runnable via
  `uv run local/accentuator/<script>.py`.
- Do NOT modify anything outside `local/accentuator/` (+ .gitignore for
  its data/). Do not run the full generation (orchestrator does). No git.
