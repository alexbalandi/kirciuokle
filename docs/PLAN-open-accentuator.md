# Plan — open accentuation engine (published-grammar route)

Goal: an accentuation engine buildable from open sources, with VLKK's
codified norm as the source of truth, at measured parity with the
VDU-based solution. Replaces the last closed dependency (VDU's lexicon)
for regular vocabulary; VDU cache remains as fallback and as QA ground
truth.

## Architecture

```
wordform ──tagger──▶ FEATS (paradigm cell) + LEMMA (edit-script head)
                        │
   lemma lexicon ◀──────┘   lemma → accent class(es) + accented headword
        │
   paradigm engine          published accentology rules:
        │                   class I–IV stress placement per cell,
        ▼                   ending accent inventory, Saussure's law,
   accented form(s)         syllable weight → accent type
```

- The tagger (v2, with lemma head) supplies the analysis; the engine is
  deterministic after that.
- Output format matches the existing dictionary schema (variants with
  morphology labels, default form, ambiguity flag), so it plugs into the
  replica as a dictionary *layer*: VDU-cache → generated-open → unknown.
  Provenance column distinguishes layers.

## Data sources & their roles

| source | role | license/status |
|---|---|---|
| Wiktionary via wiktextract/kaikki | bulk seed of lemma→accent class + accented headword (facts) | CC BY-SA |
| VLKK (Kirčiavimo žinynas, consultation bank, nutarimai) | normative authority: adjudicates divergences, defines current codification incl. permitted variants | reference use, cite |
| BŽ/DLKŽ (ekalba) | spot verification of individual facts during adjudication; NOT bulk-scraped | reference use |
| published accentology (Stundžia; Mikulėnienė & Stundžia) | the rule engine's specification | textbook knowledge |
| our D1/SQLite VDU cache (11k words) | QA ground truth for parity measurement only — never a data source for the artifact | internal |

Provenance stance: facts (accent classes) from openly licensed Wiktionary;
generation from published grammar implemented by us; VDU used only to
*measure*. Divergence adjudication cites VLKK — if VDU and the engine
disagree, current VLKK codification decides who is right.

## Phases

**W1 — data probe (CPU, ~half a day).** Parse kaikki.org Lithuanian
extract: lemma, POS, accent class ("stress pattern N"), accented headword,
and (where present) the accented declension tables. Metrics: lemma count
with class info; coverage of (a) the 11k D1 cache lemmas, (b) MATAS lemma
frequency mass; headword-accent agreement vs D1 cache. Go/no-go: ≥80%
token-mass coverage expected.

**W2 — paradigm engine (the core build).** Order of attack:
1. Nominals (nouns, adjectives): 5 declensions × 4 accent classes;
   ending accent inventory; Saussure–Fortunatov law; pronominal (definite)
   adjective forms. Cross-verify generated tables against Wiktionary's own
   rendered tables (an independent implementation of the same rules) AND
   the D1 cache.
2. Verbs: present/past/past-iter/future/conditional/imperative stress
   patterns, prefixed verbs, reflexives, participles (declined like
   adjectives with their own class logic).
3. Closed classes by explicit table: pronouns, numerals, adverbs,
   prepositions (proclitic behavior), particles.
4. Variant handling: VLKK-permitted accent variants (some words have two
   codified options) come out as multiple variants, matching how the
   pipeline already treats MULTIPLE_VARIANT.
Every generated form carries its paradigm-cell morphology → variants get
mi-style labels natively.

**W3 — integration (DEFERRED — explicitly out of scope for now).** No
change to the running site or the replica's dictionary chain. The
generator's output stays a standalone artifact (SQLite in the existing
`words` schema, with provenance) used only by the W4 parity tooling.
Wiring it into any serving path is a separate, later decision.

**W4 — parity gate & divergence adjudication (the "make it tight" part).**
1. Form-level: generate for every lemma in the D1 cache; diff all variants
   against VDU's verbatim answers. Classify each divergence:
   - ENGINE-BUG: rule misimplementation (fix, add regression test);
   - COVERAGE: lemma/class missing or ambiguous in seed (log, fallback);
   - NORM-DELTA: engine follows current VLKK codification, VDU differs
     (or vice versa) — document with VLKK citation; these are the
     interesting ones and get a dedicated report section;
   - VARIANT-SET: same default, different variant inventory (usually
     VDU listing rarer readings) — tolerated, logged.
2. Text-level: full accent eval (sample + corpus) with the open layer
   forced, vs production. Gate: no regression beyond documented
   NORM-DELTA cases.
3. Human-review pack: the NORM-DELTA list rendered with VLKK references
   for manual sign-off before the layer ships as default-on.

## The lemma head's role (and its failure mode)

The engine's correctness chain starts at the predicted lemma. Wrong lemma
→ wrong lexicon row → wrong class → wrong accent. Mitigations:
- lemma-head accuracy is measured per-release (official Lemmas F1);
- lexicon lookup validates plausibility: predicted lemma must exist and
  its POS must match the predicted UPOS, else the token falls back to the
  VDU-cache layer / unknown;
- the D1-cache layer stays first in the chain, so warm words never depend
  on the lemma head at all.

## Explicit non-goals (first release)

Proper-noun accentology beyond what the seed covers; dialectal variants;
foreign-word adaptation rules; syllabification edge cases in rare
borrowings. All fall back to the existing layers.
