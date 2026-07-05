# Open accentuator — pipeline architecture

`local/accentuator/` builds a Lithuanian accentuation dictionary from open
sources. This document explains exactly what runs, in what order, on what
data, and how quality is controlled. Plan and source-selection rationale:
[PLAN-open-accentuator.md](PLAN-open-accentuator.md).

## Where it sits (and what it is NOT)

Two pipelines exist in this repository and they must not be confused:

**Serving pipeline** (production site + `local/app` replica) — runs per
request:

```
input text ──► tagger (UDPipe / our litlat-bert models)   ← does ALL POS tagging
                  │  lemma + morphology per token
                  ▼
            dictionary lookup (D1 / SQLite `words` table)  ← accented variants
                  │  variant morphology labels matched against tagger output
                  ▼
            accented text
```

**Offline artifact pipeline** (this directory) — runs at build time, tags
nothing, sees no running text. It manufactures a `words`-shaped dictionary
whose variants the serving pipeline *could* consume one day (W3 — explicitly
deferred; nothing here is wired into serving):

```
kaikki.org dump (en.wiktionary extract, CC BY-SA)
      │  extract_lexicon.py
      ▼
lexicon.sqlite          nominals (lemma + accent class), verbs (principal
      │                 parts), forms (192k observed accented forms),
      │                 closed_draft (450 closed-class lemmas)
      ▼
generate_dictionary.py  ── modules below ──►  generated.sqlite
      ▼
parity_report.py  ◄──  local/data/words.sqlite (VDU cache = QA ground truth)
      ▼
reports/parity-vdu.md ──► adjudication ──► parity_vetoes.json (+ rule fixes)
```

## Generation modules (in execution order)

| # | module | what it emits | accent source |
|---|--------|---------------|---------------|
| 1 | `generate_vlkk_names` | given names: fetched singular paradigms + induced plurals; kaikki name entries defer to these | vardai.vlkk.lt (normative authority; fetched by `fetch_vlkk_names.py`) |
| 2 | `generate_nominals` | observed case forms of the 5.8k class-marked noun/adj/name/pron/num lemmas | kaikki inflection tables (observed facts) |
| 3 | `generate_verbs` | observed finite + non-finite verb forms, filtered/repaired by `resolve_verb_form` | kaikki tables + published rules (below) |
| 4 | `generate_other` | observed forms for adverbs, interjections, prepositions, conjunctions, particles, and nominal lemmas without a stress class | kaikki (observed facts) |
| 5 | `generate_closed` | 208 closed-class headwords | our draft, pending VLKK review |
| 6 | `generate_prefixed_verbs` | full paradigms for wordlist-attested prefix+base combos (ištraukti, atsisėsti), incl. reflexive composites | Kushnir §4.4.2 weak/strong transforms over the base cells; weakness evidence read off kaikki's real prefixed entries |
| 7 | `generate_degrees` | comparative/superlative paradigms for adjectives lacking observed rows | per-cell majorities induced from the 13k observed degree rows |
| 8 | `generate_deverbal_imas` | -imas/-ymas action nouns from suffixal verbs (mãtymas, kalbė́jimas) | accent copied from the past-3 stem; primary verbs are lexically split and skipped |
| 9 | `generate_iskas` | -iškas adjectives + -iškai adverbs with base-copied stem accent (vaĩkiškas) | base noun stems from kaikki; no attested base → no answer |
| 10 | `generate_definite` | definite (pronominal) adjective forms (aukštàsis, didỹsis) | VDU 2010 §3.3.10 table 3.24: fixed for class 1, mobile pattern otherwise |
| 11 | `generate_derived` | full paradigms for unknown lemmas that parse as base + self-accented suffix | VDU 2010 App. C suffix table + endings induced from kaikki |

`generate_verbs` additionally emits **negated counterparts** (ne-, nebe-) of
finite forms, infinitives, and participle heads: the negation is stressed
exactly when the tense's root allomorph is weak (nèkeitė, nèneša, nètiki)
and unstressed otherwise (nežinaũ, nedìrba, nemiẽga) — Kushnir §4.4.2 with
the §123 present-weakness criteria (short/lengthened plain stem vowels,
per̃ka/pir̃ko alternation, kal̃ba-type -aR- in -ėti verbs; -o and -i-theme
exceptions gulėti/turėti/galėti are strong; weak present-stem participles
are skipped).

Every emitted form passes through `add_variant`, which applies
`normalize_notation` (repositions marks written in nonstandard places —
circumflex to the second diphthong component `ãusys→aũsys`, to the mixed-
diphthong sonorant `ĩlgas→il̃gas`; an acute never sits on a sonorant. Marks
are moved, never converted: priegaidė is lexical) and drops doubly-accented
template artifacts.

`write_generated` then groups variants per word key, picks the default form
(leftmost stress, circumflex first on ties — dictionary headword convention,
validated 137/138 against VDU), and drops vetoed word keys.

## Rules implemented from published scholarship

| rule | effect | source |
|------|--------|--------|
| prefixed-verb 1/2sg retraction | `àtnešė→àtnešiau`, but `aptìko→aptikaũ` (Saussure stands); prefix-lookalikes screened by weak-root eligibility (-ė past + primary verb; -o presents and -yti pasts never weak) | Kushnir 2019 §4.4.2, §4.4.5 |
| future-3 metatony | long final-syllable stems take the circumflex: `dìrbs→dir̃bs`, `gáus→gaũs`; short nuclei keep the grave (`bùs`) | Stundžia; Kushnir 2019 (17a); VDU cache evidence |
| converb constraint | rows with a stressed prefix are invalid (`per-` excepted) and dropped | Kushnir 2019 §4.5 |
| -t- participle constraint | primary-verb past passive participles copy the *past* stem's accent position and are mobile — kaikki's infinitive-copying rows are dropped; extended-root verbs (`matýtas`) are frozen-strong and kept | Kushnir 2019 §4.7.2 |
| būdinys exclusion | bare-adverbial rows (`kriste`) are undecidable and dropped | adjudication |
| self-accented suffixes | ~100 suffixes fully determine accent + kirčiuotė: `-ýbė 1`, `-ùkas 2`, `-iẽtis 2`, `-eñtas 2`, `-áuskas 1`, the international `-ija/-cija` family | Kazlauskienė–Raškinis–Norkevičius–Vaičiūnas 2010 (VDU monograph) App. C + §3.2.4; Pakerys 2002 for curated additions |
| pretonic -ika/-ikas | stress the pre-suffix syllable: `fìzika`, `lògika`; a long pretonic *a* takes the circumflex (`pãnika`, `matemãtika`); deverbal agentives are self-accented instead (`plėšìkas`, `apgavìkas`) | VDU 2010 §3.3.7 pattern + VDU cache evidence |

## The derivation module's guards (why it can be trusted)

`generate_derived` is the only module that *guesses* (VDU's own architecture
has the same fallback: dictionary miss → "bandyti kirčiuoti priesagas").
A candidate word (from the hermitdave lt_50k frequency list) only produces a
paradigm when ALL of these hold:

1. it parses as base + suffix + an inflection ending induced (majority-voted,
   evidence-thresholded) from kaikki's classed paradigms for that
   (declension, kirčiuotė) pair;
2. the hypothesized lemma itself is attested in the wordlist (kills verb
   forms `atgauti` and names `Artūras` accidentally parsing as derivatives);
3. the lemma does not collide with a known kaikki verb form (`ištraukė`);
4. the base does not start with a verbal/nominal prefix (prefixed derivation
   retracts stress — `núotrauka`, `nevỹkėlis` — beyond these rules), except
   for pretonic internationalisms (`pãnika`, `prãktika`);
5. native diminutive/derivational suffixes additionally need a consonant-final
   base attested as a noun/adjective in the wordlist (`namẽlis`←`namas` yes;
   `modelis`, `daugelis` no).

Derived paradigms never overwrite word keys the observed modules produced.

## Veto policy and provenance tiers

When parity shows our output contradicting VDU and adjudication finds the
*source data* wrong (kaikki entry errors like `blògas`, `bombà`; ambiguous
rule classes like `-antas` where VDU attests both `seržántas` and
`muzikañtas`), the lemma/word goes into `parity_vetoes.json` with the reason
— output is removed, never patched from VDU, so provenance stays fully open.
The word falls back to UNCOVERED ("no answer" beats "wrong answer").

Provenance strings on every entry encode the tier:

| provenance | tier |
|------------|------|
| `open-accentuator:kaikki:<lemma>:...` | observed Wiktionary fact |
| `...:rule=prefix-retraction` / `rule=future-3-metatony` | observed fact repaired by a published rule |
| `open-accentuator:vdu2010-suffix:<lemma>:...` | rule-derived guess (suffix module) |
| `open-accentuator:closed-draft:...` | our closed-class draft, pending VLKK |

## Parity methodology

`parity_report.py` compares every VDU-cache word (10,015 positives) against
the artifact: EXACT (same variant set + default), DEFAULT-MATCH,
OVERLAP (shared variants), NORM-DELTA (no shared variant but VLKK — the
declared normative authority — backs our form; listed with reasons in
`parity_vetoes.json` under `norm_deltas`), DISJOINT (unadjudicated hard
disagreement), UNCOVERED. The standing quality gate is **DISJOINT = 0**:
every hard disagreement must be adjudicated (rule fix, veto, or documented
norm-delta) before committing.

Current state (2026-07-05): covered 6,131/10,015 (61.2%), EXACT 5,058
(82.5% of covered), DISJOINT 0. Most OVERLAP/DEFAULT-MATCH words are cases
where VDU lists additional accent-class variants (`Ãnglija`/`Anglijà`) that
Wiktionary has no facts for. The uncovered mass is dominated by lemmas
absent from Wiktionary whose suffix does not determine their accent.

## The guesser tier (separate artifact)

The residue no open source covers (internationalisms, rare lemmas — the
slice that structurally needs DLKŽ) is handled by a clearly-separated
lowest-confidence tier: `guess_uncovered.py` runs the BSD-licensed
`phonology_engine` (LIEPA's accentuation components, which answer arbitrary
words) over uncovered wordlist/VDU keys and writes `data/guesses.sqlite`
with `liepa-guess` provenance. Benchmarked against the VDU cache: 87.9%
exact-variant and 95.3% stress-position agreement on exactly the
dictionary-gap words. Because ~12% of guesses disagree with VDU, this tier
is **never merged into the main artifact** — the main dictionary keeps its
zero-disagreement gate, and consumers opt into guesses knowingly.

Literature grounding the approach: Kasparaitis (2000, dictionary-based);
Anbinderis & Kasparaitis (2010, decision trees over letter patterns, 95.5%);
Mackevič (2026, VU MSc): a transformer beats rule tools on word-level stress
but loses to VDU's Kirčiuoklis on contextual disambiguation — consistent
with our architecture (rules + dictionary first, guesser last).

## Planned levers (not yet implemented)

- **VLKK name recommendations** as a data source for proper names (official
  state documents are not copyright-protected in Lithuania).
- **Participle declension**: decline participle heads through the adjectival
  paradigms per Kushnir §4.6–4.7 (recovers `apgautì`-class variants).
- **Base-dependent derivation** (Kushnir ch. 3 dominance): suffixes like
  `-iškas` whose accent needs the base's accent — needs base lookup in the
  kaikki lexicon.
