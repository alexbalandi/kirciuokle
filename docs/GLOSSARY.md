# Glossary

Terms of art used across these docs — both this project's own jargon and
the Lithuanian linguistics needed to read the results. Linked from every
doc; if you meet an undefined term anywhere, it's a docs bug.

## Lithuanian linguistics (the domain)

- **Stress mark / kirčio ženklas** — one of three diacritics on the
  stressed syllable: **grave** `à` (kairinis — short stressed syllable),
  **acute** `á` (dešininis — long syllable, falling tone), **tilde /
  circumflex** `ã` (riestinis — long syllable, rising tone).
- **Priegaidė** — the tone contour (falling vs rising) on LONG stressed
  syllables; the acute-vs-tilde distinction. Lexical: it can be the only
  difference between words (áušta "dawns" vs aũšta "cools").
- **Kirčiuotė / accent paradigm (AP1–AP4)** — one of four stress-mobility
  classes a Lithuanian noun/adjective belongs to; determines which
  inflected forms move the stress to the ending. AP1 = fixed stem
  stress; AP3/AP4 are "mobile".
- **Mixed diphthong** — vowel + sonorant (il, ir, al, an, …); the tilde
  is written on the sonorant (šil̃tas).
- **Homograph** (here) — same spelling, different stress depending on the
  morphological reading (galvà nom.sg vs gálvą acc.sg) or lemma (ỹra
  "is" vs yrà "decays").
- **Kirčiuoklė / accentuator** — a tool that adds stress marks to text.
  "The VDU kirčiuoklė" = the reference tool at kalbu.vdu.lt.
- **mi label** — a morphology label string in VDU's traditional-grammar
  notation, e.g. `dkt., mot. g., vns. kilm.` (= noun, feminine,
  singular genitive). Dictionary variants carry them; our taggers can be
  matched against them.
- **DLKŽ / BLKŽ** — the authoritative (closed) dictionaries of standard
  Lithuanian; **VLKK** — the State Commission of the Lithuanian
  Language, the normative authority we treat as arbiter.

## Data & evaluation jargon (ours)

- **Silver (data/truth)** — labels produced automatically by a trusted
  pipeline rather than a human; good but with a measurable error rate.
  Our silver generator = VDU kirčiuoklė + UDPipe tagging +
  morphology-scored disambiguation.
- **Audited silver** — silver whose suspicious slices were adjudicated
  by humans/agents against citable sources, with a correction overlay
  applied at scoring time.
- **Gold** — hand-curated ground truth. For accents we have exactly one
  gold set (the chrestomatija); for morphology, the ALKSNIS test split.
- **Parity (report/gate)** — comparison of our generated dictionary
  against the VDU cache. Buckets: **EXACT** (same variants + default),
  **DEFAULT-MATCH**, **OVERLAP** (share some variants), **NORM-DELTA**
  (hard disagreement where the normative authority backs OUR form —
  citation required), **DISJOINT** (unadjudicated hard disagreement),
  **UNCOVERED** (we give no answer). **The DISJOINT=0 gate**: no commit
  while any hard disagreement is unadjudicated.
- **Veto** — a QA exclusion in parity_vetoes.json: a lemma/word whose
  generated output is removed because the source data or a rule is
  wrong there. Vetoes remove output, never patch it from the reference
  ("no answer beats wrong answer").
- **Covered / uncovered** — whether a word has an entry in our
  dictionary artifact.
- **Held(-out) slice** — the 2% of dictionary words (grouped by word,
  fixed seed 20260705) excluded from training and used to measure
  in-distribution generalization.
- **Gap slice** — the ~2.6k words the VDU cache knows but our
  dictionary does NOT cover: the out-of-vocabulary test bed for
  guessers. "The gap is lexical" = these words' stress can't be
  predicted from spelling patterns; you have to have seen the word.
- **Register gap** — a quality drop caused by text style mismatch
  (models trained on news/dictionary text scoring ~9pp lower on
  poetry/classics).
- **Answered / abstention** — whether a system produces a form for a
  token at all; abstaining is legitimate (a lower tier or nothing
  handles it) and is never counted as a wrong answer.
- **Exact / position** — exact = right letter AND right mark type;
  position = right letter, any mark. The difference isolates mark-type
  (priegaidė) errors.
- **Exact-over-all** — answered × exact: single-number usefulness of a
  system that may abstain.
- **Sequence accuracy** — share of sentences where EVERY token is
  exact; the metric the published literature reports.
- **Foreign-unmarked** — tokens standard Lithuanian text correctly
  leaves without stress marks (unadapted foreign names, acronyms).
  Leaving them unmarked is CORRECT behavior; they're excluded from
  exact/position and tracked by their own diagnostic.

## Systems & pipelines (ours)

- **The dictionary (artifact)** — generated.sqlite: 575k accented words
  built from open sources (docs/ACCENTUATOR.md), zero unadjudicated
  disagreements with the VDU cache.
- **Guess tier** — a separate artifact (guesses.sqlite) answering
  uncovered words, with per-word provenance naming which backend
  answered; never merged into the dictionary.
- **LIEPA / phonology_engine** — a BSD-licensed Python package wrapping
  the LIEPA speech synthesizer's native accentuation components (a
  lexicon + rules); answers arbitrary words.
- **Agreement ensemble / nn∧liepa** — accept an answer only when two
  independent systems produce the identical form; measured at ~99.5%
  exact where they agree.
- **Cascade** — ordered backends where the first answer wins and
  abstentions fall through (e.g. `nn&liepa+liepa`).
- **No-dict pipeline ("nodict")** — accentuation with NO dictionary
  lookup: our tagger produces a morphology label per token, the stress
  model predicts the mark from (word, label). The project's headline
  architecture; the dictionary remains as a comparison baseline.
- **Label bridge** — selecting, per token, the best-matching label from
  the CLOSED vocabulary of dictionary mi labels given the tagger's
  output (the stress model was trained on those exact strings).
- **Joint model** — one encoder, one pass per sentence, two heads (POS +
  stress); the current best system.
- **Projection (stress)** — assigning a stress target to a corpus token
  by looking its word + gold morphology up in our dictionary; tokens
  with no unique match get no stress supervision (loss-masked).
- **Rehearsal mixture** — fine-tuning on new data mixed with a slice of
  the original training data so the shared encoder doesn't drift away
  from the frozen/other head's needs.
- **Polish pass** — a short continued-training run from a finished
  checkpoint at ~0.1× learning rate with per-epoch dev selection.
- **Teacher-labeler / etalon teacher** — the calibrated consensus
  annotator: collect every system's opinion per token, accept a label
  only if its agreement pattern's measured-against-gold accuracy clears
  a threshold; produces training data with a purity certificate.
- **Agreement pattern / stratum** — which subset of systems produced
  the identical answer for a token (e.g. "vdu+joint+liepa"); each
  pattern's accuracy is measured on gold during calibration.
- **Provenance(-clean)** — every emitted entry names its source in a
  machine-readable string; nothing is ever copied from the VDU cache
  or other closed references into our artifacts.
- **VDU cache** — our stored per-word answers from the VDU kirčiuoklė
  (10,015 words), used as QA ground truth only.
- **Slots** — the agreement-feature projection of a morphology tag (POS
  family + case/gender/number/tense/person/voice/degree) used to score
  tag matches; see score_tags.
