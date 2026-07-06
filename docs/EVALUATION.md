# Evaluation methodology

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](GLOSSARY.md).

The project's quality claims climb a four-rung ladder. Each rung exists
because the one below it was caught being insufficient.

## Rung 1 — Parity vs the VDU cache (dictionary artifact QA)

`parity_report.py` compares every VDU-cache word (10,015) against
generated.sqlite: EXACT / DEFAULT-MATCH / OVERLAP / NORM-DELTA
(VLKK-backed divergence, citation required) / DISJOINT / UNCOVERED.
**Standing gate: DISJOINT = 0** — every hard disagreement gets
adjudicated before commit (rule fix, sourced veto, or documented
norm-delta). "No answer beats wrong answer": vetoes remove output,
never patch it from the reference. Current: 75.3% covered, 6,154 exact,
0 disjoint.

## Rung 2 — Silver on unseen text (pipeline-level eval)

Fresh LRT articles (never in any training set) accented by the
independent production stack: VDU kirčiuoklė + UDPipe (LINDAT) +
morphology-scored disambiguation (`build_silver_truth.py`, resumable,
throttled). Key design point: the reference pipeline shares NO
components with the systems under test — our tagger errors surface as
real misses instead of cancelling out.

## Rung 3 — Audited silver (because silver lies a measured amount)

798 suspect types (dictionary-vs-silver conflicts, silver-unmarked
words, context-unstable types, LIEPA disputes) were adjudicated by three
parallel audit passes against e-LKŽ / VLKK / wiktionary paradigms,
yielding a 453-entry overlay (37 silver corrections, 133 both-valid
widenings, 39 exclusions, 241 foreign-unmarked, 3 lenient homographs).
Evals report raw AND audited numbers. Two lessons the audit bought:
VDU systematically over-mobilizes AP1/AP2 oblique plurals (our
dictionary was right); unadapted foreign words are CORRECTLY unmarked —
they form their own eval category and are never counted as errors
(leaving them unmarked is desired model behavior, later a learned
no-stress class).

## Rung 4 — Gold (chrestomatija)

The community benchmark: hand-stressed literary texts, published
baselines (VU thesis transformer 0.711 sequence acc, VDU Kirčiuoklis
0.702 — cross-paper comparison is indicative only; extraction and
sample lengths differ). Our extraction: 2,969 sentences / 43.2k tokens.
Never trained on, never redistributed. It calibrated the whole ladder:
the audited-LRT number for the joint model (87.9%) predicted its gold
number (86.9%) within a point — the silver+audit methodology holds.

## Metric definitions (identical in every harness)

- **answered** — share of tokens/words the system gives any answer for
  (abstention is a first-class outcome, it cascades to the next tier).
- **exact** — NFC-normalized full match: right letter AND right mark.
- **position** — right stressed letter, any mark (the exact↔position
  gap isolates mark-type errors, ~7pp for most systems).
- **exact-over-all** — answered × exact; the number that matters when
  abstentions fall through to a lower tier.
- **sequence accuracy** — sentence counts only if every word token is
  exact (the thesis-comparable metric; brutal on long sentences).
- **homograph switch** — held-out words whose stress flips with the
  morphology label; row-exact and word-all-exact (unconditioned models
  cap at the majority-form share).
- **foreign-unmarked diagnostic** — share of gold-unmarked foreign
  tokens the system also leaves unmarked (higher is better; excluded
  from exact/position denominators).

## Standing rules

- Same seed (20260705) and same holdout split across every stress-model
  experiment — numbers are comparable across runs and versions.
- Holdout grouping by WORD KEY wherever labels multiply rows.
- Candidate expansion may read reference KEYS (word lists), never
  reference accent data.
- Eval corpora never enter training; the chrestomatija in particular is
  the register-gap thermometer and burns if touched.
