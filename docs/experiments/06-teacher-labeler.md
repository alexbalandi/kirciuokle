# Experiments: the teacher-labeler (in progress)

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](../GLOSSARY.md).

Goal: manufacture training data with MEASURED purity from any Lithuanian
corpus, because gold accent data cannot be bought or downloaded
(../DATASETS.md, rejected-sources). Design: SPEC34; code under
local/accentuator/teacher/.

## The founding observations
1. Where two independent systems agree on a form, accuracy jumps to
   ~99.5% (E3.5, nn∧liepa on gap words).
2. Every system we hold fails DIFFERENTLY: vdu-udpipe 95.8% on gold but
   systematically over-mobilizes AP1/AP2 oblique plurals; LIEPA has
   98.1% position precision but mark-type errors; the dictionary is
   narrow but precise; the joint model reads context.
3. Training tolerates partial coverage (per-token loss masking) — so a
   teacher may abstain freely; purity is the product, coverage is
   sacrificial.

## Design
- `collect_layers.py`: per token, gather vdu-silver / joint / liepa /
  dict+labels (accents) and joint / released-tagger (POS). No external
  calls at labeling time.
- `calibrate.py`: measure EVERY agreement pattern's accuracy against
  gold — accents on the chrestomatija (43.2k tokens), POS on the
  ALKSNIS gold test. Output: strata tables (the teacher's certificate).
- `label.py`: accept a token's label iff its pattern clears the
  threshold (defaults: 0.98 accents, 0.95 POS); output in the joint
  dataset builder's format with purity/coverage stats.

## Calibration results (2026-07-06, chrestomatija accents / ALKSNIS POS)

ACCENTS: with the default 0.98 stratum threshold, coverage
**84.8% at 99.52% purity** on literary gold — the teacher labels harder
text more accurately than any single component (best single: vdu-udpipe
95.8%). Full-consensus strata sit ≥99%; the disagreement tail (e.g.
`vdu vs joint`: 2.1% of tokens, 44% accuracy) is masked, not learned.

POS: **collapsed to 0% coverage at the 0.95 threshold — by design, and
the collapse was the lesson.** The `joint=tagger` agreement stratum
(97.6% of tokens) reaches only 85.9% full-label accuracy because the
two voters are CORRELATED: the joint model's encoder warm-started from
that tagger. Agreement only purifies INDEPENDENT lineages (the accent
voters are; the POS voters aren't). Teacher v2 fix: add a
UDPipe-lineage POS voter at silver-build time and recalibrate.
Interim: literary fine-tuning proceeds with POS loss masked on
teacher-labeled tokens + MATAS rehearsal rows carrying gold POS.

## Intended first uses
1. Public-domain Lithuanian classics (Maironis, Žemaitė…) →
   teacher-label → fine-tune the joint model (rehearsal mixture) →
   verify on the UNTOUCHED chrestomatija whether the ~9pp register gap
   closes.
2. Scale modern data: unlimited LRT text at teacher purity instead of
   raw silver.

## Guardrails
- The chrestomatija calibrates the teacher and judges the student — it
  must never be trained on itself.
- Strata below threshold are dropped, not down-weighted (a mislabeled
  token teaches a wrong lexical fact forever; a masked token costs
  nothing).
- Recalibrate whenever any component system changes.
