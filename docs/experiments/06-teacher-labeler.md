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

## Calibration results
(to be filled when SPEC34 lands — expected: full-agreement stratum
≥99%, usable coverage well above 50%)

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
