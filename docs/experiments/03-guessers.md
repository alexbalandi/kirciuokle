# Experiments: OOV stress guessers

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](../GLOSSARY.md).

Goal: answer words the dictionary doesn't cover. Bench harness:
bench_guessers.py; identical slices everywhere (held = 2% in-domain
holdout, seed 20260705; gap = 2,751 VDU-cache words the dictionary
misses). Report: ../../local/accentuator/reports/guesser-bench.md.

## E3.1 Naive suffix trie (baseline)
Longest-suffix majority vote. Held 88.1%, gap 50.5% (always answers).
Established the floor and the failure mode: majority-voting on
ambiguous patterns and projecting stress beyond the matched window.

## E3.2 Anbinderis & Kasparaitis 2010 faithful replication
Read the paper (ITC 39(1)); implemented the real method: SHORTEST
UNAMBIGUOUS letter-sequence rules from beginning/ending tries, rule may
only place stress within matched letters, unmatched → abstain; end-bgn
pipeline. 119k+106k rules from 524k types. Held 97.2%@97.6% answered —
matches the paper's regime (their 95.5% was token-level running text ≈
in-domain). Gap: 67.0% of 63.0% answered. Lesson: their result was
never about OOV generalization; the gap is lexical.

## E3.3 LIEPA / phonology_engine (BSD)
The only fully open native LT stresser (Kasparaitis-lineage lexicon
compiled into the binary). Gap 88.3% exact / 95.6% position of 99.3%
answered — wins the gap because its lexicon CONTAINS those words
(lookup, not inference). In-domain only 79.1% (conventions differ from
our dictionary). Adapter hygiene mattered: unmarked and double-marked
outputs must abstain (count marks on NFD!).

## E3.4 Neural guessers (see 04-stress-models.md for architecture)
v1 word-only: held 97.9%, gap 59.5% — best pattern model in-domain,
still loses the gap to the lookup. Confirms E3.2's lesson at scale.

## E3.5 Agreement ensemble — the key result
Where the neural model and LIEPA independently produce the IDENTICAL
form (50.5% of gap words): **99.5% exact** — dictionary-grade answers
on words no dictionary covers. Production guess tier became the cascade
`nn&liepa+liepa` (9,800 agreement-backed @99.5% + 13,649 liepa-only
@~76%, distinct provenance so consumers trust-filter). This result
seeded the teacher-labeler (06).

## E3.6 Research context (verified 2026-07)
Anbinderis 2010's own implementation: never released (shipped only in
the dead Egidius synthesizer; Sakrament's 1M-token corpus lost).
aleksas/lt-stress-project (2020): released t2t weights but unlicensed +
VDU-distilled. VU 2026 thesis code open, weights/data closed. No open
gold corpus except the chrestomatija (see 06 / DATASETS.md).
