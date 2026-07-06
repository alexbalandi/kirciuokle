# Experiments: the open dictionary

Goal: a provenance-clean accentuation dictionary from open sources,
QA'd against the VDU cache at DISJOINT=0. Architecture and rules:
../ACCENTUATOR.md. Coverage arc: 62.4% → 75.3% over ~10 module rounds.

## E2.1–E2.4 Core modules (kaikki + rules)
Observed kaikki facts (nominal paradigms, verb principal parts, 192k
accented forms) + published-rule repair: Kushnir 2019 verbal retraction
/ Saussure / future metatony / participle constraints; VDU 2010
Appendix C self-accented suffixes (93 + curated extras). Each
suffix-module round: over-generation → guards (lemma attestation,
verb-form collision, prefix screens) → 124→0 disjoints in 3 rounds.
Notable data-beats-theory moment: pretonic -ika takes a LONG a
(pãnika) — VDU evidence corrected the initial guess.

## E2.5 Prefixed-verb synthesis
Full paradigms for wordlist-attested prefix+base combos via a weakness
EVIDENCE map read off kaikki's real prefixed entries (acute pasts split
lexically: ištráukė vs pàbaigė). Evidence pollution fixed by excluding
ne-/nebe-/per- lemmas and vetoed lemmas. 24→0 disjoints.

## E2.6 VLKK tiers
vardai.vlkk.lt full crawl (8,542 names; detailed-only policy; per-name
pages return HTTP 404 WITH content — parser accepts big bodies); the
recommended-stress-variants list (376 normative entries, K-nn ids in
provenance); R-13 function words. Letter-page/detail-page priegaidė
conflicts (35/179 names) → cross-check guard drops disputed names.

## E2.7 Closures
Apocope module (-ti→-t, -ki→-k with accent kept: ateĩt, supràst),
candidate attestation widened to VDU cache KEYS (keys only —
provenance-clean), 50 cited closed-class extras (dė̃kui, galbū́t…),
nebė̃ra/tebė̃ra into the būti table. Audit-driven fixes: indėlis and
klimatas vetoed (lexicalized AP1 nouns the suffix rules wrongly
claimed), pereiti vetoed (kaikki lacks dominant per- stress, 177 bad
cells).

## Standing numbers (2026-07-06)
574,749 words; parity: covered 7,540→7,537 after honesty vetoes
(75.3%), EXACT 6,154, NORM-DELTA 9 (all cited), DISJOINT 0. The
uncovered residue (~2,475 VDU words) is lexical — the slice that
structurally needs the closed DLKŽ.

## Lessons
- Adjudicate, never average: parity_vetoes.json carries a reason and
  citation for every exclusion; vetoes remove, never patch.
- Names collide with common words (Rojus/rojus, Valys gen. vs valiõ) —
  the common reading wins, the name key gets vetoed.
- Mechanical coverage ≠ benchmark coverage: the VLKK crawl added 190k
  words but +7 parity words (names are rare in the QA set, common in
  real text).
