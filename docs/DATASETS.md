# Datasets: sources, licenses, roles

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](GLOSSARY.md).

Hard rule inherited from the start: **provenance-clean**. The VDU
kirčiuoklė cache is QA ground truth and may never be copied into any
artifact we distribute; exclusions (vetoes) are fine, extraction is not.
Every emitted dictionary entry carries a provenance string naming its
source. Full attribution texts: ../ATTRIBUTIONS/.

## Training-grade (open, redistributable)

| dataset | source / license | size | role |
|---|---|---|---|
| kaikki.org en-wiktionary extract | CC BY-SA | 5.8k classed nominals, 844 verbs, 192k observed accented forms | The dictionary pipeline's observed-facts base (accent classes, inflection tables) |
| MATAS v3.0 | CLARIN-LT, CC BY 4.0 | 2.14M tokens gold morphology | Tagger training; joint-model sentences (stress projected from our dictionary) |
| UD ALKSNIS / ALKSNIS v3 | CC BY-SA / CLARIN-LT | 3.6k sentences | Tagger + joint POS GOLD eval (test split never trained on) |
| hermitdave lt_50k frequency list | MIT (OpenSubtitles-derived) | 50k words | Candidate attestation for derivation/synthesis modules; foreign-letter no-stress rows |
| vardai.vlkk.lt | normative state resource | 8,542 names, 4,977 with paradigms | Dictionary names tier (detailed-only policy; 35 internally-inconsistent names dropped) |
| VLKK recommended-stress list + R-13 | normative acts (not copyright-protected) | 376 entries / 689 forms; function-word list | Normative dictionary tier; closed-class citations |
| e-LKŽ (lkz.lt) | copyrighted; individual facts cited de minimis | ~50 cited entries | Closed-class extras + audit citations only — never bulk-extracted |
| **Our generated dictionary** (generated.sqlite) | ours; open provenance throughout | 574,749 words / 710k stressed variants | Stress-model training data; dictionary serving tier; teacher layer |

## Evaluation-only (never trained on)

| dataset | source / license | size | role |
|---|---|---|---|
| VDU kirčiuoklė cache (words.sqlite) | fetched per-word from kalbu.vdu.lt; NOT redistributable | 10,015 positives | Dictionary parity QA (DISJOINT=0 gate); gap-slice guesser eval |
| LRT corpus | fetched articles (attribution sidecar); eval-only | 40 articles, 37.7k tokens | Unseen-text eval; silver truth from VDU+UDPipe; 453-entry human audit overlay (tracked: data/eval/lrt-silver-audit.json) |
| Kirčiuotų tekstų chrestomatija (Kavaliauskas 2014) | copyrighted teaching book; recovered via Wayback Machine (URL in extract_chrestomatija.py); local eval use only, never redistributed, never trained on | 2,969 sentences / 43.2k tokens, hand-stressed | The GOLD accent benchmark (community standard; published baselines: VU transformer 0.711, VDU Kirčiuoklis 0.702 sequence acc) |

## Guess/derived artifacts (ours, clearly tiered)

| artifact | provenance | role |
|---|---|---|
| guesses.sqlite | `agree-nn-liepa` (99.5% tier) + `liepa-guess` per word | Precomputed OOV answers, separate from the zero-disagreement dictionary |
| MATAS joint dataset (joint/data) | MATAS gold POS + OUR dictionary's projected stress (68% of letter tokens; 79% of homographs resolved by gold morphology) | Joint-model training; provenance-clean silver |
| teacher-labeled corpora (data/teacher/) | calibrated multi-system consensus (SPEC34) | Future training data with measured purity |

## Sources evaluated and REJECTED (so nobody retries them blind)

- **lt.wiktionary**: ~0 accent info (1/10 sampled pages).
- **Foreign Wiktionary editions** (ru/de/fr… via kaikki): ≤3% coverage
  of our residue, mostly duplicating en.wiktionary.
- **DLKŽ / BLKŽ (ekalba.lt)**: authoritative but closed; no API; the
  structural reason a dictionary-gap residue exists at all.
- **aleksas/matas-alksnis-stressed + LKI scrapes** (incl. a DLKŽ zip):
  VDU-machine-stressed (provenance-tainted) and copyright-grey
  respectively — off-limits under our rules.
- **CLARIN-LT / ELG / ELRC full sweep** (2026-07): zero orthographic
  stress corpora; speech corpora (LIEPA-3, SING) are phonemic TextGrids,
  not accent-marked text. "Tartis" (VDU 2021, 150k stressed words) is
  all-rights-reserved lookup-only — permission email is the path.
- **AAI-Labs 56k stressed book corpus** (the VU thesis's training set):
  proprietary.
- **Sakrament 1M stressed corpus** (Anbinderis 2010): never released,
  presumed lost.
