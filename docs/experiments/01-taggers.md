# Experiments: POS taggers

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](../GLOSSARY.md).

Goal: replace the external UDPipe dependency with own models at equal or
better quality, on open-licensed data. Full benchmark table:
../../local/README.md.

## E1.1 Encoder bake-off
Compared Lithuanian ModernBERT (VSSA-SDSA/LT-MLKM-modernBERT),
EMBEDDIA/litlat-bert, xlm-roberta-base on identical MATAS+ALKSNIS data.
**litlat-bert won by 5.4pp** over the Lithuanian-specific ModernBERT.
Lesson: language-specific pretraining is not automatically better;
measure.

## E1.2 Corpus recipe
MATAS v3.0 (2.14M tokens, CC BY 4.0) deduplicated + ALKSNIS train,
dropping any training sentence whose normalized text appears in ALKSNIS
dev/test. Labels `UPOS|FEATS` with sorted-FEATS canonicalization so
MATAS and ALKSNIS analyses share label strings.

## E1.3 Head/pooling matrix
Configurable head shape (joint combined-label vs factored) × subword
pooling (first vs mean). Shipped: combined label + first-subword.

## E1.4 GPU campaign + self-training
Fine-tuned to UDPipe-class: slots 89.1% (-ud) vs UDPipe 89.2% on the
full 684-sentence ALKSNIS gold test. A constrained-decoding
self-training pass produced an NC-free lineage at teacher quality.

## E1.5 Released models (HF, CC BY-SA 4.0)
`litlat-bert-lithuanian-morphology` (-ud), `-full` (adds full FEATS +
lemma scripts; CoNLL-18 lemmas 94.7 vs UDPipe 92.9), `-vdu`
(traditional-grammar categories for accentuation matching; 20/20 on the
sample text, 323/370 Wikipedia homograph agreement). ONNX INT8 ~880
tok/s CPU.

## Dead ends
- Trankit: 2021 codebase no longer installs; model host unreachable.
- Stanza-lt: measurably weaker (slots 84.7%), kept only as the simplest
  docker fallback.
