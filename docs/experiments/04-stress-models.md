# Experiments: neural stress models v1–v3

Architecture details: ../ARCHITECTURES.md §2. All runs: same seed/split
(word-key-grouped from v2 on), litlat-bert encoder, 3080 Ti laptop GPU,
~40–90 min per full run. Trainer: train_stress_nn.py.

## E4.1 v1 — word → stress (SPEC22 precursor)
524k dictionary default forms. Held-out 97.9% exact @100% answered
(99.6% @conf 0.9); gap slice 59.5%. Proved the hierarchical
char-attention head + validity mask; confidence works as an abstention
knob.

## E4.2 Validity-mask tightening (data-audited phonology)
Audited all 710k stressed dictionary variants for attested
(letter-context, mark) combos; banned the impossible cells (grave on
long vowels: 0–3 noise rows; bare-short-i non-grave: 1 vs 30,028; etc.).
Free accuracy for every guesser; targeted the ~7pp exact↔position
mark-type error band.

## E4.3 v2 — label conditioning (SPEC22)
(word, morphology label, form) triples via tokenizer text pairs — no
architecture change. 1.21M rows, 12.7k homograph words. Homograph
switching 72.5% row-exact (unconditioned ceiling ≈50%); unconditioned
mode IMPROVED to 98.4% as a multi-task side effect; gap unchanged
(lexical, labels can't help). End-to-end through the real tagger:
labels contributed +5.5pp on unseen LRT.

## E4.4 v3 — learned no-stress class (SPEC27)
Virtual softmax cell (mean-pooled linear) + 2,460 no-stress rows
(VDU-cache unmarked + foreign-letter wordlist words; NEVER from eval
data) + 4th epoch. Held-out foreign words correctly left unstressed:
75% (v2: ~0% capability). Gap 65.7% (+4.8pp — the extra epoch;
mobile-paradigm endings were underfit). Audited LRT: 83.5% exact /
86.8% position (v2: 81.1/83.6); foreign-abstention 62.8% @conf0
(84.4% @0.9); labels now +7.9pp.

## E4.5 Polish + LR sweep (SPEC30, user-suggested)
Low-LR restart from the trained checkpoint with dev-based best-epoch
selection. Constant 0.1× full epoch: +0.08pp dev; cosine quarter-probes
0.3×/0.1×/0.03×: +0.03/+0.05/+0.01pp. Verdict: window wide and shallow;
recipe `--lr-scale 0.1`, either schedule; real but marginal next to
architecture changes. Dev (dictionary-projected) is saturated — rank on
ALKSNIS/LRT/gold instead.

## Overfit analysis (asked and answered)
Train ≈ dev (99%+) on the same distribution → no classical overfit. The
dev→LRT gap decomposes: ~4pp reference disagreement (our conventions vs
VDU silver — the dictionary itself caps at ~96% against it), ~5–6pp
from the OOV/unprojectable population dev never scores, ~1–2pp genuine
generalization. Distribution mismatch, not memorization damage.
