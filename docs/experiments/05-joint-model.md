# Experiments: the joint POS + accentuation model

> Terms of art (gap slice, silver, DISJOINT, priegaidė…) are defined in [GLOSSARY.md](../GLOSSARY.md).

The architectural bet (user's design): one encoder, one pass per
sentence, POS head + per-token char/mark stress head, shared loss, warm
starts from the released tagger and stress_nn3. Architecture:
../ARCHITECTURES.md §3. Code: local/accentuator/joint/. Full numbers:
../../local/accentuator/reports/joint-eval.md.

## E5.1 Dataset projection (SPEC29)
MATAS gold morphology + stress projected from OUR dictionary via
slot-matched variant selection: 2.08M tokens, 1.10M stress-supervised
(68.0% of letter tokens), 79.4% of homograph lookups resolved by gold
morphology; foreign-letter tokens → no-stress targets; everything else
loss-masked. Provenance-clean silver (no external accent source).

## E5.2 Smoke (2k sentences, CPU, 250 steps)
POS 85.8% ALKSNIS label, LRT stress 57.3% audited — warm starts
connected (encoder from runs/litlat__gen2__vdu/best; 20 stress-head
tensors from v3). Signal that the architecture would work.

## E5.3 Full run (120k sentences × 2 epochs) + polish
Result vs the two-model stack on identical evals:

| | two-model (tagger + v3) | joint |
|---|---:|---:|
| audited LRT token exact | 83.5% | **87.9%** |
| audited LRT position | 86.8% | **89.5%** |
| foreign-unmarked correct | 62.8% | **76.1%** |
| POS ALKSNIS (combined label) | 86–89% band | **88.9%** (UPOS 96.9%) |
| params | ≈307M | **156M** |
| passes | 2 + label bridge | 1 (~1,660 tok/s GPU) |

Why it wins: the stress head reads full sentence context directly (no
word+label bottleneck, no label-bridge error class) and trains on
contextual running-text distribution. Polish pass: +0.08pp dev (kept as
.best.pt; recipe from E4.5).

## E5.4 Gold benchmark placement (chrestomatija)
joint 86.9% token exact / 89.5% position, sequence 28.4%; references on
the same extraction: vdu-udpipe 95.8%/96.8% (seq 61.1%), liepa 76.7%
(98.1% position of answered), dict-default 67.8%. Cross-paper: the VU
thesis transformer ≈97.9% token-level (trained IN-register on 56k
literary samples). Diagnosis: a ~9pp literary register gap — our
training data is dictionary+news; the benchmark is poetry/classics
(archaic forms: Pasiklýdot, snapuõs). The audited-LRT number (87.9%)
predicted the gold number (86.9%) within a point → the silver+audit
ladder is calibrated.

## E5.5 Literary fine-tune (v2-literary, SPEC35/36)
220k tokens of public-domain classics (lt.wikisource, PD-author
whitelist, chrestomatija FIREWALL — 159 sentences dropped for matching
gold, proving the firewall necessary), teacher-labeled at 99.5% purity,
mixed with a 25% MATAS rehearsal slice, fine-tuned 2 constant-LR
epochs (polish recipe) from joint_v1_polish.best.

Verdict on the untouched gold: **89.9% token exact / 92.1% position /
37.6% sequence** (was 86.9/89.5/28.4) — a third of the register gap to
the 95.8% VDU ceiling closed by one small corpus. Modern register
IMPROVED (audited LRT 89.8%, was 87.9%); POS held within noise
(88.76% vs 88.86%) — rehearsal works. Trade-off found: foreign-word
abstention dipped 76%→67% (literary teacher data has no foreign
words) — future corpus mixes should retain some foreign-unmarked rows.
Checkpoint: joint/checkpoints/joint_v2_literary.best.pt.

## Open levers
Scale the teacher loop (more PD volume, modern LRT at teacher purity,
foreign-row retention); teacher v2 with an independent POS voter;
ONNX export + 80% embedding prune → single ~110–120MB int8 browser
artifact (onnx/BROWSER.md math).
