# Joint POS + accentuation model — full evaluation

Single litlat-bert encoder, one forward pass per sentence, two heads:
per-token UPOS|FEATS classification + per-token hierarchical stress head
(char queries over the word's subword span, (char × mark) + no-stress
grid). Training: 120k MATAS sentences (gold morphology, CC BY 4.0) with
stress projected from our own dictionary (1.10M supervised tokens, 68.0%
of letter tokens; 79.4% of homograph lookups resolved by gold morphology);
warm start from the released -vdu tagger encoder and the v3 stress head;
2 epochs + one polish epoch at 0.1× constant LR (dev-selected best).

Checkpoint: `local/accentuator/joint/checkpoints/joint_v1_polish.best.pt`
(gitignored; reproduce via joint/build_joint_dataset.py + train_joint.py).

## Headline: joint vs the two-model stack (audited LRT, 37.7k unseen tokens)

| | two-model (tagger + stress v3) | joint single pass |
|---|---:|---:|
| stress token exact | 83.5% | **87.9%** |
| stress token position | 86.8% | **89.5%** |
| foreign-unmarked correctly untouched | 62.8% | **76.1%** |
| POS on ALKSNIS gold test (combined label) | 86–89% slot band (released) | **88.9%** (UPOS 96.9%) |
| parameters | ≈307M | **156M** |
| passes per sentence | 2 + label bridge | **1** (≈1,660 tok/s GPU) |

Raw (unaudited) LRT: 86.9% exact / 86.5% position. MATAS dev: 98.9% POS,
99.3% stress row exact (saturated — dev is dictionary-projected, so this
mostly measures projection recall).

Why it wins: the stress head reads full sentence context directly from
the encoder — no lossy word+label bottleneck, no label-bridge errors —
and trains on contextual running-text distribution instead of isolated
dictionary rows.

## Polish pass + LR sweep (user-suggested)

Low-LR polish from the trained checkpoint, dev combined score deltas:

| variant | delta vs init |
|---|---:|
| constant 0.1×, 1 full epoch | +0.08pp (best) |
| cosine 0.1×, quarter epoch | +0.05pp |
| cosine 0.3×, quarter epoch | +0.03pp |
| cosine 0.03×, quarter epoch | +0.01pp |

Verdict: the polish window is wide and shallow; `--lr-scale 0.1` with
either schedule is the documented recipe. Gains are real but small next
to the architecture win. Dev is saturated — future selection should
weigh ALKSNIS/LRT, not MATAS dev.

## Open items

- ONNX export of the joint model (single ~156MB int8 artifact) + the 80%
  embedding-vocab prune → the browser-deployable single-model bundle
  (see onnx/BROWSER.md for the measured math).
- Homograph-switch behavior inherits sentence context here; a dedicated
  homograph eval slice on running text is future work.
