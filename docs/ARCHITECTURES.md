# Model architectures

Everything uses the same encoder: **EMBEDDIA/litlat-bert** (XLM-RoBERTa
architecture, 12 layers, hidden 768, ~84k-token trilingual LT/LV/EN
vocabulary; encoder ≈150.7M params, of which the embedding matrix is
64.7M / 43%). It won the encoder bake-off against Lithuanian ModernBERT
by 5.4pp on identical tagging data — measure, don't assume.

## 1. POS taggers (released 2026; see ../local/README.md for benchmarks)

Token classification over CoNLL-U-style labels with configurable head
shape and subword pooling (first-subword alignment is the default that
shipped). Three released variants differ by label space, not
architecture: `-ud` (UPOS + core FEATS), `-full` (full FEATS + lemma
scripts), `-vdu` (traditional-grammar categories: DET→PRON, AUX→VERB —
built for accentuation-variant matching). Serving contract:
UDPipe-compatible `POST /process` (local/tagger-hf/server.py), ONNX INT8
export for CPU (~880 tok/s).

## 2. Stress models v1–v3 (local/accentuator/train_stress_nn.py)

Word-level: input is one word (optionally + a morphology label), output
is a stress placement.

**The hierarchical char-placement head** (all versions): the litlat
tokenizer's subwords don't align with stress positions, so one learned
query per CHARACTER (char embedding + within-word position embedding,
LayerNorm) cross-attends (8-head MHA) into the word's subword hidden
states; residual + FFN(×2 hidden) + LayerNorm; a final linear scores the
3 accent marks per character. Training objective: ONE masked softmax
over the flattened (char × mark) grid.

**The validity mask** (`train_guesser.valid_target`) removes
linguistically impossible cells, tightened by an audit of all 710k
stressed dictionary variants: grave never on long vowels (ą ę ė į ų ū y),
bare short i takes only grave, mixed-diphthong first-element i/u only
grave, sonorants (l m n r as second diphthong element) only tilde. Every
banned cell is an error class the model cannot produce. Applied at
training (loss) and inference.

- **v1** — word only. In-domain 97.9% exact.
- **v2 (`--labels`)** — conditioning on the morphology label via
  tokenizer TEXT PAIRS: `tokenizer(word, label)`; the char queries read
  the label subwords through the same cross-attention — zero
  architecture change. Trained on (word, label, form) triples exploded
  from dictionary variants (1.21M rows); holdout grouped BY WORD KEY so
  homograph labels cannot leak across the split. Homograph switching
  72.5% row-exact (unconditioned ceiling ≈ majority share ~50%).
- **v3 (`--v3`)** — adds a learned NO-STRESS cell: `Linear(hidden,1)`
  on the mean of char representations, appended to the flattened
  softmax. Trained with (word → no accent) rows from VDU-cache unmarked
  entries + foreign-letter wordlist words (never from eval data). This
  makes "leave foreign words unaccented" a learned prediction, not a
  confidence-threshold artifact. Softmax confidence doubles as an
  abstention knob in all versions.

## 3. Joint POS + accentuation model (local/accentuator/joint/)

One encoder, one forward pass per SENTENCE, two heads, summed loss:

- **POS head**: per word token (first-subword representation), linear
  classification over the combined `UPOS|FEATS`-style label set (804
  labels in the full dataset). Token-mean cross-entropy, pad-ignored.
- **Stress head**: the v3 hierarchical head applied per word token
  INSIDE the sentence — char queries for each word attend only to that
  word's own subword span (span attention mask); masked (char × mark)
  grid + no-stress cell. Cross-entropy only on tokens that have stress
  supervision (`ignore_index` masks the rest — coverage is optional by
  design).
- Loss = `pos_loss + stress_weight × stress_loss` (weight 1.0).
- **Warm start**: encoder from the released `-vdu` tagger checkpoint;
  stress head tensors from stress_nn3 where shapes match; POS head
  fresh.
- 156.1M params total (vs ≈307M for the two-model stack) — one
  encoder is also what makes browser deployment plausible
  (see ../local/accentuator/onnx/BROWSER.md: 80% of the embedding rows
  are prunable for LT-only text).

Training recipe: AdamW (encoder 2e-5 / heads 1e-3, wd 0.01), warmup 500
+ cosine, bf16, batch 16 sentences ×2 epochs on 120k sentences; then a
POLISH pass (`--init-checkpoint --lr-scale 0.1`, constant or cosine —
measured equivalent) with per-epoch dev eval and best-checkpoint
selection (`.best.pt`). Per-epoch atomic checkpointing throughout
(crash insurance; a lost run costs ≤1 epoch).

## Non-obvious implementation facts (learned the hard way)

- Count stress marks on NFD: composed letters (à = U+00E0) hide their
  combining marks from NFC string counting.
- Move stress marks, never convert them: priegaidė (mark type) is
  lexical; notation normalization repositions marks only.
- When inserting a combining mark, defer past the cluster's existing
  combining characters (same combining class — NFC cannot reorder ė̃).
- Label-conditioned inference must select labels from the CLOSED
  vocabulary the model trained on (slot-match the tagger output against
  dictionary labels); free-text label formatting is out-of-distribution.
  Tie-break toward the FEWEST spurious slots — score_tags does not
  penalize slots only one side fills.
