# SPEC37 — Package the joint model: ONNX + Hugging Face release

## Goal

Ship `joint/checkpoints/joint_v2_literary.best.pt` (the project's best
model: single-pass POS + accentuation) as a Hugging Face repo in the
existing family, with ONNX (fp32 + int8) inference artifacts and the
model card below. The user has explicitly authorized publishing.

New files: `local/accentuator/joint/export_joint_onnx.py`,
`local/accentuator/joint/package_hf.py`, and the release folder they
produce. Modify nothing else.

## 1. `export_joint_onnx.py`

- Load the checkpoint (rebuild via joint_lib), export ONNX with dynamic
  axes: inputs input_ids / attention_mask / char_ids /
  first_subword (+ whatever the forward genuinely needs — study
  joint_lib's forward signature); outputs pos_logits, stress_logits,
  no_stress_logits. Follow the working torch.onnx recipe in
  local/accentuator/onnx/export_stress_onnx.py (opset,
  quantize_dynamic int8 with the parity-stable scope trick if needed).
- Parity gate: 100 sentences from data/eval/lrt-smoke.txt through
  torch vs ONNX fp32 vs int8 — token-level argmax agreement (POS and
  stress separately): fp32 ≥99.5%, int8 ≥98%. Print the numbers.
- Emit joint.onnx, joint.int8.onnx, joint.meta.json (char vocab, label
  list, marks, max lens, no_stress flag) into the release folder.

## 2. `package_hf.py`

- Assemble a release folder `local/accentuator/joint/hf_release/`:
  - model.safetensors (the torch weights), joint_lib.py copied as
    modeling reference (plain file, not auto_map — the card explains
    usage), the ONNX pair + meta, tokenizer files from
    EMBEDDIA/litlat-bert (save_pretrained), labels.json, LICENSE
    (reuse the CC BY-SA 4.0 text block from
    local/tagger-hf/export_hf.py), and README.md = the model card
    below VERBATIM (fill the two {PLACEHOLDER} size values with
    measured MB).
  - `--upload` flag: create/update the HF repo
    `alexbalandi/litlat-bert-lithuanian-accentuator` via
    huggingface_hub (token from the repo root .env HF_TOKEN — load it
    like scripts elsewhere do with dotenv or manual parse; NEVER print
    it). Default is dry-run (assemble only); run `--upload` after the
    parity gate passes.

## 3. Model card (README.md — copy VERBATIM, fill only {SIZE_*})

---
language: lt
license: cc-by-sa-4.0
base_model: EMBEDDIA/litlat-bert
tags:
- lithuanian
- accentuation
- stress
- morphology
- token-classification
---

# litlat-bert-lithuanian-accentuator

Single-pass Lithuanian **accentuation + morphology** model: one encoder
(litlat-bert), one forward pass per sentence, two heads — per-token
morphological labels (804 traditional-grammar labels) and per-token
stress placement (choose the letter within the word + the accent mark:
grave/acute/circumflex, or "no stress" for unadapted foreign words).
156M parameters; ~1,660 tokens/s on a laptop RTX 3080 Ti, ~46 tokens/s
CPU int8.

Built by the open kirčiuoklė project:
https://github.com/alexbalandi/kirciuokle — architecture, training
recipes, evaluation methodology and the full experiment log are
documented there (docs/).

## Quality (measured, 2026-07)

| benchmark | metric | score |
|---|---|---|
| Kirčiuotų tekstų chrestomatija (hand-stressed gold, literary; never trained on) | token exact / position / sentence | **89.9% / 92.1% / 37.6%** |
| LRT news 37.7k tokens (audited silver reference) | token exact / position | **89.8% / 91.2%** |
| ALKSNIS gold test (morphology) | combined label / UPOS | **88.8% / 96.8%** |

Reference points on the same gold benchmark and extraction: the VDU
kirčiuoklė + UDPipe pipeline scores 95.8% token exact;
phonology_engine (LIEPA) 76.7%. Published thesis baselines on their own
extraction of the same book (indicative only): transformer 0.711,
VDU Kirčiuoklis 0.702 sequence accuracy.

## Training data & provenance

- Base training: 120k MATAS v3.0 sentences (CC BY 4.0, gold
  morphology) with stress PROJECTED from this project's own open
  accentuation dictionary (built from Wiktionary/kaikki CC BY-SA +
  published accentology rules + VLKK normative data; zero
  unadjudicated disagreements against its QA reference).
- Literary fine-tune: 220k tokens of public-domain Lithuanian classics
  (lt.wikisource; authors verified dead ≥70 years), labeled by a
  CALIBRATED consensus teacher (accept a token only where the
  agreement pattern of several independent systems measures ≥98%
  accurate on gold; achieved purity ≈99.5%). Transparency note: the
  teacher's voters include the VDU+UDPipe silver pipeline, so the
  literary fine-tune distills a consensus that contains VDU-derived
  signal; the base training does not.
- The gold benchmark was firewalled out of training (exact + 8-word
  shingle dedup; 159 sentences dropped).

## Files

- `model.safetensors` — torch weights ({SIZE_TORCH} MB). Rebuild the
  model with `joint_lib.py` (in this repo) + EMBEDDIA/litlat-bert
  tokenizer.
- `joint.onnx` / `joint.int8.onnx` — ONNX export ({SIZE_INT8} MB int8),
  inputs/outputs documented in `joint.meta.json`. Torch↔ONNX parity:
  fp32 ≥99.5%, int8 ≥98% token agreement.
- `labels.json` — the 804-label morphology inventory.

## Limitations

- Foreign/unadapted words should stay unaccented; the model does this
  for ~67% of such tokens (a known regression from 76% caused by the
  literary fine-tune — literary data contains no foreign words).
- Literary/poetic register trails the VDU reference by ~6pp (was ~9pp
  before the fine-tune); archaic orthography is out of scope.
- Sentences are processed independently (max 128 subwords per
  sentence; longer sentences are truncated by the reference serving
  code).
- The stress head only proposes linguistically valid (letter, mark)
  cells (audited validity mask); it cannot express marks the standard
  language forbids.

## License & attribution

CC BY-SA 4.0. Base model: EMBEDDIA/litlat-bert. Morphology training
data: MATAS v3.0 (CC BY 4.0, CLARIN-LT). Accent supervision:
this project's open dictionary (kaikki.org Wiktionary extraction
CC BY-SA; VLKK normative resources; published accentology per the
project docs) and public-domain literary texts.

---

## Pass criteria

1. Export runs; parity numbers printed and meet the gates.
2. Release folder assembled; README sizes filled; dry-run listing
   printed (every file + size).
3. `--upload` executes; paste the resulting model URL. HF_TOKEN comes
   from .env; never echo it.
4. GPU may be in use — export on CPU is fine.

Do not commit to git (the release folder is data; gitignore it if git
notices it).
