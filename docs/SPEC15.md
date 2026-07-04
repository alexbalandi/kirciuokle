# Phase 15 — UD-strict variant via head surgery + ALKSNIS fine-tune

Approved plan: the released tagger ships in two flavors.
- `-vdu`: the normalized-label model (DET→PRON, AUX→VERB) — accentuation-
  optimized, trained on everything.
- `-ud`: strict-UD variant for general users — produced by extending the
  `-vdu` checkpoint's classifier head with the folded-away labels and
  fine-tuning briefly on ALKSNIS (whose DET/AUX annotations are
  self-consistent; UD Lithuanian's DET/PRON split is word-list-based, so a
  short fine-tune from a strong warm start suffices).

## `local/tagger-hf/ud_variant.py`

`ud_variant.py --base-run runs/<name> --labels data/<ud-dataset>/labels.json
--out <dir>`:

1. Load the base checkpoint (`runs/<name>/best`) and the target UD label
   list (which includes `DET|...` / `AUX|...` labels).
2. Build a new classifier weight matrix (and bias) sized to the target
   label set:
   - label present in the base set → copy its row;
   - else compute its **folded twin** (apply DET→PRON, AUX→VERB to the
     UPOS part, keep FEATS canonicalized) and copy the twin's row when the
     twin exists in the base set — this starts training from "DET behaves
     like its PRON twin";
   - else initialize like a fresh head row (normal, std from config
     initializer_range) and report it.
3. Save a fully loadable HF checkpoint (config id2label/label2id updated,
   tokenizer copied) plus `head_config.json` updated for the new labels.
4. Print a summary: copied / twinned / fresh row counts.

The subsequent fine-tune is NOT part of this script — document the exact
`train.py` invocation in the module docstring
(`--model-name <surgery-out> --data-dir <ud-dataset> --learning-rate 1e-5
--epochs 6 --fallback-model ""`).

## Quality bar

- py_compile clean, `--help` works, selfcheck untouched.
- Do NOT run the surgery, prep, or training (orchestrator does). No
  changes outside `local/tagger-hf/`. No git.
