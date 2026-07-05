# Phase 16 — fully UD-compliant v2 (full FEATS + lemmatizer)

The released `-ud` variant restored UD UPOS categories but still emits
slots-only FEATS and form-as-lemma, so the official CoNLL-18 UFeats/Lemmas
columns are low by design. v2 closes both gaps.

## Part A — full UD FEATS

1. `prep_corpus.py` gains `--feats-keys all` support that actually works
   for MATAS by decoding the gold annotations we have not used yet:
   - **Multext-East codes** in the MISC column (`Multext=Ncfsgn-` style):
     implement a positional decoder for the Lithuanian MULTEXT-East v4
     tagset (per-POS position tables; noun/adjective/pronoun/numeral/verb
     positions encode Type, Definiteness, Degree, PronType-like Type,
     etc.). Map to UD keys: Definite, PronType, NumType, NumForm,
     Polarity, Poss, plus anything the existing slots repair misses.
     Decode defensively: unknown positions are skipped, decoder must be
     covered by unit checks in `selfcheck.py` with a handful of hand-
     verified examples from the corpus.
   - Extend the Jablonskis XPOS repair table where it trivially covers UD
     keys (`įvardž.`→Definite=Def, `neįvardž.`→Definite=Ind,
     `nelygin.`→Degree=Pos, `teig.`/`neig.`→Polarity if present).
   - Merge precedence stays: explicit gold UD FEATS > XPOS repair >
     Multext decode. Post-prep coverage report must show the previously
     missing keys (Definite on ADJ, PronType on PRON, Polarity on VERB)
     within 10pp of ALKSNIS dev.
2. Full-feats label space will be larger (expect 2-4k labels); print size.
   Keep `--normalize-vdu` OFF for this dataset (UD conventions,
   `--no-normalize-vdu`), keep the leakage guard and self-fill inputs
   unchanged (`--matas-file` still points at the gen-2 filled corpus).

## Part B — lemma head (edit scripts)

1. `prep_corpus.py --emit-lemma-scripts`: for each token compute a
   **casefold-aware edit script** from FORM to LEMMA (standard shortest
   common-affix scheme: strip N chars from the end, append suffix S;
   plus a lowercase-first-letter flag; fall back to a whole-word
   substitution class for irregulars). Store per-token script labels in
   the jsonl (`lemma_scripts` array) and the script inventory in
   `labels.json` metadata or a sibling `lemma_scripts.json`. Print
   inventory size and the coverage of the top-N scripts.
2. `train.py --lemma-head` (off by default): adds a second linear head
   over the script inventory on the same encoder (reuse the multi-head
   wrapper machinery from head_modeling; loss = mean of label CE and
   script CE; metrics gain `lemma_accuracy` computed by applying predicted
   scripts to dev forms and comparing to gold lemmas).
3. `export_onnx.py` and `server.py`: config-driven via head_config.json —
   when a lemma head is present, export both outputs and have the server
   decode the script to produce the LEMMA column (keep the yra→būti shim
   as an override only when the decoded lemma disagrees on form "yra").
4. `ud_variant.py`: twin-matching must also try a **slots-stripped twin**
   (drop keys outside the slots set, then fold DET/AUX) so full-feats
   labels initialize from the slots base instead of falling to fresh rows;
   report copied/twinned/stripped-twinned/fresh counts.

## Quality bar

- py_compile + `--help` everywhere; selfcheck extended (Multext decode
  samples, edit-script round-trip on examples incl. an irregular and a
  capitalized proper noun) and passing.
- Do NOT run prep/training/export (orchestrator does). No changes outside
  `local/tagger-hf/`. No git.
