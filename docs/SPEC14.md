# Phase 14 — attributions folder, HF release packaging, official benchmark

Three deliverables. Numbers for the final clean model get filled in by the
orchestrator later — build everything so they slot in.

## 1. `ATTRIBUTIONS/README.md` (new top-level folder)

The single fair-play page for everything the project builds on. For each
entry: what it is, license, exactly how this project uses it, the
attribution line to reproduce, and a citation (BibTeX where one is
standard). Entries:

- **VDU kirčiuoklė** (kalbu.vdu.lt, VDU CCL) — accentuation dictionary
  service consumed via its public web API; the D1/SQLite dictionaries are
  caches of its responses. Not redistributed with the repo.
- **kirtis.info** — the original inspiration; same underlying VDU data.
- **MATAS v3.0** — CC BY 4.0. Rimkutė, Bielinskienė, Boizou,
  Dadurkevičius, Kovalevskaitė, Utka; CLARIN-LT hdl:20.500.11821/61.
  Used as tagger training data. Document our modifications: UD FEATS
  reconstruction from the Jablonskis XPOS, constrained-decoding self-fills
  of Number/Person, DET→PRON and AUX→VERB label normalization.
- **UD_Lithuanian-ALKSNIS** — CC BY-SA 4.0 (VDU; via Universal
  Dependencies). Training + gold dev/test.
- **EMBEDDIA/litlat-bert** — CC BY-SA 4.0 (Ulčar & Robnik-Šikonja,
  EMBEDDIA project). Base encoder of the released tagger → the fine-tuned
  weights are shared alike as CC BY-SA 4.0.
- **VSSA-SDSA/LT-MLKM-modernBERT** — Apache-2.0. Evaluated as an encoder
  candidate (not in the released model).
- **UDPipe 2 / LINDAT-CLARIN** (Straka, ÚFAL) — code MPL-2.0, models
  CC BY-NC-SA. Role: production tagger via the public REST service, the
  quality bar for benchmarks, and the teacher for one internal model
  variant ("teacher2") that is NOT part of the released clean lineage.
  Cite Straka (2018) UDPipe 2 paper.
- **Stanza** (Apache-2.0, Stanford NLP) — benchmarked.
- **LIEPA `phonology_engine`** — evaluated offline accentuation candidate.
- **hermitdave/FrequencyWords** (MIT, from OpenSubtitles data) — seeding
  wordlist for the dictionary warmer.
- **Lithuanian Wikipedia** (CC BY-SA) — evaluation corpus text.
- **CoNLL 2018 shared task evaluation script** — official UD metric tool,
  downloaded at runtime (not vendored).

End with a scope note: repository code is public domain (Unlicense); data
and models above keep their own licenses; the released tagger weights are
CC BY-SA 4.0 (inherited from litlat-bert + ALKSNIS).

Root `README.md`: add a short "Attributions" section linking the folder
(place it next to the existing License section; trim any now-duplicated
credit text from Credits if it makes sense).

## 2. `local/tagger-hf/eval_conll18.py` — the official benchmark

"Popular bench in proper format" for this model class = the CoNLL 2018 UD
shared-task evaluation on the UD test set.

- Downloads (and caches to `data/raw/`) the official
  `conll18_ud_eval.py` from
  `https://universaldependencies.org/conll18/conll18_ud_eval.py` and
  imports/executes it programmatically.
- Takes `--system conllu file` OR `--tagger-url` (UDPipe REST protocol —
  our sidecar or LINDAT): when a URL is given, it feeds the gold
  ALKSNIS-test sentence texts (`# text =`) through the tagger to produce
  the system conllu (document that tokenization differences then count
  against Words/UPOS/UFeats, exactly as in the shared task).
- Reports the official table (Tokens, Words, UPOS, UFeats, Lemmas F1) as
  printed by the official script — unmodified gold, no convention
  projection, so numbers are comparable with published UD results.
- Additionally prints a clearly-labeled second table "VDU-convention
  projection (this project's metric)": gold and system both projected
  (DET→PRON, AUX→VERB, FEATS restricted to the scoring slots) before
  scoring UPOS/UFeats with the same official machinery. Two tables, no
  mixing.
- Writes both to `runs/conll18-<name>.md` given `--name`.

## 3. `local/tagger-hf/export_hf.py` + model card template

- `export_hf.py --run-dir runs/<name> --bench-json <path> --out hf_release/<name>`:
  assembles a HF-ready folder: torch weights + tokenizer from `best/`,
  `onnx/` subfolder copied from an `--onnx-dir` if given, `LICENSE`
  (CC BY-SA 4.0 text), and `README.md` rendered from
  `model_card_template.md` with placeholders filled from `final.json`,
  the conll18 markdown, and a small `--facts-json` (free-form key/values:
  bench slots, speed, comparison rows).
- `model_card_template.md` with proper HF YAML front matter:
  `language: lt`, `license: cc-by-sa-4.0`, `base_model: EMBEDDIA/litlat-bert`,
  `pipeline_tag: token-classification`, `datasets` listing MATAS + ALKSNIS,
  `tags` (lithuanian, morphology, pos, upos, feats, accentuation), widget
  example sentences. Card sections: summary; intended use (Lithuanian
  accentuation pipelines + general LT morph tagging); **convention
  deviations from strict UD** (DET→PRON, AUX→VERB, slots-only FEATS,
  form-as-lemma except yra→būti — each with one-line rationale);
  training data & lineage (gold corpora + deterministic XPOS repair +
  constrained self-fill; explicitly UDPipe-free); benchmarks (official
  CoNLL-18 table AND VDU-convention table AND speed); how to use
  (transformers snippet, ONNX Runtime snippet, UDPipe-protocol sidecar
  docker); limitations; attributions/citations mirroring ATTRIBUTIONS.
- Optional `--push repo_id` using huggingface_hub (requires HF_TOKEN env);
  implement but DO NOT run.

## Quality bar

- py_compile clean; `--help` everywhere; selfcheck untouched/passing.
- Do not run training, eval, export, or any upload. Do not modify docs/ or
  scripts/. No git.
- `npm run check` unaffected.
