# Phase 12 — surgical teacher fill for MATAS gap features

Repaired-data run results (modernBERT combined/first, 3 epochs): dev
feats-exact 86.1% (was 59.8%), slots 82.0%, UPOS 90.1%. Remaining known
label gaps that XPOS cannot repair: Number on (mostly 3rd-person)
VERB/AUX (Jablonskis leaves morphologically ambiguous number unmarked;
ALKSNIS annotates it from context), and Person/Reflex on PRON. Fix by
asking the production teacher (UDPipe 2) to fill ONLY those keys where
gold is silent.

## `local/tagger-hf/teacher_fill.py`

1. Input: `data/raw/MATAS3.conllu` (or `--input`); output
   `data/raw/MATAS3.teacher.conllu` (or `--output`).
2. Select sentences containing at least one gap token, where gap token =
   - UPOS in {VERB, AUX} and FEATS lacks `Number`, or
   - UPOS == PRON and FEATS lacks `Person` or `Reflex`.
   Sentences without gap tokens are passed through unchanged.
3. For selected sentences, query the UDPipe REST service with
   **pre-tokenized CoNLL-U input** so alignment is exact:
   `POST {UDPIPE_URL}/process` with fields `input=conllu`, `tagger=`,
   `model=lithuanian-alksnis`, `data=<conllu chunk>` where the chunk is a
   skeleton built from the gold tokenization (columns: ID, FORM, rest `_`).
   Default `UDPIPE_URL=https://lindat.mff.cuni.cz/services/udpipe/api`,
   overridable by env/flag (so a local sidecar can be used instead).
   Batch ~80 sentences per request; polite pacing flag `--rps` (default
   1.0 requests/sec); retry a failed batch once, then leave those
   sentences unfilled and count them.
4. Merge rule, per gap token: copy ONLY the missing keys among
   {Number for VERB/AUX; Person, Reflex for PRON} from the teacher token's
   FEATS into the gold FEATS (canonical sorted order). Never overwrite an
   existing gold key. Never touch UPOS/XPOS/LEMMA or other tokens.
5. Progress lines every ~5k sentences; resumable: if the output file
   exists, `--resume` skips already-written sentences (append mode by
   sentence count) — or simpler, write to a temp file and only rename at
   the end plus document that reruns restart (choose one; document it).
6. Final report: sentences processed/filled/failed; per-key fill counts.

## prep integration

`prep_corpus.py --matas-file <path>` (default the raw file) so the
teacher-filled file is a drop-in. No other prep changes.

## Quality bar

- py_compile clean; `--help` works; a `--limit N` flag for a small trial
  run (the orchestrator will trial with N≈200 sentences first).
- Do NOT run the full fill or any training. No changes outside
  `local/tagger-hf/`. No git.
