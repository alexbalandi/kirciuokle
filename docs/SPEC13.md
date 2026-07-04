# Phase 13 — scorer steers toward VDU conventions (DET→PRON)

Rationale: the accent pipeline scores tagger output against VDU variant
morphology labels. Lithuanian traditional grammar — and therefore VDU's
label inventory — has no determiner category: demonstratives/possessives
are all `įv.` (PRON). Today a token the tagger calls DET gets a pos
MISMATCH (−3) against every `įv.` variant, so DET-tagged tokens are
disambiguated only by their other features. Mapping DET into the PRON pos
family fixes this for every tagger backend (LINDAT UDPipe included) and
stops penalizing models that follow the VDU convention.

## Change

In the scoring projection ("tokenTags"), UPOS `DET` now maps to pos
`"PRON"` (exactly like NOUN/PROPN and CCONJ/SCONJ merge today). Apply the
identical one-line change to EVERY port — they must stay faithful to each
other:

1. `src/worker/disambiguation.ts` `tokenTags` (+ a unit test: DET token
   scores pos-match against an `įv.` variant).
2. `local/app/kirciuokle/disambiguate.py` (+ mirror test in
   `local/app/tests/`).
3. `scripts/accent_text.py` `token_tags`.
4. `scripts/bench_taggers.py` slots projection.
5. `local/tagger-hf/metrics.py` slots projection.

Comment at each site: pos family follows VDU conventions (no DET in
Lithuanian traditional grammar), with a pointer to docs/SPEC13.md.

## Quality bar

- `npm run check` passes (worker tests updated/added).
- `uv run --project local/app pytest local/app/tests` passes.
- py_compile clean on the touched Python files;
  `uv run local/tagger-hf/selfcheck.py` still passes.
- NOTE in local/README.md's benchmark section: the slots metric now merges
  DET→PRON per VDU conventions; previously published numbers (UDPipe 89.0,
  Stanza 84.7, ours 87.4) predate the change and will be re-baselined by
  the orchestrator — do not edit the old table rows, just add the note.
- Do not modify `docs/`. No training, no deploys, no git.
