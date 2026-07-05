# SPEC23 — Closed-class extras loader (interjections/particles)

## Context

`local/accentuator/closed_class_extra.json` (already written, tracked)
holds 50 curated high-frequency interjections/particles/adverbs with
per-word source citations (e-LKŽ, en.wiktionary, VLKK R-13). Wire it into
generation.

Modify ONLY `local/accentuator/generate_dictionary.py`.

## Implementation

New module `generate_closed_extra(grouped) -> int`, called right after
`generate_closed` in `generate_dictionary` (summary key `closed_extra`
after `closed_rows`):

- Read `Path(__file__).parent / "closed_class_extra.json"` (note: NOT the
  data dir — this file is tracked source). Missing file → return 0.
- For each word entry and each form in `entry["forms"]`:
  `add_variant(grouped, form=form, pos=entry["pos"], tags=(),
  provenance=f"open-accentuator:closed-extra:{word}")`.
- Count = number of words that emitted at least one variant.
- Do NOT skip words already in `grouped` — these are curated normative
  variants and merging with other modules' output is intended.

## Pass criteria (in order)

1. `uv run local/accentuator/generate_dictionary.py` — summary shows
   `closed_extra: 50`.
2. Spot query: `uv run python -c "import sqlite3,json; db=sqlite3.connect('local/accentuator/data/generated.sqlite'); [print(r) for r in db.execute(\"SELECT word, default_form FROM words WHERE word IN ('galbūt','dėkui','štai','tegul','oho')\")]"`
   → all five present with accented default forms.
3. `uv run local/accentuator/parity_report.py` — `covered` strictly
   greater than 7,501. If `disjoint` is nonzero: DO NOT edit
   parity_vetoes.json, DO NOT change the JSON data file — paste the full
   DISJOINT samples from reports/parity-vdu.md into your final message for
   human adjudication (disagreements with the VDU cache are expected for a
   few clitics and will be resolved against the citations by the human).
4. `uv run local/accentuator/selfcheck.py` passes.
5. Report before/after parity numbers.

Do not commit.
