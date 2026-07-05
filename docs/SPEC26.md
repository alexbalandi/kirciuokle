# SPEC26 — Dictionary fixes from the LRT audit

Three opus audit passes over the LRT silver corpus surfaced dictionary
defects. The veto additions are already committed by the human
(parity_vetoes.json: indėlis, klimatas, pereiti lemmas; išgirdęs/išgirdę
words). Implement the two remaining CODE fixes.

Modify ONLY `local/accentuator/generate_dictionary.py`.

## 1. VLKK names: letter-page cross-check guard

vardai.vlkk.lt is internally inconsistent for 35 of 179 cross-checkable
names: the letter-page index form (e.g. Marcèlė, Nijõlė — confirmed
correct by the audit and the VDU cache in both verified cases) disagrees
in priegaidė with the detail page's paradigm cells (Marcẽlė, Nijòlė).
Data layout in `data/vlkk_names.json` (dict name→entry): letter-page-only
entries have `accented` but no `cells`; detail entries have `cells`
(sometimes under an ascii-folded key, e.g. "Marcele" while the letter
entry sits at "Marcelė").

In `generate_vlkk_names`, before emitting a detail entry's paradigm:
- compute the plain lowercase name from the entry's nominative cell
  (strip_accents + lower);
- if ANY cells-less entry in the JSON has an `accented` whose plain
  lowercase form equals it AND whose NFC/normalize_lt form differs from
  the nominative cell's, SKIP the whole name (no emission — disputed
  answers are dropped, never averaged) and count it;
- report the skipped count in the module's return path so the summary
  print can show it (extend the summary dict with `vlkk_names_disputed`
  if that is cleanest, or print a one-liner).
Build the letter-page index ONCE outside the loop.

## 2. BUTI_FORMS: nebėra / tebėra

The audit confirmed these high-frequency būti forms are missing (LKŽ:
nebė̃ra, tebė̃ra). Add to the static BUTI_FORMS table, same shape as the
existing rows (find the table, mirror an existing 3rd-person row like the
yra/nėra handling; label/tags consistent with the neighbors). Use
combining marks: nebė̃ra = n-e-b-ė+U+0303-r-a, tebė̃ra likewise.

## Pass criteria (in order)

1. `uv run local/accentuator/generate_dictionary.py` — summary shows the
   disputed-names count (expect ~35) and completes.
2. Spot query: nebėra and tebėra present with the tilde forms;
   marcelė/nijolę ABSENT (or carrying only non-vlkk variants);
   indėlių/klimato/pereis/išgirdęs absent.
3. `uv run local/accentuator/parity_report.py` — DISJOINT must stay 0; if
   nonzero paste samples, do not adjudicate. covered may drop slightly
   (vetoed words move to UNCOVERED) — report before/after.
4. `uv run local/accentuator/selfcheck.py` passes.

Do not commit.
