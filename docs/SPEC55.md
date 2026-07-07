# SPEC55 — Lightweight LT spellcheck + correction (no NN)

Users paste Lithuanian that drops the LT diacritics (`zmogus` for `žmogus`,
`aciu` for `ačiū`) or has small typos. Offer dictionary-based correction using
a compact client-side wordlist — no model, no network, works in web AND local
mode, keeps the "nothing leaves your device" promise. Fix only where sure;
leave genuinely foreign/unknown words alone.

The wordlist + generator already exist (do NOT rebuild them):
- `scripts/build_spellcheck_wordlist.py` → `public/spellcheck-lt.txt` (one
  un-stressed LT surface form per line, 130k forms, ~316 KB over the wire).
  Commit the generated `public/spellcheck-lt.txt`.
- 91% of ASCII-folds are unambiguous (single confident candidate).

`npm run check` + `build` green. No deploy. No commit (the human commits).

## Asset serving

- Vite serves `public/` → the worker's ASSETS serves `/spellcheck-lt.txt` in all
  envs. Verify a built `dist/client/spellcheck-lt.txt` exists and is fetchable.
- Client fetches it **lazily** (first accent run, or first time a correctable
  word would render) with `fetch("/spellcheck-lt.txt")` → `.text()` →
  `split("\n")`. If Cloudflare doesn't transfer-compress it, fall back to
  shipping `public/spellcheck-lt.txt.gz` + `DecompressionStream("gzip")`; pick
  whichever actually serves compressed and document it.

## Engine — `src/client/spellcheck.ts` (new, framework-free, unit-tested)

- `foldAscii(w)`: ą→a č→c ę→e ė→e į→i š→s ų→u ū→u ž→z (+ uppercase), lowercase.
- Build once from the wordlist:
  - `valid: Set<string>` of `foldCase`-preserved lowercased forms (membership).
  - `foldIndex: Map<fold, string[]>` — ASCII fold → the diacritic-bearing forms.
  - `deleteIndex: Map<string, string[]>` — SymSpell delete-1 index over the
    lowercased forms, for edit-distance-1 typo lookup (build deletes of each
    form; at query time also delete-1 the query and intersect).
- `suggest(word): { status: "ok" | "restore" | "typo" | "unknown", candidates: string[] }`
  - already a valid form (case-insensitive) → `ok`.
  - else `foldIndex[fold(word)]` non-empty AND the word itself is pure-ASCII (no
    LT letters) → `restore` (diacritics were dropped); candidates ranked
    shortest-edit / fewest diacritics first; a single candidate = high
    confidence.
  - else edit-distance-1 matches via `deleteIndex` → `typo` (offered, never
    auto-applied).
  - else `unknown` (foreign/OOV — no fix).
  - Reapply the query's case (title/upper/lower) to each candidate.

## Integration (`src/client/main.ts`)

- After a result renders, for each **word** part that the accentuator left
  `unknown` (or a new "not a known form" check), call `suggest`. Attach the
  result to the rendered part.
- `restore`/`typo` words render with the **same dotted underline as unknown**
  (reuse `.token-unknown`; add a subtle affordance, e.g. `cursor: pointer` +
  `data-correctable`), and are clickable.

## UI

- **Click a correctable word** → popover (reuse the variant-popover plumbing,
  appears by the word): a localized heading ("Did you mean…"), the candidate
  spelling(s) as buttons. Choosing one **rewrites that word in the textarea**
  (preserve surrounding text/whitespace exactly) and re-runs accentuation.
- **Toolbar icons** in `.input-actions` (near the accent button): make the
  existing **"Stress all"** (accent-button) an **icon button** (keep its label
  as tooltip/aria-label), and add a **"Fix all"** icon next to it. "Fix all"
  applies every `restore` word that has a single confident candidate to the
  textarea in one pass, then re-accents. Disable "Fix all" when there are none.
  Keep the boxes/alignment work intact.
- Legend: the dotted "not in dictionary" entry now also covers "typo/needs
  diacritics" — keep one dotted style; optionally note it in the legend copy.

## Localization (LT/EN/RU)

Add strings: correction heading ("Did you mean"), fixAll label/aria, stressAll
aria (now icon), and any "no known correction" text. **LT accents:** verify with
the project's process (VLKK data `local/accentuator/data/vlkk_recommendations.json`,
the teacher/model, en.wiktionary) — do NOT trust a single VDU pass, and do NOT
invent marks. If unsure of a mark, leave that LT string unaccented and flag it.

## Tests

- Unit-test `spellcheck.ts`: `foldAscii`; `suggest("zmogus")`→restore with a
  `ž…` candidate; a valid word → `ok`; a 1-edit typo → `typo`; gibberish →
  `unknown`; case reapplied.
- `npm run check` + `build` green; report the test count.
- Playwright/mocked: type a diacritic-less sentence, accent, assert correctable
  words get the dotted underline, a click applies the fix + re-accents, and
  "Fix all" corrects the confident ones. Screenshot the popover + toolbar icons.

Do not commit, do not deploy.
