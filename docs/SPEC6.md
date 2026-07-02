# Phase 6 — ruby-annotated grammar terms (learn the Lithuanian terminology)

The users are EN/RU speakers learning Lithuanian. When the interface is in
EN or RU, the morphology labels in variant popovers must still expose the
original Lithuanian terms so the terminology is absorbed naturally — without
cluttering the popover.

## Design: furigana-style ruby annotations

In EN/RU mode each translated grammar term is rendered as an HTML ruby pair:
the translation is the base text, the full Lithuanian term sits above it in
small muted type:

```
 kilmininkas          moteriškoji giminė
 родительный  ,       женский род        , ...
```

i.e. `<ruby>родительный<rt>kilmininkas</rt></ruby>`. Untranslatable
fragments and dictionary meanings render as plain text without ruby. In LT
mode behavior is unchanged (expanded Lithuanian terms, no ruby).

## i18n.ts changes (data tables must not be touched)

Add a structured variant of the existing translation next to
`translateMorphology` (keep that function and its tests working — it is the
plain-text form):

```ts
export type MorphSegment = { text: string; lt?: string };
export function morphologySegments(info: string, lang: Lang): MorphSegment[];
```

- Reuse the existing greedy longest-match logic (factor the shared walk out
  rather than duplicating it).
- For a matched abbreviation in EN/RU: `{ text: GLOSS[lang], lt: GLOSS.lt }`
  (`lt` = the full Lithuanian term, e.g. "kilmininkas" — not the "kilm."
  abbreviation).
- For lang "lt", or unmatched fragments, separators (", ", "; "), and
  meanings (after " - "): segments without `lt`.
- Punctuation/separators may be merged into neighboring plain segments —
  whatever is simplest — but token order must be preserved and
  `segments.map(s => s.text).join("")` must equal
  `translateMorphology(info, lang)`.

## Rendering (main.ts / style.css)

- The variant-option info line builds DOM from `morphologySegments`:
  segments with `lt` become `<ruby>{text}<rt>{lt}</rt></ruby>`, others are
  text nodes.
- Style: `rt` ≈ 0.6em, `--text-muted`, no wrap inside a ruby pair;
  increase the option line-height enough that annotations don't collide
  with the row above.
- Language switching re-renders an open popover correctly (or closes it —
  current behavior; do not regress it).

## Quality bar

- Unit tests for `morphologySegments`: EN and RU two-reading case
  ("bdv., vyr. g., vns. šauksm.; bdv., vyr. g., vns. vard.") checking both
  the join-equality invariant and that matched segments carry the correct
  full-LT `lt`; a string with an unknown fragment (no `lt`); an
  " - meaning" case (meaning has no `lt`); lang "lt" produces no `lt`
  fields.
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/`, `docs/`, or the gloss data tables in `i18n.ts`.
  No deploy, no dev server, no git.
