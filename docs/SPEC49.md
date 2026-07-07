# SPEC49 — Word-popover redesign (design prescription, apply exactly)

The current popover reads as uneven and off-balance: per-segment
stacked translations create a ragged two-story ribbon, chosen vs
other readings use inconsistent container styles, the popover
left-anchors to the word, and the percent pills are visually heavy.
Apply the following prescription in `src/client/` (style.css +
popover-building code in main.ts). Web and Local modes share ALL of
it. Keep tests/check/build green; update Playwright assertions that
reference popover structure.

## Geometry

- Popover horizontally CENTERED on the clicked word, with a 10px
  caret (CSS triangle) pointing at the word; flip above/below by
  available space as now; clamp to viewport with 8px margins (when
  clamped, the caret stays pointing at the word, not the popover
  center).
- Fixed width 320px (min 280 on tiny screens); max-height 340px with
  overflow-y auto (relevant for web-mode "all").
- One elevation: border-radius 10px, 0.5px hairline border, single
  soft shadow. No nested boxes inside.

## Structure — one flat row per reading (no group sub-cards)

Each reading is ONE row block:
1. Line 1: accented form (lang="lt", 15px, weight 600, left) ·
   right-aligned quiet percentage (13px, muted color, NO pill
   background; "67%" / "8.4%" per the formatting rules) — web mode
   shows a checkmark glyph instead of a percentage on the chosen
   reading, nothing on others.
2. Line 2: morphology, ONE line, muted 13px: Lithuanian abbreviations
   joined with a middot separator: "dkt. · mot. g. · vns. · kilm.".
3. Line 3 (ONLY when UI language is not LT): the translated glosses
   as ONE parallel line, same order, same middot separators, 12px,
   more muted: "существительное · жен. род · ед. ч. · родительный".
   No per-segment stacking, ever. Long lines wrap as whole lines.
- Rows separated by hairline dividers; identical vertical padding
  (10px top/bottom) for every row including the chosen one.
- Chosen/selected state (web user-choice AND the top local reading):
  3px accent left bar + very light tint across the FULL row — the
  same treatment everywhere, no unique boxes.
- Whole row is the click target (choose reading); hover = subtle
  surface tint; cursor pointer only when choosing is possible.
- If a reading would render an empty morphology line, omit line 2/3
  rather than leaving blank space.

## Details

- Numeral fragments (SPEC47) keep their single localized line, same
  row anatomy.
- Loading/error states inside the popover use the same row padding
  (no layout jump).
- Remove any group headers that duplicate the accented form; the
  headword line IS the header.

## Pass criteria

1. check + build green.
2. Playwright on the user's sample paragraph (the Gintautė text in
   docs/SPEC49-sample.txt — create it from the spec commit): open
   popovers on an ambiguous word, a plain word, and (in Local mode
   with the dev bundle) a probability word — screenshot each to
   docs-popover-1/2/3.png; assert centering (popover center within
   4px of word center when unclamped) and single-line morphology.
3. Both languages: LT shows two-line rows (no translation line), RU
   shows three-line rows with parallel gloss line.

Do not commit, do not deploy.
