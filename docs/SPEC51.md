# SPEC51 — Pixel-locked panel alignment (input box === result box)

## Absolute requirement (user)

The input textarea (`#source-text`) and the result box (`#result-output`)
must be IDENTICAL in width AND height at all times, with identical top
and bottom edges — never drifting. The two panels must be equal width.
This is a hard invariant enforced by an automated geometry test.

Files: `index.html`, `src/client/style.css`, `src/client/main.ts`
(only if a resize hook is needed), tests. `npm run check`/`build` green.

## Layout approach: shared grid rows (CSS subgrid)

Restructure so the two boxes occupy the SAME grid row, making them
equal-height by grid semantics (no JS height math, no flex drift):

- `.workspace`: `display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); grid-template-rows: auto 1fr auto; gap:18px;` — three rows: header / BOX / below-box.
- Each panel (`.input-panel`, `.result-panel`): `display:grid; grid-row: 1 / span 3; grid-template-rows: subgrid;` so its children land on the parent rows. Keep the card border/padding on the panel; use an inner padding wrapper if the subgrid conflicts with padding (a panel-inner grid), whichever renders cleanly.
- Row 1 = header (`.panel-head` / `.result-toolbar`). Row 2 = the box
  (textarea / result-output) — SAME row → equal height. Row 3 = a
  single "below-box" wrapper per panel holding everything under the box
  (input: actions + mode-explainer + local-status + message; result:
  legend + primer). Wrap those into one `.panel-below` div each so each
  panel has exactly 3 direct grid children.
- The `tagger-notice` (currently between header and result box) must NOT
  push the result box out of row 2: move it to render as an overlay or
  place it inside row 3 / as an absolutely-positioned banner, so the box
  stays in row 2 aligned with the textarea. It is normally hidden.

Box styling parity (both textarea and result-output): identical
`padding` (14px both — result-output is currently 15px, fix), identical
border, border-radius, box-sizing:border-box, `min-height:240px`,
`max-height:62vh`, `width:100%`, `overflow:auto`. Neither uses `flex`
growth anymore (grid row governs height).

Auto-grow: if the textarea should still grow with pasted text, drive the
ROW, not the element — let row 2 be `1fr` (fills) OR set both boxes to
`height:100%` within their row and let the row size to `max-content` of
the taller box capped at 62vh; the point is they share the row so both
grow together. Verify long-text growth keeps them identical.

Below-820px stacked layout (existing @media): boxes stack; the
equal-size invariant applies per the single column (each full width).

## The geometry gate (the real deliverable — Playwright)

`test/harnesses/spec51-align.cjs`: load the site, and for the matrix
{viewport 1280px, 1024px, 900px} × {empty, long pasted text (the SPEC49
sample paragraph ×3)} × {web mode}, assert with getBoundingClientRect:

- `#source-text` and `#result-output`: |left diff| ≤ 0.5px,
  |width diff| ≤ 0.5px, |top diff| ≤ 0.5px, |bottom diff| ≤ 0.5px.
- `.input-panel` and `.result-panel`: |width diff| ≤ 0.5px.
- Print each case's measured rects; FAIL loudly on any breach.

Add a lightweight vitest is fine too, but the Playwright pixel gate is
mandatory and must pass.

## Pass criteria

1. `npm run check` + `build` green.
2. `node test/harnesses/spec51-align.cjs <vite-url>` passes every matrix
   case; paste the measured left/width/top/bottom for the 1280px
   long-text case (must show the two boxes pixel-identical).
3. Stacked layout (<820px) still usable; no horizontal scroll on body.

Do not commit, do not deploy.
