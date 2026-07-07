# SPEC58 — Spellcheck UX: underlines on the left, accentuation-only on the right

Status: ready for implementation (codex builds the structure; Claude tunes the
overlay alignment + verifies in the browser)

## Convention (owner's words, authoritative)

1. **Until accentuation runs, nothing appears on the right** — the result pane
   shows only accentuated text. No live preview on the right anymore.
2. **Spelling underlines + click-to-fix appear first on the LEFT** (in the input
   area, live as the user types/pastes). After the user accentuates, the RIGHT
   *also* shows them (current behaviour) — best of both worlds.
3. **Fixing a word on the RIGHT (accentuated text) re-accentuates that sentence.
   Fixing on the LEFT does not auto-accentuate** — it just edits the text.

## Current state (what changes)

- Today `runPreviewSpellcheck()` renders a *preview* (tokenised text + spellcheck
  underlines, `preview: true`) into the **right** pane (`#result-output`) before
  accentuation. **This preview moves to the left and the right becomes
  accentuation-only.**
- `renderedParts` currently holds either preview parts or accentuated parts and
  renders into `#result-output`. After this change `renderedParts` is
  **accentuated-only**; the left overlay gets its own separate state.
- Reusable pieces to keep: `tokenizeForPreview()` (preview.ts), `suggestBatch()`
  (spellcheckClient.ts), `spellingContextForIndex()`, `isCorrectableSpelling()`,
  `replaceTextareaRanges()`, `openCorrectionPopover()` (generalise it), the
  sentence-scoped `reaccentuateEdits()`.

## Part 1 — Right pane: accentuation-only

- Delete the preview path into `#result-output`. `runPreviewSpellcheck` no longer
  writes `renderedParts`/renders the right; it drives the **left overlay** (Part 2).
- The right pane shows its empty placeholder until `submitText()` (accentuation)
  fills it. After accentuation it renders accented parts **and** runs
  `annotateUnknownWordsWithSpellcheck()` so the right keeps its underlines +
  click-to-fix exactly as today (rule 2, second half).
- Remove `isPreviewResult()` and the `preview` flag usages that gated the right
  pane (the right is never a preview now). `RenderedPartCore.preview` can be
  dropped, or repurposed for the left state — implementer's choice, but the right
  pane must never be a preview.
- `updateCopyButtonState()` / copy button: enabled when there are accented parts
  (drop the `isPreviewResult()` condition — it's moot).

## Part 2 — Left overlay (live underlines over the textarea)

A native `<textarea>` can't hold clickable underlines, so overlay a
character-aligned layer over it. **The two panes already share one text-layout
ruleset (`textarea, .result-output` in style.css) with `scrollbar-gutter: stable`;
the overlay is a third layer in that same lockstep.**

### DOM

Wrap the textarea in a positioned container (in `index.html`):

```html
<div class="textarea-wrap">
  <textarea id="source-text" ...></textarea>
  <div class="textarea-overlay" id="source-overlay" aria-hidden="true"></div>
</div>
```

### CSS (starting point — Claude will pixel-tune)

- `.textarea-wrap { position: relative; }`
- `.textarea-overlay`:
  - `position: absolute; inset: 0;` (covers the textarea; the textarea's 1px border
    box and the overlay's box must line up — inset by the border if needed so text
    starts at the same pixel).
  - **Inherit the shared text-layout ruleset** — add `.textarea-overlay` to the
    `textarea, .result-output` selector so padding (14px), font, `white-space:
    pre-wrap`, `overflow-wrap`, `tab-size`, and `scrollbar-gutter: stable` match
    EXACTLY. Wrapping must be identical to the textarea or the underlines drift.
  - `color: transparent;` — the overlay's text is invisible; only underline
    decorations show. The textarea underneath provides the visible text + caret.
  - `pointer-events: none;` — clicks pass through to the textarea for editing…
  - `overflow: hidden;` — it never shows its own scrollbar; it's scrolled
    programmatically (see scroll sync). Must still reserve the same gutter as the
    textarea so content width matches — verify in-browser; if `scrollbar-gutter`
    doesn't reserve under `overflow: hidden`, use `overflow-y: scroll` +
    `scrollbar-width: none` + `::-webkit-scrollbar { width: 0 }` instead.
  - Same `min-height`/`max-height`/border-radius as the textarea so the box matches.
- `.spell-underline` (the clickable misspelling markup inside the overlay):
  - `pointer-events: auto; cursor: pointer;`
  - a visible underline in the "taisytina" colour (reuse the legend-unknown /
    `token-correctable` colour), e.g. `text-decoration: underline;
    text-decoration-style: wavy; text-decoration-color: <unknown>;
    text-underline-offset: 2px;` — the underline shows even though `color` is
    transparent (decoration uses its own colour).

The textarea keeps its opaque background; the overlay is transparent so the
textarea text shows through, with the overlay underlines drawn on top, aligned.

### Left state + render

- New module-level `leftTokens: RenderedPartCore[]` (separate from `renderedParts`).
- `renderLeftOverlay()`: rebuild `#source-overlay` from `leftTokens`. Each token
  tiles the text exactly (`tokenizeForPreview` guarantees join === textarea.value):
  - a **word** token that is correctable (`isCorrectableSpelling(spelling)` — a
    `restore`, or a `typo` with candidates) → a `<span class="spell-underline"
    data-index=…>` (clickable).
  - every other token (ok words, separators) → a plain text node (transparent,
    preserves layout/wrapping).
  - The joined text content MUST equal `textarea.value` exactly (assert/guard).

### Live trigger

Keep the existing debounce (`PREVIEW_SPELLCHECK_DEBOUNCE_MS`, ~600ms) + immediate
on paste. `runLeftSpellcheck()` (rename of `runPreviewSpellcheck`):
1. Bump the staleness id.
2. If `textarea.value.trim()` empty → clear `leftTokens`, clear overlay, return.
3. `leftTokens = tokenizeForPreview(textarea.value)`.
4. `suggestBatch` over the word tokens with neighbour context
   (`spellingContextForIndex` adapted to `leftTokens`), respecting the staleness id.
5. `renderLeftOverlay()`.

The overlay updates on every input; it's independent of the right pane. Editing the
textarea after accentuating updates the left overlay but leaves the right pane as
the last accentuation (right updates only on Accentuate).

### Scroll sync

On `textarea` scroll, mirror to the overlay AND the result output:
`overlay.scrollTop = textarea.scrollTop; overlay.scrollLeft = textarea.scrollLeft;`
and keep the existing textarea↔result 1:1 sync. (Overlay has no scrollbar of its
own.) Also re-sync overlay scroll after `renderLeftOverlay()` and on resize
(`syncBoxHeights`).

### Click → fix (LEFT, no accentuation)

Click on a `.spell-underline` opens the correction popover at that word (reuse
`openCorrectionPopover`, generalised to take the token list + a fix callback).
Picking a candidate calls **`applyLeftSpellingCorrection(index, candidate)`**:
- `replaceTextareaRanges([{ start, end, text: candidate }])` (uses the token's
  `sourceStart`/`sourceEnd`),
- close popover,
- `runLeftSpellcheck()` to refresh the overlay,
- **do NOT accentuate.**

## Part 3 — Fix branching (right unchanged)

- RIGHT pane correctable tokens keep calling `applySpellingCorrection(index,
  candidate)` → `replaceTextareaRanges` → **`reaccentuateEdits`** (sentence-scoped
  re-accentuation) — rule 3, first half. Unchanged.
- LEFT overlay uses `applyLeftSpellingCorrection` (Part 2) — no accentuation.
- The "fix all" button (`#fix-all-button`) currently fixes right-pane single-
  candidate restores and re-accentuates. Keep it bound to the RIGHT (accentuated)
  result. (Left has no fix-all; per-word only. If trivial, a left fix-all is
  out of scope for this spec.)

## Part 4 — Verify

- `npm run check` green (tsc + Vitest). Existing preview.ts unit tests still pass;
  `tokenizeForPreview` round-trip unchanged.
- Add a small unit test if any pure helper is added (e.g. left-token context).
- Manual/browser (Claude will do this): left underlines appear live as you type,
  are **character-aligned** with the words in the textarea (the key check),
  scroll in sync, and clicking one fixes the word with no accentuation; the right
  pane stays empty until Accentuate, then shows accented text with its own
  underlines; fixing on the right re-accentuates the sentence.

## Constraints

- Preserve exact character alignment — this is the owner's hard rule. The overlay
  shares the text-layout ruleset; do not give it any divergent padding/font/wrap.
- No new dependencies. Keep the textarea copyable and editable as now.
- The overlay is `aria-hidden` (decorative); keep the textarea's own
  accessibility. Clicking underlines is a mouse convenience.
- Don't touch the worker, model, or accentuation engine.
