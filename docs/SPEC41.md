# SPEC41 — Pilot UI parity with the production site

## Goal

Make `bundled_weights_pilot/` look and speak like the production
Cloudflare app: same POS label style, similar legend, LT/EN/RU
localization, same visual tone. Modify only files under
`bundled_weights_pilot/` plus ONE asset-generation addition to its
`prepare_model.py`.

## 1. POS labels in production style (the label bridge, in JS)

The model outputs combined `UPOS|FEATS` labels; the site must display
VDU traditional-grammar `mi` strings ("dkt., mot. g., vns. kilm.") so
the production morphology glosses work. Do NOT write a free-text
formatter — reuse the closed-vocabulary bridge:

- prepare_model.py additionally dumps `model/label_bridge.json`:
  (a) the distinct dictionary `mi` label vocabulary with pre-parsed
  slot dicts — generate with the SAME code eval_nodict_pipeline.py
  uses (import its label-vocabulary builder / kirciuokle.disambiguate
  parse_mi); (b) for each of the 804 model labels, its slot dict
  (UPOS+FEATS → slots exactly like kirciuokle.disambiguate.token_tags:
  DET→PRON, AUX→VERB, VerbForm=Part → PART_VERB pos, Degree=Pos
  dropped).
- JS: for each model label above the 0.1 probability cut, look up its
  slots, score against every mi label's slots (faithful port of
  score_tags: +4/-3 pos, +2/-2 per slot, skip one-sided), pick the max
  with the FEWEST-spurious-slots tie-break; merge duplicate mi strings
  by summing probabilities. Popover rows: mi string + summed
  percentage, sorted desc. Cache per model-label (804 entries — the
  bridge result is input-independent, compute once at load).

## 2. Legend + word classes like production

Mirror index.html/src/client/style.css semantics:
- solid underline ("chosen by context" color) = single mi label ≥0.9
  summed probability;
- amber "multiple variants" underline = ≥2 mi rows above 0.1;
- dotted "not accented / foreign" = the no-stress cell won.
Legend row labels reuse the production wording per language; include
the stress-mark primer link + modal — copy the primer strings VERBATIM
from src/client/i18n.ts (they are hand-authored; do not retype accents).

## 3. Localization

LT/EN/RU switcher like production. Copy from src/client/i18n.ts:
the UI strings the pilot needs (tagline, input label, buttons, legend,
primer) AND the morphology abbreviation glosses (the hand-authored
`mi`-abbreviation translations) — copy as a generated JS module with a
header comment naming the source file as the single source of truth.
Pilot-specific strings (model download status, batch progress,
tokens/s) get new entries in all three languages (write natural LT/RU —
short UI strings).

## 4. Visual tone

Borrow the production stylesheet's look for panels/underline classes/
popovers (copy the relevant rules; keep the pilot self-contained). The
page must still clearly label itself as the in-browser pilot (subtitle
line, EN example: "runs fully in your browser — nothing is sent to a
server", localized).

## Pass criteria

1. prepare_model.py regenerates assets incl. label_bridge.json; print
   its sizes and 5 sample bridge mappings (UD label → mi string) —
   paste them; they must read like real dictionary labels.
2. Browser smoke (Playwright ok): accentuate a sentence; popover shows
   mi-style labels with percentages; a word with ≥2 readings shows the
   amber class; a foreign name shows dotted+unmarked; switcher flips
   all UI text across lt/en/ru incl. the primer modal.
3. Screenshot saved to bundled_weights_pilot/docs-screenshot.png for
   human review.
4. README updated (bridge design, localization source-of-truth note).

Do not commit.
