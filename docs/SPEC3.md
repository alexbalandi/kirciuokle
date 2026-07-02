# Phase 3 — LT/EN/RU interface localization

The audience is a mixed EN/RU-speaking group: the *words* stay Lithuanian,
but all interface chrome and — most importantly — the grammatical morphology
labels in variant popovers must be viewable in English or Russian
(e.g. „kilm." → "genitive" / «родительный»).

## Already done — do not rewrite

`src/client/i18n.ts` exists and is authoritative. It exports:

- `type Lang = "lt" | "en" | "ru"`, `LANGS`
- `UI: Record<Lang, UiStrings>` — every user-facing string
- `translateMorphology(info: string, lang: Lang): string` — token-wise
  translation of VDU mi strings ("bdv., vyr. g., vns. vard." →
  "adjective, masculine, singular nominative"), keeping unknown fragments
  and dictionary meanings (after " - ") verbatim
- `detectLang(): Lang` — localStorage `lang` key, else navigator.language
  (lt→lt, ru→ru, else en)

The translation tables were hand-reviewed — do NOT edit their content.
Wire everything through this module.

## What to build

1. **Language switcher** in the header: three small buttons `LT | EN | RU`
   (current one highlighted). Clicking: saves to localStorage, sets
   `document.documentElement.lang`, re-renders all UI strings, the legend,
   any open state (result stays, popover may close), and morphology text.
   Initial language: `detectLang()`.
2. **index.html**: static Lithuanian strings move to being rendered/updated
   from `UI[lang]` (keep `Kirčiuoklė` brand name and footer link targets
   as-is in all languages; footer becomes
   `{footerData}: <a>VDU kirčiuoklė</a> (kalbu.vdu.lt) · {footerInspired} <a>kirtis.info</a>`).
3. **Popovers**: variant `info` rendered through
   `translateMorphology(info, lang)`; when lang is "lt" it still goes
   through the function (it expands abbreviations to full Lithuanian terms —
   that is desired).
4. **Error messages**: map API failures to `UI[lang]` keys by HTTP status
   (400→errEmpty, 413→errTooLong, 502→errUpstream, else errUnexpected)
   instead of showing server-sent Lithuanian text.
5. **Char counter** stays numeric. `title` attr on unknown tokens and the
   tagger notice use `UI[lang]`.

## Quality bar

- Unit tests for `translateMorphology`: the "bdv., vyr. g., vns. šauksm.;
  bdv., vyr. g., vns. vard." two-reading case in en and ru; a string with an
  unknown fragment (kept verbatim); a "mi - meaning" case (meaning verbatim);
  longest-match ("būt. k. l." must not be eaten by "būt. l.").
- Unit test for detectLang precedence (mock localStorage/navigator).
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/`, `docs/`, or the data tables in `i18n.ts`
  (wiring-level changes to its exports are allowed only if strictly needed —
  prefer none).
