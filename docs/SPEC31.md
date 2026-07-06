# SPEC31 — Stress-mark primer (kirčio ženklai) on the site

## Goal

Not every visitor knows what the three Lithuanian stress marks mean. Add a
small, non-cluttering primer: a link in the legend row that opens a modal
explaining the marks, localized LT/EN/RU, with a link to VLKK as the
authoritative source. LOCAL verification only — do NOT deploy, do NOT run
wrangler.

Files to modify: `index.html`, `src/client/i18n.ts`, `src/client/main.ts`,
`src/client/style.css`. Nothing else.

## UI

- In the legend row (next to the legend items, after `legend-unknown`),
  add a small link/button `id="primer-link"` styled like a subtle text
  link with a leading "?" character or dotted underline — visually
  quieter than the legend entries.
- Clicking opens a modal (`role="dialog"`, `aria-modal="true"`,
  `aria-labelledby` the title id): centered card, max-width 560px,
  backdrop click and Escape both close, × button top-right, focus moves
  to the dialog on open and back to the link on close.
- Modal content structure: title; one intro paragraph; three mark
  entries (each: a large mark glyph in a fixed-width box, name, one-line
  description, examples); one line about mixed diphthongs; one
  minimal-pair line; a footer link to VLKK opening in a new tab
  (`rel="noopener"`).
- All example words must be wrapped in `<span lang="lt">` and rendered
  in the same font as the result output. The mark glyphs for the three
  boxes: `à` `á` `ã` (use the accented letter, larger font, not a bare
  combining mark).
- The modal must work with the existing language switcher: strings come
  from the i18n table and re-render on language change like the rest of
  the UI.

## Content (copy VERBATIM — accent marks are combining characters and
must survive exactly; do not retype, copy)

New `UiStrings` fields (add to the type and all three languages):
`primerLink`, `primerTitle`, `primerIntro`, `primerGraveName`,
`primerGraveDesc`, `primerGraveEx`, `primerAcuteName`, `primerAcuteDesc`,
`primerAcuteEx`, `primerTildeName`, `primerTildeDesc`, `primerTildeEx`,
`primerMixed`, `primerPair`, `primerMore`.

LT:
- primerLink: `Kas yra kirčio ženklai?`
- primerTitle: `Kirčio ženklai`
- primerIntro: `Kirčiuotas skiemuo tariamas ryškiausiai. Ilguosiuose skiemenyse dar skiriama priegaidė — balso „kryptis“ skiemens viduje. Žymimi trys ženklai:`
- primerGraveName: `Kairinis`
- primerGraveDesc: `trumpas kirčiuotas skiemuo.`
- primerGraveEx: `bùs, vìsas, làbas`
- primerAcuteName: `Dešininis`
- primerAcuteDesc: `ilgas skiemuo, tvirtapradė priegaidė — pabrėžiama skiemens pradžia.`
- primerAcuteEx: `výras, káina, áukštas`
- primerTildeName: `Riestinis`
- primerTildeDesc: `ilgas skiemuo, tvirtagalė priegaidė — pabrėžiama skiemens pabaiga.`
- primerTildeEx: `nãmas, laũkas, geraĩ`
- primerMixed: `Mišriuosiuose dvigarsiuose (il, ir, al, an…) riestinis rašomas ant l, m, n, r: šil̃tas, var̃das.`
- primerPair: `Priegaidė gali skirti žodžius: áušta (švinta) ir aũšta (vėsta).`
- primerMore: `Plačiau — VLKK: tartis ir kirčiavimas`
VLKK URL (all languages): `https://www.vlkk.lt/aktualiausios-temos/tartis-ir-kirciavimas`

EN:
- primerLink: `What do the stress marks mean?`
- primerTitle: `Lithuanian stress marks`
- primerIntro: `The stressed syllable is pronounced most prominently. Long syllables additionally carry a pitch contour (priegaidė). Three marks are used:`
- primerGraveName: `Grave`
- primerGraveDesc: `short stressed syllable.`
- primerGraveEx: `bùs, vìsas, làbas`
- primerAcuteName: `Acute`
- primerAcuteDesc: `long syllable, falling contour — the start of the syllable is emphasized.`
- primerAcuteEx: `výras, káina, áukštas`
- primerTildeName: `Circumflex`
- primerTildeDesc: `long syllable, rising contour — the end of the syllable is emphasized.`
- primerTildeEx: `nãmas, laũkas, geraĩ`
- primerMixed: `In mixed diphthongs (il, ir, al, an…) the circumflex sits on the sonorant: šil̃tas, var̃das.`
- primerPair: `The contour can distinguish words: áušta (day breaks) vs aũšta (it cools down).`
- primerMore: `More at VLKK: pronunciation and accentuation (in Lithuanian)`

RU:
- primerLink: `Что означают знаки ударения?`
- primerTitle: `Литовские знаки ударения`
- primerIntro: `Ударный слог произносится наиболее отчётливо. В долгих слогах различается ещё и интонация слога (priegaidė). Используются три знака:`
- primerGraveName: `Гравис`
- primerGraveDesc: `краткий ударный слог.`
- primerGraveEx: `bùs, vìsas, làbas`
- primerAcuteName: `Акут`
- primerAcuteDesc: `долгий слог, нисходящая интонация — выделяется начало слога.`
- primerAcuteEx: `výras, káina, áukštas`
- primerTildeName: `Циркумфлекс`
- primerTildeDesc: `долгий слог, восходящая интонация — выделяется конец слога.`
- primerTildeEx: `nãmas, laũkas, geraĩ`
- primerMixed: `В смешанных дифтонгах (il, ir, al, an…) циркумфлекс ставится на сонорном: šil̃tas, var̃das.`
- primerPair: `Интонация различает слова: áušta (светает) и aũšta (остывает).`
- primerMore: `Подробнее — VLKK: произношение и ударение (на литовском)`

## Pass criteria

1. `npm run check` passes (typecheck catches a missed UiStrings field).
2. `npm run build` (or the project's vite build script) passes.
3. Grep the built/source i18n for `šil̃tas` and `aũšta` — combining marks
   intact (U+0303 present).
4. Do NOT deploy. Do NOT start long-running servers; if you start vite
   dev to look, kill it before finishing.

Do not commit.
