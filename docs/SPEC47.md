# SPEC47 — Local-mode design fixes (post-review)

Four fixes to the SPEC46 implementation on the main site. Files:
`src/client/**`, `index.html`, tests. Keep `npm run check`/`build`
green; extend tests where behavior changes.

## 1. Consent gate before the model download (REQUIRED)

Switching to Local must NOT auto-download. Show an inline consent card
in the local-status area: localized text (LT/EN/RU) — EN: "To
accentuate locally, the site downloads the model once — about {size}
of traffic. It stays saved in your browser for future visits." with a
primary button "Download model ({size})" carrying a CLOUD-ARROW-DOWN
inline SVG icon (a fetch-from-cloud glyph — deliberately not the
tray/save-file icon). Only the button starts `ensureEngine`. If the
model is already in Cache API (probe cheaply via cache match before
deciding), skip the card and load directly. Mode preference may persist
as "local" but a fresh visit with no cached model shows the card again
instead of downloading. Add a vitest for the state logic (no download
side effect until consent; skip when cache hit) and extend the
Playwright harness: flipping to Local asserts NO /local-model/ model
fetch happens before the button click.

## 2. User-facing units: MB not MiB

Display sizes as decimal-ish MB with no unit pedantry: "~538 MB" (bytes
/ 1e6, rounded). Keep whatever internal math; the stats popover may
keep precise bytes. Update the explainer/consent/progress strings.

## 3. Percent formatting

`formatProbability`: >=10% → integer ("67%"); <10% → one decimal
("8.4%"). Update tests.

## 4. Hyphenated numerals ("81-erių")

The word regex splits "81-erių" so the local pipeline scores the bare
fragment "erių" (nonsense POS at high confidence). Fix in the local
tokenization: treat `\d+(?:[.,]\d+)?-<letters>` as ONE display token
whose ACCENTABLE part is the letter suffix — accent the suffix in
place (model input = suffix only), but mark the token as a numeral
fragment: no POS popover (or popover shows a localized "numeral
ending" line instead of model labels), never the ambiguous/amber
class. Web mode behavior unchanged. Unit test: "81-erių vilnietė" →
two tokens, first renders accented suffix with no probability chips.

## Pass criteria

1. check + build green; new/updated tests listed.
2. Playwright: consent flow (no fetch before click), cached-skip path,
   MB strings, percent formats, "81-erių" popover behavior.
3. Screenshot of the consent card saved to docs-consent.png.

Do not commit, do not deploy.
