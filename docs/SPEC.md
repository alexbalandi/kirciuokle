# Kirčiuoklė — Lithuanian text accentuator (Cloudflare Workers + TypeScript)

Build a small, clean, production-quality web app that fully stresses (accentuates)
arbitrary Lithuanian text and lets the user copy the result. Everything in
TypeScript. Deployable to Cloudflare with `wrangler deploy`.

## Upstream API (already reverse-engineered and verified — use exactly this)

Engine: VDU kirčiuoklė at `https://kalbu.vdu.lt` (same data as kirtis.info).
No cookies needed. All requests are plain form-encoded POSTs.

### 1. Nonce
`GET https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/` (HTML page).
Extract with regex `/"NONCE":"([0-9a-f]+)"/`. The nonce is a WordPress-style
nonce; cache it (module-level variable with timestamp, TTL ~6 hours). If any
API call below returns `code != 200` or unparseable JSON, refresh the nonce
once and retry.

### 2. Accent full text
```
POST https://kalbu.vdu.lt/ajax-call
Content-Type: application/x-www-form-urlencoded
action=text_accents&nonce=<nonce>&body=<text>
```
Response: `{"code":200,"message":"<stringified JSON>"}` where the parsed
message is:
```json
{"textParts":[
  {"string":"Čia","accented":"Čià","accentType":"ONE","type":"WORD"},
  {"string":" ","type":"SEPARATOR"},
  {"string":"yra","accented":"ỹra","accentType":"MULTIPLE_MEANING","type":"WORD"},
  {"string":"Velvet","accentType":"NONE","type":"WORD"},
  {"string":"Waver","type":"NON_LT"}
]}
```
- `accentType: "ONE"` — unambiguous, `accented` present.
- `accentType: "MULTIPLE_MEANING"` — ambiguous, `accented` holds the default
  variant; the user must be able to switch variants in the UI.
- `accentType: "NONE"` — word not in dictionary, no `accented`.
- `type: "NON_LT"` — token with non-Lithuanian characters, no `accented`.
- Accents come as combining marks (U+0300 grave, U+0301 acute, U+0303 tilde).
- Input limit: the upstream UI caps at 5000 chars. Chunk request bodies at
  ≤4500 chars, splitting on sentence boundaries (`. ! ? \n`, falling back to
  space, falling back to hard cut), and concatenate the resulting textParts.

### 3. Word variants (for ambiguous words)
```
POST https://kalbu.vdu.lt/ajax-call
action=word_accent&nonce=<nonce>&word=<word>
```
Parsed message:
```json
{"accentInfo":[
  {"accented":["ỹra"],"information":[{"mi":"vksm., es. l., 3 asm."}]},
  {"accented":["yrà"],"information":[{"mi":"vksm., es. l., 3 asm."}]}
]}
```
`information[].mi` is the morphology label; `information[].meaning` sometimes
present. Flatten to a list of `{form, info}` (info = joined `mi` strings,
include `meaning` when present).

## Architecture

Single Cloudflare Worker serving both the static frontend (Workers static
assets) and a JSON API that proxies the upstream (browser cannot call VDU
directly — no CORS there).

Use the official `@cloudflare/vite-plugin` setup: `vite dev` for local dev
(runs the worker in workerd), `vite build` + `wrangler deploy` to ship.
Latest stable deps. No frontend framework — vanilla TS + CSS.

```
package.json            scripts: dev, build, check (tsc + vitest run), deploy, cf-typegen
wrangler.jsonc          main = worker entry, assets binding with SPA not_found_handling
vite.config.ts
tsconfig.json (+ per-env configs as needed)
index.html
src/client/main.ts      UI logic
src/client/style.css
src/shared/types.ts     API types shared by worker & client
src/worker/index.ts     routing, error handling
src/worker/vdu.ts       upstream client: nonce cache, text_accents, word_accent, chunking
test/                   vitest unit tests
README.md
```

## Worker API

- `POST /api/accent` body `{"text": string}` →
  `{"parts": Part[]}` where
  `Part = {"text": string, "accented"?: string, "type": "word"|"sep", "ambiguous"?: true, "unknown"?: true}`
  (map `NON_LT` and `accentType NONE` to `type:"word", unknown:true`;
  SEPARATOR → `type:"sep"`). Reject empty text (400) and text > 20000 chars
  (413) with `{"error": string}`.
- `GET /api/word?w=<word>` → `{"variants": [{"form": string, "info": string}]}`.
  Cache successful responses in `caches.default` for 7 days (synthetic cache
  key URL; `waitUntil` for the cache write).
- Upstream failures → 502 with `{"error": string}`. Never leak stack traces.

## Frontend UX (Lithuanian labels)

- Title **Kirčiuoklė**, one-line description: „Įklijuokite lietuvišką tekstą —
  gausite pilnai sukirčiuotą.“
- Autosizing `<textarea>`, button **Sukirčiuoti** (also Ctrl+Enter). Char
  counter (max 20000). Loading state on the button while fetching.
- Result panel rendering the parts stream:
  - unambiguous words: plain text (accented form);
  - ambiguous words: subtle amber underline; **click opens a small popover**
    listing all variants from `/api/word` with their morphology labels;
    clicking a variant replaces that occurrence (client-side cache of variant
    lookups per word). Popover closes on Esc / outside click.
  - unknown / non-LT words: grey dotted underline, `title="Žodyne nerasta"`.
- **Kopijuoti** button: `navigator.clipboard.writeText` of the current visible
  stressed text (reflecting the user's variant choices); brief „Nukopijuota ✓“
  feedback. Keep a legend explaining the two underline styles.
- Footer: „Duomenys: VDU kirčiuoklė (kalbu.vdu.lt) · įkvėpta kirtis.info“ with
  links.
- Clean minimal styling, system font stack, `prefers-color-scheme: dark`
  support, sensible max-width, works on mobile.

## Quality bar

- `npm run check` (typecheck + vitest) and `npm run build` must pass.
- `npx wrangler deploy --dry-run` must succeed.
- Unit tests (vitest) for: chunking logic (boundaries, long-word fallback),
  textParts→Part normalization, nonce extraction regex, variant flattening.
  Mock `fetch` for upstream tests — tests must not hit the network.
- Handle NFC: normalize output text with `.normalize("NFC")` before display
  and copying.
- Do NOT touch `scripts/` (existing Python tooling) or `docs/`.
- README: what it is, local dev, deploy to Cloudflare (wrangler login +
  `npm run deploy`), API contract, credit to VDU & kirtis.info, note that the
  `scripts/accent_text.py` CLI does the same from the terminal via uv.
