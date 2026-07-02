# Kirčiuoklė — Lithuanian text accentuator

A small Cloudflare Workers + TypeScript app that adds stress marks
(kirčiavimas) to arbitrary Lithuanian text, with context-aware
disambiguation of homographs and an LT/EN/RU interface.

Live: https://kirciuokle.alexbalandi.workers.dev

## How it works

The browser only ever talks to this project's `/api/*` routes. The Worker
orchestrates:

1. **Accentuation** — the text is sent to the VDU kirčiuoklė, which returns
   every word's accented form and flags ambiguous words (same spelling,
   different stress).
2. **Contextual disambiguation** — in parallel, the text is tagged with
   UDPipe 2 (lemma, POS, case/gender/number/tense/person). Each ambiguous
   word's variants carry their own morphology labels; the Worker scores them
   against the in-context tag and picks the best match per occurrence. A
   small lemma table handles homographs whose variants share identical
   morphology (e.g. *yra*: *būti* → *yrà*, *irti* → *ỹra*).
3. **Durable dictionary** — every word's variant set is memoized permanently
   in a Workers KV namespace on first sight, verbatim from VDU. Words VDU
   does not know are cached negatively for 30 days.

In the UI, green-underlined words were resolved by context, amber ones are
unresolved ties, and dotted ones are not in the dictionary. Every ambiguous
word is clickable — a popover lists all variants with their morphology
(localized to LT/EN/RU) and lets you override the choice per occurrence.
If the tagger is unreachable, the app degrades gracefully to VDU defaults
and shows a notice.

## External services

The Worker (and the Python CLI) call exactly two external services:

| Service | Endpoint we call | Used for | Notes |
|---|---|---|---|
| **VDU kirčiuoklė** ([kalbu.vdu.lt](https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/)) | `POST https://kalbu.vdu.lt/ajax-call` with form-encoded `action=text_accents` (whole text) or `action=word_accent` (single word variants) | Accent placement, variant lists, morphology labels — the dictionary source of truth | Requires a nonce scraped from the [tool page](https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/) (regex `"NONCE":"([0-9a-f]+)"`), cached ~6 h and refreshed on failure. Same engine/data as kirtis.info. |
| **LINDAT/CLARIN UDPipe 2** ([service page](https://lindat.mff.cuni.cz/services/udpipe/)) | `POST https://lindat.mff.cuni.cz/services/udpipe/api/process` with `tokenizer`, `tagger`, `model=lithuanian-alksnis`, `data=<text>` | Contextual tagging (lemma, POS, morphological features) for homograph disambiguation | Free fair-use REST service by ÚFAL. The [Lithuanian-ALKSNIS model](https://ufal.mff.cuni.cz/udpipe/2) is CC BY-NC-SA. Failure degrades gracefully (defaults + notice). |

Word-variant lookups are served from our own KV dictionary after first
sight, so VDU's `word_accent` is only hit for never-seen words. The
`text_accents` and UDPipe calls currently run once per request.

## Local development

```sh
npm install
npm run dev
```

`npm run dev` runs `vite dev` with the official `@cloudflare/vite-plugin`
(workerd runtime, KV simulated locally).

## Checks

```sh
npm run check              # TypeScript + Vitest
npm run build
npx wrangler deploy --dry-run
```

## Deploying to Cloudflare

Copy `.env.example` to `.env` and fill in a Cloudflare API token (create one
from the "Edit Cloudflare Workers" template at
https://dash.cloudflare.com/profile/api-tokens) and your account ID. Then:

```sh
npm run deploy
```

Wrangler reads the credentials from `.env` automatically. Alternatively,
`npx wrangler login` for interactive OAuth.

## API

`POST /api/accent` with `{ "text": "Čia yra tekstas." }` returns:

```json
{
  "tagger": "ok",
  "parts": [
    { "text": "Čia", "accented": "Čià", "type": "word" },
    { "text": " ", "type": "sep" },
    {
      "text": "yra",
      "accented": "yrà",
      "type": "word",
      "ambiguous": true,
      "resolvedBy": "lemma",
      "chosen": 1,
      "variants": [
        { "form": "ỹra", "info": "vksm., es. l., 3 asm." },
        { "form": "yrà", "info": "vksm., es. l., 3 asm." }
      ]
    }
  ]
}
```

`tagger` is `"ok"` or `"unavailable"`. Word parts may carry `unknown: true`
(not in dictionary), `ambiguous: true` with `variants`/`chosen`, and
`resolvedBy` (`"lemma"` or `"context"`; absent means unresolved — the VDU
default was kept).

`GET /api/word?w=yra` returns `{ "variants": [{ "form": "ỹra", "info": "vksm., es. l., 3 asm." }] }`.

Empty text is rejected with `400`, text over 20 000 characters with `413`,
upstream failures surface as `502`.

## Command-line tools

- `uv run scripts/accent_text.py input.txt` — same pipeline from the
  terminal (VDU + UDPipe + lemma exceptions); reports unknown and
  auto-resolved words on stderr. This is the reference implementation the
  Worker's TypeScript port is tested against.
- `uv run scripts/eval_accenter.py corpus.txt` — differential quality gate:
  scores a candidate accentuation engine (currently LIEPA
  `phonology_engine`) against the production pipeline on a corpus, reporting
  agreement, coverage, and disagreement samples. Run it before swapping any
  engine.

## Localization

The interface and all grammatical terminology are available in Lithuanian,
English, and Russian (e.g. „kilm." → "genitive" / «родительный»). The
hand-reviewed gloss tables live in `src/client/i18n.ts`. The accented words
themselves are, of course, Lithuanian.

## Credits

Accentuation data: [VDU kirčiuoklė](https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/)
(Vytautas Magnus University) — the same database behind
[kirtis.info](https://kirtis.info), which inspired this project. Tagging:
[UDPipe 2](https://ufal.mff.cuni.cz/udpipe/2) by ÚFAL, Charles University,
via the LINDAT/CLARIN infrastructure.
