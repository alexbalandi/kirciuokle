# Kirčiuoklė — Lithuanian text accentuator

## Human-written section

This project was done with a guided agentic coding to achieve simple thing :
have a service on the web that lets you fully accentuate reasonably long
lithuanian text. What started as simple gluework for two services (VDU for
dictionary and UDPipe for POS-tagging) now also has its own small transformer
that can run locally on most consumer hardware.

Lithuanian accentuation is a bit a rabbit hole - different stress marks,
language structure slightly diverging from conventional POS-format, complex
patterns (look at [Yuri Kushnir work](https://yuriykushnir.com/?page=work) to
get a glimpse), certain friction in standards and that's BEFORE we get into
there being several dialects, some of them enough to become their own language
(looking at you, Samogitian).

This project tries to build on the work of others and make strongest and
cheapest possible, fully open-source solution for accentuating Lithuanian text
that learners like me want to read properly.

As you might guess, it doesn't achieve 100% accuracy, but we get close enough
to that on "texts you meet in real life", while still being around 90%
per-token accuracy on "put correct mark into correct place on a word,
including those from literary texts and archaic forms".

### Where to start?

This readme should be enough for you (or your agents) to quickly deploy
locally or on your cloudflare. Just double-check that you get the model files
(all loaded on huggingface) properly. If you want to understand more what work
was done, why and how, check one of the following:

* [Main overview readme](https://github.com/alexbalandi/kirciuokle/blob/main/docs/README.md)
* [Attributions](https://github.com/alexbalandi/kirciuokle/blob/main/ATTRIBUTIONS/README.md)

Unlike this section, they are human-validated not human-written, but they will
give you the idea. If you have questions / suggestions, just file an issue,
I'm also open to any prs. If you want to reach me personally - do it with
email - [alexbalandi@gmail.com](mailto:alexbalandi@gmail.com)

---

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
3. **Durable dictionary, local-first** — the Worker tokenizes the text
   itself and accents it from a D1 dictionary that stores, per word,
   VDU's variants plus the canonical default form and accent type for both
   the lowercase and the capitalized spelling (VDU's answers are
   case-sensitive: *Alyta* vs *alyta*). Never-seen words are fetched from
   VDU on the fly (and remembered); if a request has too many unknown
   words, it transparently falls back to the legacy VDU whole-text path.
   Words VDU does not know are cached negatively for 30 days.
   `ACCENT_SOURCE=local` is the default; `?source=vdu` forces the legacy
   path per request (used by the A/B parity gate).
   `scripts/seed_dictionary.py` pre-warms the dictionary politely from a
   frequency list or a text file.

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

With the local-first path (the default), warm-dictionary requests only call
UDPipe — VDU is consulted solely for never-seen words. The A/B parity gate
(`?source=local` vs `?source=vdu` on a 2,600-token corpus) showed the local
path reproduces the legacy output except where the legacy VDU path itself
misbehaves (escaped apostrophes, swallowed spaces after sentence dots) —
the local path is faithful to the input in those cases.

## Local development

```sh
npm install
npm run dev
```

`npm run dev` runs `vite dev` with the official `@cloudflare/vite-plugin`
(workerd runtime, D1 simulated locally).

The D1 database already exists in Cloudflare. The migration commands for the
orchestrator are:

```sh
npx wrangler d1 migrations apply kirciuokle-words --local
npx wrangler d1 migrations apply kirciuokle-words --remote
```

## Checks

```sh
npm run check              # TypeScript + Vitest
npm run build
npx wrangler deploy --dry-run
```

## Self-hosting locally

The self-hosted replica lives in [`local/`](local/). It packages the built
frontend, a FastAPI port of the Worker pipeline, a SQLite dictionary, and a
swappable UDPipe-compatible tagger. See [`local/README.md`](local/README.md)
for Docker Compose usage, dictionary provisioning, and offline test commands.

## Deploying to Cloudflare

Copy `.env.example` to `.env` and fill in a Cloudflare API token (create one
from the "Edit Cloudflare Workers" template at
https://dash.cloudflare.com/profile/api-tokens) and your account ID. Then:

```sh
npm run deploy:dev     # -> kirciuokle-dev.<subdomain>.workers.dev (serves the model from R2)
npm run deploy:prod    # -> kirciuokle.<subdomain>.workers.dev  (production)
```

Wrangler reads the credentials from `.env` automatically. Alternatively,
`npx wrangler login` for interactive OAuth. The bare `npm run deploy` is
disabled on purpose — pick `dev` or `prod` explicitly. Deploy/rollback flow and
the dev-auto / prod-gated policy live in [`AGENTS.md`](AGENTS.md).

## API

`POST /api/accent` with `{ "text": "Čia yra tekstas." }` returns:

```json
{
  "source": "vdu",
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
- `uv run scripts/seed_dictionary.py --limit 10000` — politely pre-warms the
  D1 dictionary from the Lithuanian frequency list (or
  `--words-from-text file.txt` for a specific text). Resumable; already
  complete words are skipped.

## Localization

The interface and all grammatical terminology are available in Lithuanian,
English, and Russian (e.g. „kilm." → "genitive" / «родительный»). The
hand-reviewed gloss tables live in `src/client/i18n.ts`. The accented words
themselves are, of course, Lithuanian.

## Retraining the local model

The in-browser accentuation model (`joint_vN`) can be retrained, evaluated,
and re-shipped end to end — datasets → training → gates → prune/quantize →
browser bundle → R2/HF. The full runbook, with the exact commands and the
gotchas that bite, is in [docs/RETRAIN.md](docs/RETRAIN.md).

## License

The code in this repository is released into the public domain under
[The Unlicense](LICENSE) — use it however you wish, no attribution
required.

Data sources, corpora, services, and model artifacts keep their own terms.
Dictionary contents are not distributed with this repository — you warm your
own instance via normal use or `scripts/seed_dictionary.py`.

## Attributions

See [ATTRIBUTIONS/README.md](ATTRIBUTIONS/README.md) for the full list of
upstream services, corpora, models, benchmark tools, licenses, and citations.
