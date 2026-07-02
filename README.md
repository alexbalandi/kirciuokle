# Kirčiuoklė

Maža Cloudflare Workers ir TypeScript aplikacija lietuviškam tekstui sukirčiuoti. Naršyklė kalba tik su šio projekto `/api/*` maršrutais, o Worker tarpininkauja VDU kirčiuoklei `kalbu.vdu.lt` ir kontekstinei UDPipe analizei.

## Vietinis darbas

```sh
npm install
npm run dev
```

`npm run dev` paleidžia `vite dev`, kuris naudoja oficialų `@cloudflare/vite-plugin` ir workerd runtime.

## Kokybės patikros

```sh
npm run check
npm run build
npx wrangler deploy --dry-run
```

`npm run check` vykdo TypeScript patikrą ir Vitest testus.

## Diegimas į Cloudflare

```sh
npx wrangler login
npm run deploy
```

`npm run deploy` kviečia `wrangler deploy`.

## API

`POST /api/accent`

```json
{ "text": "Čia yra tekstas." }
```

Grąžina:

```json
{
  "tagger": "ok",
  "parts": [{ "text": "Čia", "accented": "Čià", "type": "word" }]
}
```

`GET /api/word?w=yra`

```json
{ "variants": [{ "form": "ỹra", "info": "vksm., es. l., 3 asm." }] }
```

Tuščias tekstas atmetamas su `400`, tekstas virš 20000 simbolių su `413`, o upstream klaidos grąžinamos kaip `502`.

## Duomenys

Kirčiavimo duomenys gaunami iš VDU kirčiuoklės (`kalbu.vdu.lt`), tos pačios duomenų bazės, kuria remiasi `kirtis.info`. Dviprasmiškiems žodžiams parinkti naudojamas nemokamas sąžiningo naudojimo LINDAT UDPipe 2 REST servisas su `lithuanian-alksnis` modeliu (modelio licencija CC BY-NC-SA).

Terminalui skirta `scripts/accent_text.py` CLI daro tą patį iš komandinės eilutės per `uv`.
