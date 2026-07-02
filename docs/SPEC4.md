# Phase 4 — durable word dictionary in Workers KV

Replace the ephemeral `caches.default` variant caching with a permanent
Workers KV dictionary: once a word's accent variants have been fetched from
VDU, they are ours forever. This shrinks VDU dependence for `word_accent`
lookups to first-sight-only and keeps variant popovers + disambiguation
working from our own data when VDU is flaky.

## KV namespace (already created)

Add to `wrangler.jsonc`:
```jsonc
"kv_namespaces": [
  { "binding": "WORDS", "id": "e4570710b6c047dabc4456e2c7106e57" }
]
```
`Env` gains `WORDS: KVNamespace`.

## Storage model

- Key: the word, NFC-normalized, lowercased (existing `normalizeWordKey`).
- Value (JSON): `{ "variants": AccentVariant[], "fetchedAt": <ISO string> }`
  where `AccentVariant = {form, info, mi}` exactly as today. Store variants
  verbatim from VDU — this is a memoization of the source of truth, never a
  transformation.
- **Negative results are stored too**: a word VDU has no data for stores
  `{ "variants": [], "fetchedAt": ... }`. This avoids re-asking VDU for
  every unknown foreign word. (Store negatives with `expirationTtl` of 30
  days so VDU dictionary additions eventually surface; positive entries are
  permanent — accent data does not rot.)

## Code changes

- `lookupWordVariantsCached(word, ctx)` becomes
  `lookupWordVariantsKV(word, env, ctx)`:
  1. `env.WORDS.get(key, "json")` → hit: return variants.
  2. Miss: `lookupWordVariants(word)` from VDU → `ctx.waitUntil(env.WORDS.put(...))`
     (with `expirationTtl: 30*24*3600` only when variants are empty).
  3. If VDU errors on a miss: propagate the current behavior (caller already
     treats failures as empty variant lists per-word).
- Thread `env` through: `handleAccent`/`handleWord` in `src/worker/index.ts`
  pass `env` to the lookup; the `accentText` options callback closes over it.
- Remove the `caches.default` logic (KV reads are edge-cached by Cloudflare
  anyway); keep the `cache-control` response header on `/api/word`.
- The `/api/accent` and `/api/word` behavior and response shapes are
  unchanged.

## Quality bar

- Unit tests (mock `KVNamespace` with an in-memory Map): KV hit skips VDU
  fetch; miss fetches VDU then writes KV; empty-variant result written with
  `expirationTtl`; malformed KV JSON treated as miss (and overwritten).
- `npm run check`, `npm run build`, `npx wrangler deploy --dry-run` pass.
- Do not modify `scripts/` or `docs/`.
