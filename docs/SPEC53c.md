# SPEC53c — R2 model serving + dev deploy environment (code + config only)

R2 is not yet enabled on the account, so this is CODE + CONFIG ONLY —
write it so it's ready to deploy the moment R2 is on. Do NOT deploy, do
NOT run wrangler apart from `--dry-run`. Files: `src/worker/**`,
`wrangler.jsonc`, `local/README.md`.

## Worker: serve /local-model/* from R2 when bound

- In the Worker, add a handler: if the request path starts with
  `/local-model/` AND an R2 binding `MODEL_BUCKET` exists on `env`,
  serve the object from R2:
  - key = path after `/local-model/`; `env.MODEL_BUCKET.get(key)`;
    404 if missing.
  - headers: `Cross-Origin-Resource-Policy: same-origin`,
    `Cross-Origin-Embedder-Policy: require-corp`,
    `Cross-Origin-Opener-Policy: same-origin`,
    `cache-control: public, max-age=31536000, immutable`,
    and content-type by extension (.onnx → application/octet-stream,
    .wasm → application/wasm, .mjs/.js → text/javascript; charset=utf-8,
    .json → application/json; charset=utf-8, .model → application/octet-stream).
    Pass through the R2 object's httpEtag; support Range requests if
    trivial (optional).
  - If `MODEL_BUCKET` is absent (current prod), do nothing — fall through
    to existing behavior (assets/SPA), unchanged.
- The HTML document response must carry `Cross-Origin-Opener-Policy:
  same-origin` and `Cross-Origin-Embedder-Policy: require-corp` so the
  page is cross-origin isolated (WASM threads). Add these to the
  document response only when serving the SPA shell; verify it does not
  break the existing /api/* or asset responses.
- Type the binding: extend the Worker `Env` with
  `MODEL_BUCKET?: R2Bucket`.

## wrangler.jsonc: add a `dev` environment

- Add `env.dev` with: `name: "kirciuokle-dev"`, its own `assets` (same
  dir), the same `d1_databases` DICT binding, `vars.ACCENT_SOURCE:
  "local"`, and `r2_buckets: [{ binding: "MODEL_BUCKET", bucket_name:
  "kirciuokle-models" }]`.
- Leave the TOP-LEVEL (prod) config unchanged — NO r2_buckets at top
  level. The diff must show only an added `env.dev` block (+ optional
  `$schema`-safe formatting).

## local/README.md runbook (commented, do not run)

Add a short "Deploy the in-browser (Local mode) dev site" section:

```sh
# One-time, after enabling R2 in the Cloudflare dashboard:
wrangler r2 bucket create kirciuokle-models
# Upload the generated bundle (uv run scripts/prepare_local_model.py ... first):
#   for each file in local-model/ (recursively):
#   wrangler r2 object put kirciuokle-models/<relpath> --file local-model/<relpath>
npm run build
wrangler deploy --env dev      # deploys https://kirciuokle-dev.<subdomain>.workers.dev
```

## Pass criteria

1. `npm run check` + `build` green.
2. `wrangler deploy --env dev --dry-run` succeeds (config + worker
   compile); paste the output. (Dry run does not need R2 enabled.)
3. `git diff wrangler.jsonc` shows only an added env.dev; prod unchanged.
4. Worker unit/logic: a request to `/local-model/x` with no
   MODEL_BUCKET falls through unchanged (existing tests still pass).

Do not commit, do not deploy (beyond --dry-run).
