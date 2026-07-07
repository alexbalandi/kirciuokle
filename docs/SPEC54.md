# SPEC54 — Local model versioning: keep old, optionally update to new

## Goal

When a newer model is released to R2, a returning user whose browser already
has an older model cached must: (a) keep using their cached model with **no
forced re-download**, and (b) see an **optional** "Update available (X MB)"
control they can click to fetch the new version. On update, download the new
file, switch to it, and purge the old cache entry. Old versions stay usable.

Introducing the scheme requires **no re-upload of the big model files** — the
current files become "version 1"; only `manifest.json` gains a `version` and is
re-uploaded. Future releases use versioned filenames so old + new coexist in R2.

Constraints: do NOT rename or re-upload the current `.onnx` bundle files. Keep
the change surgical — reuse the existing tier / consent / cache plumbing built
in SPEC53b. `npm run check` + `build` green. Do not deploy or commit.

## Manifest schema (`src/client/local/types.ts` `ModelManifest`)

- Add top-level optional `version?: string` (release id). Keep `tiers`
  (tier→filename), `models` (filename→`{bytes, sha256, tier, version?}`).
- `resolveModelTierInfo` should also surface the version: extend
  `LocalModelTierInfo` with `version: string | null` (= `manifest.version ??
  null`).

## Freshness

- `fetchJson` (assets.ts): add an optional `RequestInit` param. Fetch
  `manifest.json` with `{ cache: "no-store" }` (in `loadModelAssets` and
  `loadModelTierInfo`) so the client always sees the current version. Leave
  `joint.meta.json` / `label_bridge.json` as-is (immutable is fine — they are
  version-invariant here).
- Worker (`src/worker/index.ts` `handleLocalModel` / `modelContentType`): serve
  `.json` with `cache-control: no-cache, must-revalidate`; keep `.onnx`,
  `.wasm`, `.mjs`, `.js`, `.model` at `public, max-age=31536000, immutable`.
  Everything else unchanged. Update the worker unit test accordingly.

## Active-version tracking (assets.ts)

- localStorage key: `accent-local-active-v1-<tier>` → JSON
  `{ file: string, bytes: number, version: string }` (the model the user is
  currently on for that tier). Add read/write helpers.
- Add `resolveActiveLoad(manifest, tier)` returning:
  `{ loadFile, loadBytes, loadVersion, updateAvailable, updateFile,
     updateBytes, updateVersion }`.
  - `current` = from `resolveModelTierInfo` (file, bytes, version).
  - `active` = stored record for tier.
  - If `active` exists AND `active.file` is cached with a valid byte-length
    (reuse `cacheHit(url, active.bytes)`): load `active`; `updateAvailable =
    current.file !== active.file`; the `update*` fields describe `current`.
  - Else: load `current`; `updateAvailable = false`.
- `loadModelAssets(tier)` loads `loadFile` (not always current). On a
  successful load, if `loadFile === current.file`, persist `active = current`.
  Return `updateAvailable` + `update` (`{file,bytes,version}` | null) +
  `loadVersion` in `LoadedModelAssets`. Thread these through the `ready` status
  event and `LocalStats` so the UI + stats popover can show them.
- `hasCachedLocalModel(tier)`: check the **active** file if a record exists,
  else the current tier file. (So a returning user with an old cached model
  reaches the "ready" path, not "needs-consent".)

## Update action (assets.ts + engine.ts + main.ts)

- assets.ts `updateToCurrentModel(tier, onStatus)`: fetch fresh manifest →
  current; `fetchWithCache(currentUrl, currentBytes, onStatus)`; on success,
  purge the OLD active file (`deleteCachedModel`) and persist `active =
  current`. Returns the new tier info.
- main.ts: an "update" handler that runs `updateToCurrentModel(activeTier)` then
  disposes + recreates the local engine (so it loads the new active), reusing
  the existing status line for progress. Guard against concurrent runs.

## UI (`index.html` + `src/client/main.ts` + `style.css`)

- In the collapsible `.panel-extras` block, when the model is ready AND
  `updateAvailable`, render a compact, localized notice with a button:
  "Update available — new model ({size})" / "Update". Clicking runs the update
  handler; disable while updating; on success the notice disappears.
- In the WASM-stats popover, add a subtle "Model version: {version}" line.

## Localization (`src/client/i18n.ts`, LT/EN/RU)

Add: `localUpdateAvailable`, `localUpdateButton` ("Update to new model
({size})"), `localUpdating`, `statsModelVersion`. Keep LT correct — VERIFY any
LT stress marks against the VDU accentuator (do not guess).

## Generator (`scripts/prepare_local_model.py`)

- Add `--version` (default: first 10 hex of the heavy model's sha256). Write
  `manifest["version"]`.
- Add `--versioned-filenames`: when set, copy each shipped model to
  `<stem>-<version>.onnx` and set `tiers` / `models` to the versioned names
  (for FUTURE releases so old + new coexist in R2). When NOT set (this release),
  keep the current filenames and only add `version`.

## Verification

1. `npm run check` + `build` green; report test count.
2. Unit tests (mock the Cache API) for `resolveActiveLoad`:
   - active file cached → loads active, `updateAvailable` true when current
     differs;
   - active recorded but not cached → loads current;
   - no active record → loads current, no update.
3. Worker test: `.json` gets `no-cache`, `.onnx` stays `immutable`.
4. Playwright (LIGHT tier or fully mocked fetch — never the 470 MB path): first
   load records active; simulate a manifest `version` + filename bump → the
   "Update available" control appears; clicking it downloads the new file
   (mocked), updates active, purges old, and the notice clears. Screenshot the
   update control.

Do not commit, do not deploy.
