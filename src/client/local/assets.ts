import type {
  CacheStatus,
  JointMeta,
  LabelBridge,
  LocalModelStatus,
  ManifestRuntimeFile,
  ModelManifest,
} from "./types";

export const LOCAL_MODEL_BASE = "/local-model/";
export const LOCAL_MODEL_CACHE_NAME = "main-local-accent-model-v1";
export const LOCAL_DEFAULT_MODEL_FILE = "joint.int8.partial.onnx";
export const LOCAL_MODEL_SIZE_FALLBACK = 470_223_894;

const CACHE_CHUNK_BYTES = 16 * 1024 * 1024;
const CACHE_HEADROOM_MULTIPLIER = 1.2;
const CACHE_WRITE_WATCHDOG_MS = 20_000;

export type CacheWriteState = {
  url: string;
  status: CacheStatus;
  attempted: boolean;
  completed: boolean;
  skipped: boolean;
  watchdogFired: boolean;
  reason: string | null;
  error: string | null;
  start?: () => Promise<CacheStatus>;
};

type CacheEligibility = {
  ok: boolean;
  reason: string;
  usage: number | null;
  quota: number | null;
  headroom: number | null;
  required: number | null;
};

export type LoadedModelAssets = {
  manifest: ModelManifest;
  meta: JointMeta;
  bridge: LabelBridge;
  modelBytes: Uint8Array;
  modelFile: string;
  expectedBytes: number | null;
  cacheStatus: CacheStatus;
  cacheWriteState: CacheWriteState | null;
};

export type ModelStatusSink = (status: LocalModelStatus) => void;

export function modelBaseUrl(): string {
  return new URL(LOCAL_MODEL_BASE, window.location.href).href;
}

export function runtimeBaseUrl(): string {
  return new URL("runtime/", modelBaseUrl()).href;
}

export async function hasCachedLocalModel(
  modelFile = LOCAL_DEFAULT_MODEL_FILE,
): Promise<boolean> {
  const modelUrl = new URL(modelFile, modelBaseUrl()).href;
  return cacheHit(modelUrl);
}

export async function loadModelAssets(
  onStatus: ModelStatusSink = () => {},
): Promise<LoadedModelAssets> {
  onStatus({ type: "metadata" });

  const base = modelBaseUrl();
  const [manifest, meta, bridge] = await Promise.all([
    fetchJson<ModelManifest>(new URL("manifest.json", base).href),
    fetchJson<JointMeta>(new URL("joint.meta.json", base).href),
    fetchJson<LabelBridge>(new URL("label_bridge.json", base).href),
  ]);

  await verifyRuntimeManifestFiles(manifest, onStatus);

  const modelFile = manifest.default_model || meta.int8_onnx || LOCAL_DEFAULT_MODEL_FILE;
  const modelUrl = new URL(modelFile, base).href;
  const expectedBytes =
    manifest.models?.[modelFile]?.bytes ??
    manifest.model_bytes ??
    (await headContentLength(modelUrl));
  const cacheState = await cacheHit(modelUrl);
  const threads = preferredWasmThreads();

  onStatus({
    type: "modelInfo",
    expectedBytes,
    cacheState,
    threads,
  });

  const modelLoad = await fetchWithCache(modelUrl, expectedBytes, onStatus);
  const modelSha = manifest.models?.[modelFile]?.sha256;
  if (modelSha) {
    await assertSha256(modelLoad.bytes, modelSha, modelFile);
  }

  return {
    manifest,
    meta,
    bridge,
    modelBytes: modelLoad.bytes,
    modelFile,
    expectedBytes,
    cacheStatus: modelLoad.cacheWriteState?.status ?? modelLoad.cacheStatus,
    cacheWriteState: modelLoad.cacheWriteState,
  };
}

export function preferredWasmThreads(): number {
  return crossOriginIsolated
    ? Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)))
    : 1;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) {
    return "?";
  }

  return `${Math.round(Number(bytes) / 1_000_000)} MB`;
}

async function verifyRuntimeManifestFiles(
  manifest: ModelManifest,
  onStatus: ModelStatusSink,
): Promise<void> {
  const runtimeFiles = manifest.runtime?.files ?? {};
  const runtimePath = manifest.runtime?.path ?? "runtime/";
  const entries = Object.entries(runtimeFiles).filter((entry): entry is [
    string,
    ManifestRuntimeFile,
  ] => Boolean(entry[1]?.sha256));

  for (const [file, info] of entries) {
    const url = new URL(`${runtimePath}${file}`, modelBaseUrl()).href;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }

    const bytes = await readResponseBytes(response, info.bytes, false, (received, total) => {
      onStatus({
        type: "verify-runtime",
        file,
        received,
        total: total ?? info.bytes,
      });
    });
    await assertSha256(bytes, info.sha256, file);
  }
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }

  return (await response.json()) as T;
}

async function headContentLength(url: string): Promise<number | null> {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return Number(response.headers.get("content-length") || 0) || null;
  } catch {
    return null;
  }
}

async function cacheHit(url: string): Promise<boolean> {
  const cache = await openModelCache();
  if (!cache) {
    return false;
  }

  return Boolean(
    (await cache.match(cacheRequest(url))) ||
      (await cache.match(cacheChunkMetadataRequest(url))),
  );
}

async function openModelCache(): Promise<Cache | null> {
  if (!("caches" in window)) {
    return null;
  }

  try {
    return await caches.open(LOCAL_MODEL_CACHE_NAME);
  } catch (error) {
    logCacheWriteIssueOnce(error);
    return null;
  }
}

function cacheRequest(url: string): Request {
  return new Request(url, { credentials: "same-origin" });
}

function cacheChunkMetadataRequest(url: string): Request {
  return new Request(`${url}?local-cache=chunks`, { credentials: "same-origin" });
}

function cacheChunkRequest(url: string, index: number): Request {
  return new Request(`${url}?local-cache=chunk-${index}`, {
    credentials: "same-origin",
  });
}

async function readChunkedCacheBytes(
  cache: Cache,
  url: string,
  expectedBytes: number | null,
  onStatus: ModelStatusSink,
): Promise<Uint8Array | null> {
  const metadataResponse = await cache.match(cacheChunkMetadataRequest(url));
  if (!metadataResponse) {
    return null;
  }

  let metadata: unknown;
  try {
    metadata = await metadataResponse.json();
  } catch {
    return null;
  }

  if (!isChunkMetadata(metadata)) {
    return null;
  }

  if (expectedBytes && metadata.bytes !== expectedBytes) {
    return null;
  }

  const chunks: Uint8Array[] = [];
  let received = 0;
  for (let index = 0; index < metadata.chunks; index += 1) {
    const chunkResponse = await cache.match(cacheChunkRequest(url, index));
    if (!chunkResponse) {
      return null;
    }

    const chunk = new Uint8Array(await chunkResponse.arrayBuffer());
    chunks.push(chunk);
    received += chunk.byteLength;
    onStatus({
      type: "transfer",
      cached: true,
      received,
      total: metadata.bytes,
    });
  }

  if (received !== metadata.bytes) {
    return null;
  }

  return assembleBytes(chunks, received);
}

function isChunkMetadata(value: unknown): value is {
  version: 1;
  bytes: number;
  chunks: number;
} {
  if (!value || typeof value !== "object") {
    return false;
  }

  const record = value as Record<string, unknown>;
  return (
    record.version === 1 &&
    Number.isInteger(record.bytes) &&
    Number(record.bytes) > 0 &&
    Number.isInteger(record.chunks) &&
    Number(record.chunks) > 0
  );
}

async function fetchWithCache(
  url: string,
  expectedBytes: number | null,
  onStatus: ModelStatusSink,
): Promise<{
  bytes: Uint8Array;
  cacheStatus: CacheStatus;
  cacheWriteState: CacheWriteState | null;
}> {
  const cache = await openModelCache();
  let response = cache ? await cache.match(cacheRequest(url)) : null;
  const cached = Boolean(response);
  let cacheStatus: CacheStatus = cache ? (cached ? "hit" : "miss") : "unavailable";
  let cacheWriteState: CacheWriteState | null = null;
  let cacheEligibility: CacheEligibility | null = null;

  if (!response && cache) {
    const bytes = await readChunkedCacheBytes(cache, url, expectedBytes, onStatus);
    if (bytes) {
      return { bytes, cacheStatus: "hit", cacheWriteState: null };
    }
  }

  if (!response) {
    response = await fetch(url);
    if (!response.ok) {
      throw new Error(`${url}: ${response.status}`);
    }

    if (cache) {
      const modelBytes =
        expectedBytes || Number(response.headers.get("content-length") || 0);
      cacheEligibility = await estimateCacheWriteHeadroom(modelBytes);
    }
  }

  const bytes = await readResponseBytes(response, expectedBytes, cached, (received, total) => {
    onStatus({ type: "transfer", cached, received, total });
  });

  if (expectedBytes && bytes.byteLength !== expectedBytes) {
    throw new Error(
      `${url}: received ${bytes.byteLength} bytes, expected ${expectedBytes}.`,
    );
  }

  if (cache && !cached) {
    cacheWriteState = startCacheWriteAfterMainRead(cache, url, cacheEligibility);
    cacheStatus = cacheWriteState.status;
  }

  return { bytes, cacheStatus, cacheWriteState };
}

async function readResponseBytes(
  response: Response,
  expectedBytes: number | null,
  cached: boolean,
  onProgress: (received: number, total: number | null) => void,
): Promise<Uint8Array> {
  const total = Number(response.headers.get("content-length") || expectedBytes || 0) || null;

  if (!response.body) {
    const out = new Uint8Array(await response.arrayBuffer());
    onProgress(out.byteLength, total ?? out.byteLength);
    return out;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    chunks.push(value);
    received += value.byteLength;
    onProgress(received, total);
  }

  if (cached && total && received !== total) {
    throw new Error(`Cached model ended at ${received} bytes, expected ${total}.`);
  }

  return assembleBytes(chunks, received);
}

function assembleBytes(chunks: Uint8Array[], received: number): Uint8Array {
  const out = new Uint8Array(received);
  let offset = 0;

  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }

  chunks.length = 0;
  return out;
}

async function estimateCacheWriteHeadroom(
  modelBytes: number | null,
): Promise<CacheEligibility> {
  const bytes = Number(modelBytes);
  const required =
    Number.isFinite(bytes) && bytes > 0 ? bytes * CACHE_HEADROOM_MULTIPLIER : null;

  if (!required) {
    return {
      ok: false,
      reason: "unknown-model-size",
      usage: null,
      quota: null,
      headroom: null,
      required,
    };
  }

  if (typeof navigator.storage?.estimate !== "function") {
    return {
      ok: false,
      reason: "storage-estimate-unavailable",
      usage: null,
      quota: null,
      headroom: null,
      required,
    };
  }

  try {
    const estimate = await navigator.storage.estimate();
    const usage = Number(estimate.usage || 0);
    const quota = Number(estimate.quota);

    if (!Number.isFinite(quota)) {
      return {
        ok: false,
        reason: "storage-quota-unavailable",
        usage: Number.isFinite(usage) ? usage : null,
        quota: null,
        headroom: null,
        required,
      };
    }

    const safeUsage = Number.isFinite(usage) ? usage : 0;
    const headroom = Math.max(0, quota - safeUsage);
    return {
      ok: headroom >= required,
      reason: headroom >= required ? "sufficient-quota" : "insufficient-quota",
      usage: safeUsage,
      quota,
      headroom,
      required,
    };
  } catch (error) {
    return {
      ok: false,
      reason: `storage-estimate-failed:${errorMessage(error)}`,
      usage: null,
      quota: null,
      headroom: null,
      required,
    };
  }
}

function makeCacheWriteState(url: string): CacheWriteState {
  return {
    url,
    status: "miss",
    attempted: false,
    completed: false,
    skipped: false,
    watchdogFired: false,
    reason: null,
    error: null,
  };
}

function startCacheWriteAfterMainRead(
  cache: Cache,
  url: string,
  eligibility: CacheEligibility | null,
): CacheWriteState {
  const state = makeCacheWriteState(url);
  const decision =
    eligibility ?? {
      ok: false,
      reason: "storage-estimate-not-started",
      usage: null,
      quota: null,
      headroom: null,
      required: null,
    };

  if (!decision.ok) {
    state.status = "unavailable";
    state.completed = true;
    state.skipped = true;
    state.reason = decision.reason;
    return state;
  }

  state.start = async () => {
    if (state.attempted) {
      return state.status;
    }

    state.attempted = true;
    try {
      await writeCacheFromIndependentFetch(cache, url, state);
      state.status = "stored";
      state.completed = true;
    } catch (error) {
      state.status = "failed";
      state.completed = true;
      state.reason = state.watchdogFired ? "watchdog-timeout" : "cache-write-failed";
      state.error = errorMessage(error);
      logCacheWriteIssueOnce(error);
    }

    return state.status;
  };

  return state;
}

async function writeCacheFromIndependentFetch(
  cache: Cache,
  url: string,
  state: CacheWriteState,
): Promise<void> {
  const controller = new AbortController();
  let cacheResponse: Response | null = null;
  let timeoutId = 0;
  let rejectWatchdog: ((reason?: unknown) => void) | null = null;
  const timeout = new Promise<never>((_, reject) => {
    rejectWatchdog = reject;
  });
  const touchWatchdog = () => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      state.watchdogFired = true;
      cancelCacheWriteBranch(controller, cacheResponse);
      rejectWatchdog?.(new Error("Cache API write watchdog timed out"));
    }, CACHE_WRITE_WATCHDOG_MS);
  };

  touchWatchdog();
  const operation = (async () => {
    cacheResponse = await fetch(cacheRequest(url), { signal: controller.signal });
    if (!cacheResponse.ok) {
      throw new Error(`${url}: ${cacheResponse.status}`);
    }

    touchWatchdog();
    await writeChunkedCacheResponse(cache, url, cacheResponse, touchWatchdog);
  })();
  operation.catch(() => {});

  try {
    await Promise.race([operation, timeout]);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function writeChunkedCacheResponse(
  cache: Cache,
  url: string,
  response: Response,
  touchWatchdog: () => void,
): Promise<void> {
  await cache.delete(cacheChunkMetadataRequest(url));

  const total = Number(response.headers.get("content-length") || 0) || null;
  const chunks: number[] = [];
  let received = 0;
  let chunkIndex = 0;
  let buffer = new Uint8Array(CACHE_CHUNK_BYTES);
  let offset = 0;

  const writeChunk = async (bytes: Uint8Array) => {
    await cache.put(
      cacheChunkRequest(url, chunkIndex),
      new Response(arrayBufferFromBytes(bytes), {
        headers: {
          "content-type": "application/octet-stream",
          "x-local-cache-chunk": String(chunkIndex),
        },
      }),
    );
    chunks.push(bytes.byteLength);
    chunkIndex += 1;
    touchWatchdog();
  };

  const appendBytes = async (value: Uint8Array) => {
    let sourceOffset = 0;
    while (sourceOffset < value.byteLength) {
      const count = Math.min(
        CACHE_CHUNK_BYTES - offset,
        value.byteLength - sourceOffset,
      );
      buffer.set(value.subarray(sourceOffset, sourceOffset + count), offset);
      sourceOffset += count;
      offset += count;
      received += count;

      if (offset === CACHE_CHUNK_BYTES) {
        await writeChunk(buffer);
        buffer = new Uint8Array(CACHE_CHUNK_BYTES);
        offset = 0;
      }
    }
  };

  if (response.body) {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      touchWatchdog();
      await appendBytes(value);
    }
  } else {
    await appendBytes(new Uint8Array(await response.arrayBuffer()));
  }

  if (offset > 0) {
    await writeChunk(buffer.slice(0, offset));
  }

  if (total !== null && received !== total) {
    throw new Error(`Cache branch ended at ${received} bytes, expected ${total}.`);
  }

  await cache.put(
    cacheChunkMetadataRequest(url),
    new Response(
      JSON.stringify({
        version: 1,
        bytes: received,
        chunkSize: CACHE_CHUNK_BYTES,
        chunks: chunks.length,
        chunkBytes: chunks,
      }),
      { headers: { "content-type": "application/json" } },
    ),
  );
}

function cancelCacheWriteBranch(controller: AbortController, response: Response | null): void {
  try {
    controller.abort();
  } catch {
    // Best effort: this branch is independent from the model bytes already read.
  }

  if (response?.body && !response.body.locked) {
    response.body.cancel().catch(() => {});
  }
}

let cacheWriteIssueLogged = false;

function logCacheWriteIssueOnce(error: unknown): void {
  if (cacheWriteIssueLogged) {
    return;
  }

  cacheWriteIssueLogged = true;
  console.info("Cache API store failed; continuing without cached model.", error);
}

async function assertSha256(
  bytes: Uint8Array,
  expectedHex: string,
  label: string,
): Promise<void> {
  if (!crypto.subtle) {
    throw new Error(`Cannot verify ${label}: WebCrypto digest is unavailable.`);
  }

  const digest = await crypto.subtle.digest("SHA-256", arrayBufferFromBytes(bytes));
  const actual = hex(new Uint8Array(digest));
  if (actual !== expectedHex.toLowerCase()) {
    throw new Error(`SHA-256 mismatch for ${label}: ${actual} !== ${expectedHex}.`);
  }
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function arrayBufferFromBytes(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
