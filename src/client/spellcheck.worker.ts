// Pure client-side spellcheck worker — a standard browser Web Worker (NOT a
// Cloudflare Worker). It builds the ~580k-form engine and answers lookups off
// the main thread, so downloading + parsing + indexing never freezes the UI.
// The wordlist/bigram files are stored with the Cache API, so they download once
// and are reused every session (works offline thereafter), just like the model.
import { SpellcheckEngine } from "./spellcheck";
import type { SpellcheckContext, SpellcheckSuggestion } from "./spellcheck";

const WORDLIST_URL = "/spellcheck-lt.txt";
const BIGRAMS_URL = "/spellcheck-bigrams.txt";
// Bump this suffix whenever the shipped wordlist/bigrams change so cached copies
// are refreshed (stale caches from older suffixes are pruned on startup).
const CACHE_NAME = "spellcheck-assets-v1";

type WorkerRequest = {
  id: number;
  words: Array<{ word: string; prev?: string; next?: string }>;
};

type WorkerResponse =
  | { id: number; results: SpellcheckSuggestion[] }
  | { id: number; error: string };

// Minimal typing for the dedicated-worker global so this file compiles under the
// project's DOM lib without pulling in the conflicting WebWorker lib.
type WorkerScope = {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse): void;
  caches?: CacheStorage;
};

const scope = self as unknown as WorkerScope;

let enginePromise: Promise<SpellcheckEngine> | null = null;
let bigramsPromise: Promise<void> | null = null;

void pruneStaleCaches();

async function cachedText(url: string): Promise<string> {
  const cacheStorage = scope.caches;
  if (cacheStorage) {
    try {
      const cache = await cacheStorage.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        return hit.text();
      }
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${url}: ${response.status}`);
      }
      await cache.put(url, response.clone());
      return response.text();
    } catch {
      // Cache API unavailable/denied (some private modes) → fall through to fetch.
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.text();
}

function toLines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function loadEngine(): Promise<SpellcheckEngine> {
  enginePromise ??= cachedText(WORDLIST_URL).then(
    (text) => new SpellcheckEngine(toLines(text)),
  );
  return enginePromise;
}

function ensureBigrams(engine: SpellcheckEngine): Promise<void> {
  bigramsPromise ??= cachedText(BIGRAMS_URL)
    .then((text) => {
      engine.setBigrams(toLines(text));
    })
    .catch(() => {
      // Context ranking simply degrades to a no-op if the table can't load.
    });
  return bigramsPromise;
}

async function pruneStaleCaches(): Promise<void> {
  const cacheStorage = scope.caches;
  if (!cacheStorage) {
    return;
  }
  try {
    const keys = await cacheStorage.keys();
    await Promise.all(
      keys
        .filter((key) => key.startsWith("spellcheck-assets-") && key !== CACHE_NAME)
        .map((key) => cacheStorage.delete(key)),
    );
  } catch {
    // Best-effort cleanup only.
  }
}

scope.onmessage = (event) => {
  const { id, words } = event.data;
  void handleRequest(id, words);
};

async function handleRequest(
  id: number,
  words: WorkerRequest["words"],
): Promise<void> {
  try {
    const engine = await loadEngine();
    if (words.some((word) => word.prev || word.next)) {
      await ensureBigrams(engine);
    }

    const results = words.map((word) => {
      const context: SpellcheckContext | undefined =
        word.prev || word.next ? { prev: word.prev, next: word.next } : undefined;
      return engine.suggest(word.word, context);
    });

    scope.postMessage({ id, results });
  } catch (error) {
    scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
}
