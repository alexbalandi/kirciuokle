// Pure client-side spellcheck worker — a standard browser Web Worker (NOT a
// Cloudflare Worker). It builds the correction engine + a real hunspell instance
// and answers lookups off the main thread, so downloading + parsing + indexing
// never freezes the UI. The wordlist/bigram/dictionary files are stored with the
// Cache API, so they download once and are reused every session (works offline
// thereafter), just like the model.
// Import the CJS build explicitly (see vite.config.ts): the ESM build is broken
// under bundlers. Types come from the package's declaration via the ambient shim
// in hunspell-asm-cjs.d.ts.
import { loadModule } from "hunspell-asm/dist/cjs/index.js";
import { SpellcheckEngine } from "./spellcheck";
import type { SpellcheckContext, SpellcheckSuggestion } from "./spellcheck";

const WORDLIST_URL = "/spellcheck-lt.txt";
const BIGRAMS_URL = "/spellcheck-bigrams.txt";
// Comprehensive accept dictionary (BSD-3 Lithuanian hunspell, ispell-lt): lemmas +
// affix rules, applied by real hunspell (compiled to wasm). This is authoritative
// morphology — every valid inflected form is recognised, so real text isn't
// false-flagged the way a finite corpus wordlist would be.
const HUNSPELL_AFF_URL = "/lt.aff";
const HUNSPELL_DIC_URL = "/lt.dic";
// Bump this suffix whenever the shipped assets change so cached copies are refreshed
// (stale caches from older suffixes are pruned on startup).
const CACHE_NAME = "spellcheck-assets-v6";

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

function postDiag(diag: unknown): void {
  (scope.postMessage as (m: unknown) => void)({ diag });
}

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

async function buildHunspell(
  aff: string,
  dic: string,
): Promise<(word: string) => boolean> {
  const factory = await loadModule();
  const encoder = new TextEncoder();
  const affPath = factory.mountBuffer(encoder.encode(aff), "lt.aff");
  const dicPath = factory.mountBuffer(encoder.encode(dic), "lt.dic");
  const hunspell = factory.create(affPath, dicPath);
  return (word) => hunspell.spell(word);
}

function loadEngine(): Promise<SpellcheckEngine> {
  enginePromise ??= (async () => {
    const [wordlist, aff, dic] = await Promise.all([
      cachedText(WORDLIST_URL),
      cachedText(HUNSPELL_AFF_URL).catch(() => ""),
      cachedText(HUNSPELL_DIC_URL).catch(() => ""),
    ]);
    const engine = new SpellcheckEngine(toLines(wordlist));
    // The hunspell dictionary is the authoritative accept check; if it fails to
    // load/build, the engine falls back to the wordlist's own `valid` set.
    if (aff && dic) {
      try {
        const spell = await buildHunspell(aff, dic);
        engine.setAcceptPredicate(spell);
        postDiag({ hunspell: "active", affLen: aff.length, dicLen: dic.length });
      } catch (e) {
        postDiag({
          hunspell: "failed",
          error: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
          stack: e instanceof Error ? (e.stack || "").split("\n").slice(0, 5) : [],
        });
      }
    } else {
      postDiag({ hunspell: "no-dict", affLen: aff.length, dicLen: dic.length });
    }
    return engine;
  })();
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
