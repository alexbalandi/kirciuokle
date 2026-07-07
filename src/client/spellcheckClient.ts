// Main-thread client for the spellcheck Web Worker. Batches all of a text's
// words into one message so the worker answers them in a single pass. Falls back
// to running the engine in-thread if the browser can't spawn the worker, so
// spellcheck always works — the worker is purely a "don't block the UI"
// optimisation, not a correctness dependency.
import {
  loadSpellcheckEngine,
  suggest as suggestInThread,
  type SpellcheckSuggestion,
} from "./spellcheck";

export type SpellcheckWord = { word: string; prev?: string; next?: string };

type WorkerResponse =
  | { id: number; results: SpellcheckSuggestion[] }
  | { id: number; error: string };

const UNKNOWN: SpellcheckSuggestion = { status: "unknown", candidates: [] };

// undefined = not yet attempted, null = unavailable (use in-thread fallback).
let worker: Worker | null | undefined;
let nextId = 0;
const pending = new Map<
  number,
  { resolve: (results: SpellcheckSuggestion[]) => void; count: number }
>();

function getWorker(): Worker | null {
  if (worker !== undefined) {
    return worker;
  }

  try {
    const instance = new Worker(new URL("./spellcheck.worker.ts", import.meta.url), {
      type: "module",
    });
    instance.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      const entry = pending.get(message.id);
      if (!entry) {
        return;
      }
      pending.delete(message.id);
      entry.resolve(
        "results" in message
          ? message.results
          : Array.from({ length: entry.count }, () => UNKNOWN),
      );
    };
    instance.onerror = () => {
      // Construction/load failure: drop to in-thread for all future calls and
      // resolve anything in flight so callers never hang.
      worker = null;
      for (const [id, entry] of pending) {
        entry.resolve(Array.from({ length: entry.count }, () => UNKNOWN));
        pending.delete(id);
      }
    };
    worker = instance;
  } catch {
    worker = null;
  }

  return worker;
}

async function suggestInThreadBatch(
  words: SpellcheckWord[],
): Promise<SpellcheckSuggestion[]> {
  await loadSpellcheckEngine();
  return Promise.all(
    words.map((word) =>
      suggestInThread(
        word.word,
        word.prev || word.next ? { prev: word.prev, next: word.next } : undefined,
      ),
    ),
  );
}

export async function suggestBatch(
  words: SpellcheckWord[],
): Promise<SpellcheckSuggestion[]> {
  if (words.length === 0) {
    return [];
  }

  const instance = getWorker();
  if (!instance) {
    return suggestInThreadBatch(words);
  }

  const id = ++nextId;
  return new Promise<SpellcheckSuggestion[]>((resolve) => {
    pending.set(id, { resolve, count: words.length });
    instance.postMessage({ id, words });
  });
}
