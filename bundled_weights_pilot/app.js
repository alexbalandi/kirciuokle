import * as ort from "./model/runtime/ort.min.mjs";
import {
  AutoTokenizer,
  env as transformersEnv,
} from "./model/runtime/transformers.min.js";
import {
  detectLang,
  LANGS,
  morphologySegments,
  PILOT_UI,
  UI,
} from "./i18n.js";

const CACHE_NAME = "bundled-weights-pilot-v1";
const MODEL_DIR = "./model/";
const RUNTIME_DIR = new URL("./model/runtime/", import.meta.url).href;
const MANIFEST_URL = `${MODEL_DIR}manifest.json`;
const META_URL = `${MODEL_DIR}joint.meta.json`;
const LABEL_BRIDGE_URL = `${MODEL_DIR}label_bridge.json`;
const MAX_SUBWORDS = 128;
const MIN_TOKEN_BUDGET = 256;
const MAX_TOKEN_BUDGET = 8192;
const DEFAULT_TOKEN_BUDGET = 2048;
const MAX_POPOVER_ROWS = 5;
const POS_PROB_CUT = 0.1;
const RESOLVED_PROB = 0.9;
const CACHE_CHUNK_BYTES = 16 * 1024 * 1024;
const CACHE_HEADROOM_MULTIPLIER = 1.2;
const CACHE_WRITE_WATCHDOG_MS = 20_000;
const SESSION_PROXY_WATCHDOG_MS = 45_000;
const SESSION_PROXY_PROGRESS_WATCHDOG_MS = 180_000;
const WASM_PAGE_BYTES = 64 * 1024;
const WASM_32BIT_CEILING_BYTES = 4 * 1024 * 1024 * 1024;
const WASM_HIGH_WATER_RATIO = 0.75;
const MEMORY_TEST_MAX_MB_PARAM = "pilotWasmMaxMb";
const MEMORY_TEST_ALLOC_FAIL_PARAM = "pilotForceAllocationFailure";
const WASM_RUNTIME_RESOURCE_RE = /ort-wasm.*\.(?:mjs|wasm)(?:[?#]|$)/u;
const TOKEN_RE = /[\p{L}\p{M}\p{N}_]+|[^\p{L}\p{M}\p{N}_\s]/gu;
const SENTENCE_END_RE = /[.!?…]+(?:["')\]]+)?\s+(?=[A-ZĄČĘĖĮŠŲŪŽ])/gu;
const STRESS_MARKS = new Set(["\u0300", "\u0301", "\u0303"]);
const COMBINING_DOT_ABOVE = "\u0307";
const I_DOT_BASES = new Set(["i", "I", "j", "J"]);
const GRAVE = "\u0300";
const ACUTE = "\u0301";
const TILDE = "\u0303";
const VOWELS = new Set(Array.from("aeiouyąęėįūų"));
const LONG_VOWELS = new Set(Array.from("yąęėįūų"));
const SONORANTS = new Set(Array.from("lmnr"));
const PURE_DIPHTHONGS = new Set(["au", "ai", "ei", "ui", "uo", "ie"]);
const VOWEL_BASES = new Set(Array.from("aeiouy"));
const SCORING_SLOTS = [
  "case",
  "gender",
  "number",
  "tense",
  "person",
  "voice",
  "degree",
];
const VLKK_PRIMER_URL =
  "https://www.vlkk.lt/aktualiausios-temos/tartis-ir-kirciavimas";
const FOCUSABLE_SELECTOR =
  'a[href], button:not(:disabled), textarea:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])';
const PRIMER_MIXED_WORDS = getPrimerMixedWords();
const PRIMER_PAIR_WORDS = getPrimerPairWords();
const SESSION_UI = {
  en: {
    sessionWorker: "Initializing model (worker)…",
    sessionFallback: "Initializing model (fallback)…",
    sessionMain: "Initializing model (main thread)…",
    modeLabel: "mode",
    workerMode: "worker",
    mainMode: "main thread",
  },
  lt: {
    sessionWorker: "Modelis inicijuojamas (darbinė gija)…",
    sessionFallback: "Modelis inicijuojamas (atsarginis režimas)…",
    sessionMain: "Modelis inicijuojamas (pagrindinė gija)…",
    modeLabel: "režimas",
    workerMode: "darbinė gija",
    mainMode: "pagrindinė gija",
  },
  ru: {
    sessionWorker: "Инициализация модели (воркер)…",
    sessionFallback: "Инициализация модели (резервный режим)…",
    sessionMain: "Инициализация модели (основной поток)…",
    modeLabel: "режим",
    workerMode: "воркер",
    mainMode: "основной поток",
  },
};
const wasmMemoryTracker = installWasmMemoryTracker();
const memoryTestOverrides = readMemoryTestOverrides();
const cacheWriteClosedUrls = new Set();
let cacheWriteIssueLogged = false;

const dom = {
  metaDescription: document.querySelector('meta[name="description"]'),
  appTitle: getElement("app-title"),
  pilotEyebrow: getElement("pilot-eyebrow"),
  heroTagline: getElement("hero-tagline"),
  pilotSubtitle: getElement("pilot-subtitle"),
  languageButtons: Array.from(
    getElement("language-switcher").querySelectorAll("button[data-lang]"),
  ),
  form: getElement("accent-form"),
  inputLabel: getElement("input-label"),
  textarea: getElement("source-text"),
  tokenBudgetLabel: getElement("token-budget-label"),
  tokenBudget: getElement("token-budget"),
  button: getElement("accent-button"),
  copyButton: getElement("copy-button"),
  counter: getElement("char-counter"),
  modelStatus: getElement("model-status"),
  runStatus: getElement("run-status"),
  memoryStatus: getElement("memory-status"),
  progressBar: getElement("progress-bar"),
  resultHeading: getElement("result-heading"),
  result: getElement("result-output"),
  legend: getElement("legend"),
  legendLabel: getElement("legend-label"),
  legendResolved: getElement("legend-resolved"),
  legendAmbiguous: getElement("legend-ambiguous"),
  legendUnknown: getElement("legend-unknown"),
  primerLink: getElement("primer-link"),
  primerBackdrop: getElement("primer-backdrop"),
  primerDialog: getElement("primer-dialog"),
  primerClose: getElement("primer-close"),
  primerTitle: getElement("primer-title"),
  primerIntro: getElement("primer-intro"),
  primerGraveName: getElement("primer-grave-name"),
  primerGraveDesc: getElement("primer-grave-desc"),
  primerGraveEx: getElement("primer-grave-ex"),
  primerAcuteName: getElement("primer-acute-name"),
  primerAcuteDesc: getElement("primer-acute-desc"),
  primerAcuteEx: getElement("primer-acute-ex"),
  primerTildeName: getElement("primer-tilde-name"),
  primerTildeDesc: getElement("primer-tilde-desc"),
  primerTildeEx: getElement("primer-tilde-ex"),
  primerMixed: getElement("primer-mixed"),
  primerPair: getElement("primer-pair"),
  primerMore: getElement("primer-more"),
  popover: getElement("pos-popover"),
  siteFooter: getElement("site-footer"),
};

let runtime = null;
let activeRun = 0;
let lastPlainText = "";
let lang = detectLang();
let copied = false;
let copyResetTimer = 0;
let modelStatusState = { type: "loading" };
let runStatusState = { type: "ready" };
let memoryStatusState = readMemoryStatus();
let runtimeExecutionMode = null;

boot();

function boot() {
  setLanguage(lang, { persist: false });
  dom.textarea.value =
    "Lietuva yra graži šalis. Šiandien Vilniuje lyja. Mokslininkai tiria kalbos modelius.";
  updateCounter();
  dom.textarea.addEventListener("input", updateCounter);
  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    runAccentuation();
  });
  dom.copyButton.addEventListener("click", async () => {
    if (!lastPlainText) {
      return;
    }
    await navigator.clipboard.writeText(lastPlainText);
    copied = true;
    renderUi();
    setRunStatusState({ type: "copied" });
    window.clearTimeout(copyResetTimer);
    copyResetTimer = window.setTimeout(() => {
      copied = false;
      renderUi();
    }, 1400);
  });
  dom.languageButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const next = parseLang(button.dataset.lang);
      if (next) {
        setLanguage(next);
      }
    });
  });
  dom.primerLink.addEventListener("click", openPrimer);
  dom.primerClose.addEventListener("click", closePrimer);
  dom.primerBackdrop.addEventListener("click", (event) => {
    if (event.target === dom.primerBackdrop) {
      closePrimer();
    }
  });
  dom.primerDialog.addEventListener("keydown", (event) => {
    if (event.key === "Tab") {
      trapPrimerFocus(event);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    if (isPrimerOpen()) {
      event.preventDefault();
      closePrimer();
      return;
    }
    hidePopover();
  });
  document.addEventListener("click", (event) => {
    if (
      event.target instanceof Element &&
      (event.target.closest(".pos-popover") || event.target.closest(".token"))
    ) {
      return;
    }
    hidePopover();
  });
  window.addEventListener("resize", hidePopover);
  updateMemoryStatus();
  window.setInterval(updateMemoryStatus, 2000);
  loadRuntime().catch((error) => {
    console.error(error);
    setModelStatusState({ type: "loadFailed", message: error.message });
  });
}

function installWasmMemoryTracker() {
  const NativeMemory = WebAssembly.Memory;
  const memories = new Map();

  const remember = (memory, maximumPages = null) => {
    if (
      !memory ||
      !memory.buffer ||
      typeof memory.buffer.byteLength !== "number"
    ) {
      return;
    }
    const current = memories.get(memory) || {};
    const maxBytes = Number.isFinite(maximumPages)
      ? Number(maximumPages) * WASM_PAGE_BYTES
      : current.maxBytes;
    memories.set(memory, { maxBytes });
  };

  WebAssembly.Memory = new Proxy(NativeMemory, {
    construct(target, args, newTarget) {
      const descriptor = args[0] || {};
      const memory = Reflect.construct(target, args, newTarget);
      remember(memory, descriptor.maximum);
      return memory;
    },
  });

  const captureExports = (exports) => {
    if (!exports) {
      return;
    }
    for (const value of Object.values(exports)) {
      remember(value);
    }
  };

  const nativeInstantiate = WebAssembly.instantiate.bind(WebAssembly);
  WebAssembly.instantiate = async (...args) => {
    const result = await nativeInstantiate(...args);
    captureExports(result?.instance?.exports || result?.exports);
    return result;
  };

  if (WebAssembly.instantiateStreaming) {
    const nativeInstantiateStreaming =
      WebAssembly.instantiateStreaming.bind(WebAssembly);
    WebAssembly.instantiateStreaming = async (...args) => {
      const result = await nativeInstantiateStreaming(...args);
      captureExports(result?.instance?.exports || result?.exports);
      return result;
    };
  }

  return {
    read() {
      const entries = Array.from(memories, ([memory, info]) => {
        const bytes = memory.buffer.byteLength;
        return {
          bytes,
          maxBytes:
            info.maxBytes && info.maxBytes >= bytes
              ? info.maxBytes
              : WASM_32BIT_CEILING_BYTES,
        };
      });
      const wasmBytes = entries.reduce((sum, item) => sum + item.bytes, 0);
      const wasmMaxBytes = entries.reduce(
        (sum, item) => sum + item.maxBytes,
        0,
      );
      return {
        wasmBytes,
        wasmMaxBytes: wasmMaxBytes || WASM_32BIT_CEILING_BYTES,
        wasmMemoryCount: entries.length,
      };
    },
  };
}

function readMemoryTestOverrides() {
  const params = new URLSearchParams(window.location.search);
  const forcedWasmMaxMb = Number(params.get(MEMORY_TEST_MAX_MB_PARAM));
  return {
    wasmMaxBytes:
      Number.isFinite(forcedWasmMaxMb) && forcedWasmMaxMb > 0
        ? forcedWasmMaxMb * 1024 * 1024
        : null,
    forceAllocationFailure:
      params.get(MEMORY_TEST_ALLOC_FAIL_PARAM) === "1",
  };
}

async function loadRuntime() {
  setModelStatusState({ type: "metadata" });
  const [manifest, meta, bridge] = await Promise.all([
    fetchOptionalJson(MANIFEST_URL),
    fetchJson(META_URL),
    fetchJson(LABEL_BRIDGE_URL),
  ]);
  const modelFile = manifest?.default_model || meta.int8_onnx || "joint.int8.onnx";
  const modelUrl = `${MODEL_DIR}${modelFile}`;
  const expectedBytes =
    manifest?.models?.[modelFile]?.bytes ??
    manifest?.model_bytes ??
    (await headContentLength(modelUrl));

  transformersEnv.allowLocalModels = true;
  transformersEnv.allowRemoteModels = false;
  transformersEnv.localModelPath = "";

  const threads = crossOriginIsolated
    ? Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)))
    : 1;
  ort.env.wasm.wasmPaths = RUNTIME_DIR;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = true;
  configureTransformersOnnxRuntime();

  const labelBridgeCache = buildLabelBridgeCache(bridge, meta.labels);
  const cacheState = await cacheHit(modelUrl);
  setModelStatusState({
    type: "modelInfo",
    expectedBytes,
    cacheState,
    threads,
  });

  const [tokenizer, modelLoad] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_DIR, { local_files_only: true }),
    fetchWithCache(modelUrl, expectedBytes),
  ]);
  let modelBytes = modelLoad.bytes;
  const modelByteLength = modelBytes.byteLength;
  const loadMemoryBeforeSession = await measureBeforeSessionCreate();
  const sessionInfo = await createSessionWithProgressiveProxy({
    modelBytes,
    modelUrl,
    expectedBytes,
    modelByteLength,
    threads,
  });
  modelBytes = sessionInfo.modelBytes;
  const session = sessionInfo.session;
  runtimeExecutionMode = sessionInfo.mode;
  const loadMemoryBeforeRelease = updateMemoryStatus();
  modelLoad.bytes = null;
  modelBytes = null;
  const loadMemoryAfterRelease = await measureAfterLoadBufferRelease();
  modelLoad.cacheWriteState?.start?.();
  window.__pilotLoadMemory = {
    beforeSession: loadMemoryBeforeSession,
    beforeRelease: loadMemoryBeforeRelease,
    afterRelease: loadMemoryAfterRelease,
  };
  runtime = {
    session,
    tokenizer,
    meta,
    labels: meta.labels,
    labelBridgeCache,
    charVocab: meta.char_vocab,
    marks: meta.marks,
    maxChars: Number(meta.max_chars || 30),
    padId: Number(tokenizer.pad_token_id ?? 1),
    bosId: Number(tokenizer.bos_token_id ?? 0),
    eosId: Number(tokenizer.eos_token_id ?? 2),
    threads: sessionInfo.threads,
    modelFile,
    modelBytes: modelByteLength,
    executionMode: sessionInfo.mode,
    cacheStatus: modelLoad.cacheWriteState?.status ?? modelLoad.cacheStatus,
  };
  setModelStatusState({
    type: "readyModel",
    modelFile,
    bytes: modelByteLength,
    cacheStatus: modelLoad.cacheWriteState?.status ?? modelLoad.cacheStatus,
    threads: sessionInfo.threads,
    executionMode: sessionInfo.mode,
  });
  watchCacheWriteStatus(modelLoad.cacheWriteState, {
    modelFile,
    bytes: modelByteLength,
    threads: sessionInfo.threads,
    executionMode: sessionInfo.mode,
  });
  dom.button.disabled = false;
  window.__pilotRuntimeReady = true;
  window.__pilotLabelBridgeSize = labelBridgeCache.size;
  window.__pilotRuntimeConfig = {
    ortVersion: ort.env.versions?.web ?? null,
    wasmPaths: ort.env.wasm.wasmPaths,
    proxy: ort.env.wasm.proxy,
    mode: sessionInfo.mode,
    executionMode: sessionInfo.mode,
    proxyFallback: sessionInfo.mode !== "worker",
    wasmRuntimeFetchObserved: sessionInfo.diagnostics.wasmRuntimeFetchObserved,
    sessionCreate: sessionInfo.diagnostics,
    threads: sessionInfo.threads,
  };
}

async function createSessionWithProgressiveProxy({
  modelBytes,
  modelUrl,
  expectedBytes,
  modelByteLength,
  threads,
}) {
  const diagnostics = makeSessionDiagnostics();
  const runtimeFetchMonitor = installWasmRuntimeFetchMonitor(RUNTIME_DIR, diagnostics);
  let currentModelBytes = modelBytes;

  try {
    setModelStatusState({
      type: "session",
      bytes: modelByteLength,
      mode: "worker",
    });
    try {
      const session = await createInferenceSessionAttempt(currentModelBytes, {
        diagnostics,
        mode: "worker",
        proxy: true,
        useWatchdog: true,
        runtimeFetchMonitor,
      });
      completeSessionDiagnostics(diagnostics, "worker");
      return {
        session,
        mode: "worker",
        diagnostics,
        modelBytes: currentModelBytes,
        threads,
      };
    } catch (error) {
      warnProxySession(
        diagnostics,
        "ORT proxy worker session failed; retrying on the main thread.",
        error,
      );
      ort.env.wasm.proxy = false;
      ort.env.wasm.numThreads = 1;
      setModelStatusState({
        type: "session",
        bytes: modelByteLength,
        mode: "fallback",
      });

      if (currentModelBytes.byteLength !== modelByteLength) {
        warnProxySession(
          diagnostics,
          "ORT proxy attempt detached the model buffer before fallback; refetching model bytes.",
          {
            expectedBytes: modelByteLength,
            actualBytes: currentModelBytes.byteLength,
          },
        );
        currentModelBytes = await refetchModelBytesForFallback(modelUrl, expectedBytes);
      }

      const session = await createInferenceSessionAttempt(currentModelBytes, {
        diagnostics,
        mode: "main",
        proxy: false,
        useWatchdog: false,
        runtimeFetchMonitor,
      });
      completeSessionDiagnostics(diagnostics, "main");
      return {
        session,
        mode: "main",
        diagnostics,
        modelBytes: currentModelBytes,
        threads: ort.env.wasm.numThreads,
      };
    }
  } finally {
    runtimeFetchMonitor.restore();
  }
}

async function createInferenceSessionAttempt(
  modelBytes,
  { diagnostics, mode, proxy, useWatchdog, runtimeFetchMonitor },
) {
  ort.env.wasm.proxy = proxy;
  const attempt = makeSessionAttempt(mode, proxy);
  diagnostics.attempts.push(attempt);
  publishSessionDiagnostics(diagnostics);
  const workerInstrumentation =
    proxy ? installProxyWorkerInstrumentation(attempt, diagnostics) : null;
  const watchdog = useWatchdog
    ? startSessionWatchdog(attempt, diagnostics, runtimeFetchMonitor)
    : null;
  let abandoned = false;
  let settled = false;

  try {
    if (proxy) {
      probeProxyWorkerConstruction();
    }
    const createPromise = ort.InferenceSession.create(modelBytes, makeSessionOptions());
    createPromise.then(
      (session) => {
        settled = true;
        if (!abandoned) {
          return;
        }
        attempt.lateResolution = true;
        publishSessionDiagnostics(diagnostics);
        warnProxySession(
          diagnostics,
          "Abandoned ORT proxy session resolved after fallback started; releasing it.",
          { mode },
        );
        releaseOrtSession(session);
      },
      (error) => {
        settled = true;
        if (!abandoned) {
          return;
        }
        warnProxySession(
          diagnostics,
          "Abandoned ORT proxy session rejected after fallback started.",
          error,
        );
      },
    );
    const session = watchdog
      ? await Promise.race([createPromise, watchdog.promise])
      : await createPromise;
    attempt.status = "resolved";
    attempt.elapsedMs = performance.now() - attempt.startedAt;
    return session;
  } catch (error) {
    attempt.status = error?.pilotWatchdog ? "watchdog" : "rejected";
    attempt.error = plainError(error);
    attempt.elapsedMs = performance.now() - attempt.startedAt;
    if (error?.pilotWatchdog) {
      abandoned = true;
      terminateAttemptWorkers(workerInstrumentation, attempt, diagnostics);
    }
    publishSessionDiagnostics(diagnostics);
    throw error;
  } finally {
    if (!settled && watchdog?.timedOut) {
      abandoned = true;
    }
    watchdog?.clear();
    workerInstrumentation?.restore();
    publishSessionDiagnostics(diagnostics);
  }
}

function probeProxyWorkerConstruction() {
  const blob = new Blob([""], { type: "text/javascript" });
  const url = URL.createObjectURL(blob);
  let worker = null;
  try {
    worker = new Worker(url, { type: "module", name: "ort-proxy-preflight" });
  } finally {
    worker?.terminate();
    URL.revokeObjectURL(url);
  }
}

function makeSessionOptions() {
  return {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  };
}

function makeSessionDiagnostics() {
  const diagnostics = {
    startedAt: performance.now(),
    completedAt: null,
    mode: null,
    wasmRuntimeFetchObserved: false,
    wasmRuntimeFetchUrls: [],
    attempts: [],
    warnings: [],
  };
  publishSessionDiagnostics(diagnostics);
  return diagnostics;
}

function makeSessionAttempt(mode, proxy) {
  return {
    mode,
    proxy,
    status: "running",
    startedAt: performance.now(),
    elapsedMs: null,
    watchdogMs: proxy ? SESSION_PROXY_WATCHDOG_MS : null,
    watchdogExtended: false,
    workerCount: 0,
    workerEvents: [],
    error: null,
    lateResolution: false,
  };
}

function completeSessionDiagnostics(diagnostics, mode) {
  diagnostics.completedAt = performance.now();
  diagnostics.mode = mode;
  publishSessionDiagnostics(diagnostics);
}

function publishSessionDiagnostics(diagnostics) {
  window.__pilotSessionCreate = {
    startedAt: diagnostics.startedAt,
    completedAt: diagnostics.completedAt,
    elapsedMs:
      diagnostics.completedAt == null
        ? performance.now() - diagnostics.startedAt
        : diagnostics.completedAt - diagnostics.startedAt,
    mode: diagnostics.mode,
    wasmRuntimeFetchObserved: diagnostics.wasmRuntimeFetchObserved,
    wasmRuntimeFetchUrls: diagnostics.wasmRuntimeFetchUrls.slice(),
    attempts: diagnostics.attempts.map((attempt) => ({
      mode: attempt.mode,
      proxy: attempt.proxy,
      status: attempt.status,
      elapsedMs:
        attempt.elapsedMs == null ? performance.now() - attempt.startedAt : attempt.elapsedMs,
      watchdogMs: attempt.watchdogMs,
      watchdogExtended: attempt.watchdogExtended,
      workerCount: attempt.workerCount,
      workerEvents: attempt.workerEvents.slice(),
      error: attempt.error,
      lateResolution: attempt.lateResolution,
    })),
    warnings: diagnostics.warnings.slice(),
  };
}

function warnProxySession(diagnostics, message, detail) {
  const warning = {
    message,
    detail: plainError(detail),
    atMs: performance.now() - diagnostics.startedAt,
  };
  diagnostics.warnings.push(warning);
  publishSessionDiagnostics(diagnostics);
  console.warn(message, detail);
}

function plainError(value) {
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack,
    };
  }
  if (value && typeof value === "object") {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return String(value);
    }
  }
  return value == null ? null : String(value);
}

function startSessionWatchdog(attempt, diagnostics, runtimeFetchMonitor) {
  let timeoutId = 0;
  let rejectWatchdog = null;
  let cleared = false;
  const watchdog = {
    timedOut: false,
    promise: new Promise((_, reject) => {
      rejectWatchdog = reject;
    }),
    clear() {
      cleared = true;
      window.clearTimeout(timeoutId);
      unsubscribe();
    },
  };

  const extendForRuntimeFetch = () => {
    if (attempt.watchdogMs >= SESSION_PROXY_PROGRESS_WATCHDOG_MS) {
      return;
    }
    attempt.watchdogMs = SESSION_PROXY_PROGRESS_WATCHDOG_MS;
    attempt.watchdogExtended = true;
    publishSessionDiagnostics(diagnostics);
    schedule();
  };

  const unsubscribe = runtimeFetchMonitor.subscribe(extendForRuntimeFetch);
  if (runtimeFetchMonitor.seen) {
    extendForRuntimeFetch();
  }

  function schedule() {
    if (cleared) {
      return;
    }
    window.clearTimeout(timeoutId);
    const elapsed = performance.now() - attempt.startedAt;
    const remaining = Math.max(0, attempt.watchdogMs - elapsed);
    timeoutId = window.setTimeout(() => {
      watchdog.timedOut = true;
      const error = new Error(
        `ORT proxy worker session create timed out after ${attempt.watchdogMs}ms.`,
      );
      error.pilotWatchdog = true;
      warnProxySession(diagnostics, error.message, {
        mode: attempt.mode,
        wasmRuntimeFetchObserved: diagnostics.wasmRuntimeFetchObserved,
        wasmRuntimeFetchUrls: diagnostics.wasmRuntimeFetchUrls,
      });
      rejectWatchdog(error);
    }, remaining);
  }

  schedule();
  return watchdog;
}

function installWasmRuntimeFetchMonitor(prefix, diagnostics) {
  const listeners = new Set();
  let restored = false;
  const nativeFetch = window.fetch;
  const boundFetch = nativeFetch.bind(window);
  let observer = null;

  const note = (url, source) => {
    if (!isWasmRuntimeResource(url, prefix)) {
      return;
    }
    const entry = `${source}:${url}`;
    if (!diagnostics.wasmRuntimeFetchUrls.includes(entry)) {
      diagnostics.wasmRuntimeFetchUrls.push(entry);
    }
    if (!diagnostics.wasmRuntimeFetchObserved) {
      diagnostics.wasmRuntimeFetchObserved = true;
    }
    publishSessionDiagnostics(diagnostics);
    listeners.forEach((listener) => listener(url));
  };

  window.fetch = (...args) => {
    note(fetchRequestUrl(args[0]), "fetch");
    return boundFetch(...args);
  };
  const wrappedFetch = window.fetch;

  if (typeof PerformanceObserver === "function") {
    try {
      observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          note(entry.name, "resource");
        }
      });
      observer.observe({ type: "resource", buffered: true });
    } catch (error) {
      warnProxySession(
        diagnostics,
        "Could not install WASM runtime PerformanceObserver; fetch hook remains active.",
        error,
      );
    }
  }

  for (const entry of performance.getEntriesByType?.("resource") || []) {
    note(entry.name, "resource");
  }

  return {
    get seen() {
      return diagnostics.wasmRuntimeFetchObserved;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    restore() {
      if (restored) {
        return;
      }
      restored = true;
      if (window.fetch === wrappedFetch) {
        window.fetch = nativeFetch;
      }
      observer?.disconnect();
    },
  };
}

function isWasmRuntimeResource(url, prefix) {
  if (!url || !String(url).startsWith(prefix)) {
    return false;
  }
  return WASM_RUNTIME_RESOURCE_RE.test(String(url));
}

function fetchRequestUrl(request) {
  if (typeof request === "string") {
    return new URL(request, location.href).href;
  }
  if (request instanceof URL) {
    return request.href;
  }
  return request?.url || "";
}

function installProxyWorkerInstrumentation(attempt, diagnostics) {
  const NativeWorker = window.Worker;
  if (typeof NativeWorker !== "function") {
    warnProxySession(diagnostics, "ORT proxy requested Worker, but Worker is unavailable.", {
      workerType: typeof NativeWorker,
    });
    return null;
  }

  const workers = new Set();
  const InstrumentedWorker = new Proxy(NativeWorker, {
    construct(target, args) {
      const workerUrl = fetchRequestUrl(args[0]);
      try {
        const worker = Reflect.construct(target, args);
        workers.add(worker);
        attempt.workerCount = workers.size;
        publishSessionDiagnostics(diagnostics);
        worker.addEventListener("error", (event) => {
          recordWorkerEvent(attempt, diagnostics, "error", workerUrl, event);
        });
        worker.addEventListener("messageerror", (event) => {
          recordWorkerEvent(attempt, diagnostics, "messageerror", workerUrl, event);
        });
        return worker;
      } catch (error) {
        attempt.workerEvents.push({
          type: "construct-error",
          url: workerUrl,
          message: error?.message || String(error),
        });
        attempt.workerCount = workers.size;
        publishSessionDiagnostics(diagnostics);
        warnProxySession(diagnostics, "ORT proxy Worker construction failed.", {
          url: workerUrl,
          error: plainError(error),
        });
        throw error;
      }
    },
  });

  window.Worker = InstrumentedWorker;
  return {
    terminateAll() {
      for (const worker of workers) {
        try {
          worker.terminate();
        } catch (error) {
          warnProxySession(diagnostics, "Failed to terminate abandoned ORT worker.", error);
        }
      }
    },
    restore() {
      if (window.Worker === InstrumentedWorker) {
        window.Worker = NativeWorker;
      }
    },
  };
}

function recordWorkerEvent(attempt, diagnostics, type, url, event) {
  const detail = {
    type,
    url,
    message: event?.message || "",
    filename: event?.filename || "",
    lineno: event?.lineno || 0,
    colno: event?.colno || 0,
  };
  attempt.workerEvents.push(detail);
  publishSessionDiagnostics(diagnostics);
  warnProxySession(diagnostics, `ORT proxy Worker ${type}.`, detail);
}

function terminateAttemptWorkers(workerInstrumentation, attempt, diagnostics) {
  if (!workerInstrumentation) {
    return;
  }
  workerInstrumentation.terminateAll();
  attempt.workerEvents.push({
    type: "terminated",
    url: "",
    message: "Watchdog fired; abandoned proxy workers were terminated.",
  });
  publishSessionDiagnostics(diagnostics);
}

function releaseOrtSession(session) {
  try {
    if (typeof session?.release === "function") {
      session.release();
      return;
    }
    if (typeof session?.dispose === "function") {
      session.dispose();
    }
  } catch (error) {
    console.warn("Failed to release abandoned ORT session.", error);
  }
}

async function refetchModelBytesForFallback(url, expectedBytes) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (expectedBytes && bytes.byteLength !== expectedBytes) {
    throw new Error(
      `Fallback model refetch returned ${bytes.byteLength} bytes, expected ${expectedBytes}.`,
    );
  }
  return bytes;
}

function configureTransformersOnnxRuntime() {
  const onnx = transformersEnv.backends?.onnx;
  if (!onnx?.wasm) {
    return;
  }
  onnx.wasm.wasmPaths = RUNTIME_DIR;
  onnx.wasm.proxy = true;
}

function watchCacheWriteStatus(state, readyInfo) {
  if (!state?.promise) {
    return;
  }
  state.promise.then((cacheStatus) => {
    if (runtime?.modelFile !== readyInfo.modelFile) {
      return;
    }
    runtime.cacheStatus = cacheStatus;
    if (
      modelStatusState.type === "readyModel" &&
      modelStatusState.modelFile === readyInfo.modelFile
    ) {
      setModelStatusState({
        type: "readyModel",
        modelFile: readyInfo.modelFile,
        bytes: readyInfo.bytes,
        cacheStatus,
        threads: readyInfo.threads,
        executionMode: readyInfo.executionMode,
      });
    }
  });
}

async function runAccentuation() {
  if (!runtime) {
    return;
  }
  const text = dom.textarea.value.normalize("NFC");
  const runId = ++activeRun;
  hidePopover();
  dom.button.disabled = true;
  dom.copyButton.disabled = true;
  lastPlainText = "";
  copied = false;
  renderUi();
  setProgress(0);

  if (!text.trim()) {
    setResultEmpty(UI[lang].resultEmpty);
    setRunStatusState({ type: "ready" });
    dom.button.disabled = false;
    return;
  }

  const sentences = splitSentences(text).map((sentence, index) =>
    prepareSentence(sentence, index),
  );
  renderPlaceholders(sentences);
  const baseTokenBudget = clampInt(
    dom.tokenBudget.value,
    MIN_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    DEFAULT_TOKEN_BUDGET,
  );
  let effectiveTokenBudget = adaptiveTokenBudget(baseTokenBudget);
  let batches = buildBatches(sentences, effectiveTokenBudget);
  const totalTokens = sentences.reduce((sum, sentence) => sum + sentence.tokens.length, 0);
  const decodedBySentence = Array(sentences.length);
  let renderedSentences = 0;
  let inferredTokens = 0;
  const startMs = performance.now();

  setRunStatusState({
    type: "running",
    sentences: sentences.length,
    batches: batches.length,
  });

  try {
    for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
      if (runId !== activeRun) {
        return;
      }
      const batch = batches[batchIndex];
      let decoded;
      try {
        decoded = await runBatch(batch, runtime);
      } catch (error) {
        if (isMemoryAllocationError(error)) {
          console.warn(error);
          updateMemoryStatus();
          setRunStatusState({ type: "memoryLimit" });
          window.__pilotLastRun = {
            sentences: sentences.length,
            batches: batches.length,
            tokens: inferredTokens,
            totalTokens,
            elapsedMs: performance.now() - startMs,
            tokensPerSecond: 0,
            renderedSentences,
            memoryLimitReached: true,
            baseTokenBudget,
            effectiveTokenBudget,
          };
          return;
        }
        throw error;
      }
      for (const sentenceResult of decoded) {
        const sentence = sentences[sentenceResult.index];
        decodedBySentence[sentence.index] = sentenceResult.tokens;
        renderSentence(sentence, sentenceResult.tokens);
        releaseSentenceInferenceInputs(sentence);
        renderedSentences += 1;
        inferredTokens += sentenceResult.tokens.filter((token) => token.predicted).length;
      }
      const elapsed = performance.now() - startMs;
      const tokensPerSecond = inferredTokens / Math.max(elapsed / 1000, 0.001);
      const memory = updateMemoryStatus();
      const nextTokenBudget = adaptiveTokenBudget(baseTokenBudget, memory);
      if (nextTokenBudget !== effectiveTokenBudget && batchIndex + 1 < batches.length) {
        const pending = batches.slice(batchIndex + 1).flat();
        const rebuilt = buildBatches(pending, nextTokenBudget);
        batches = batches.slice(0, batchIndex + 1).concat(rebuilt);
        effectiveTokenBudget = nextTokenBudget;
      }
      setProgress(renderedSentences / sentences.length);
      setRunStatusState({
        type: "batch",
        renderedSentences,
        sentences: sentences.length,
        batch: batchIndex + 1,
        batches: batches.length,
        tokensPerSecond,
      });
      await nextFrame();
    }

    const elapsedMs = performance.now() - startMs;
    const tokensPerSecond = inferredTokens / Math.max(elapsedMs / 1000, 0.001);
    lastPlainText = collectRenderedText(sentences, decodedBySentence);
    dom.copyButton.disabled = !lastPlainText;
    setProgress(1);
    setRunStatusState({
      type: "done",
      sentences: sentences.length,
      inferredTokens,
      totalTokens,
      tokensPerSecond,
      elapsedMs,
    });
    window.__pilotLastRun = {
      sentences: sentences.length,
      batches: batches.length,
      tokens: inferredTokens,
      totalTokens,
      elapsedMs,
      tokensPerSecond,
      renderedSentences,
      baseTokenBudget,
      effectiveTokenBudget,
    };
  } catch (error) {
    console.error(error);
    setRunStatusState({ type: "error", message: error.message });
  } finally {
    if (runId === activeRun) {
      dom.button.disabled = false;
    }
  }
}

function splitSentences(text) {
  const sentences = [];
  for (const paragraph of text.split(/\n+/u)) {
    const trimmed = paragraph.trim();
    if (!trimmed) {
      continue;
    }
    let start = 0;
    SENTENCE_END_RE.lastIndex = 0;
    for (const match of trimmed.matchAll(SENTENCE_END_RE)) {
      const end = match.index + match[0].length;
      const piece = trimmed.slice(start, end).trim();
      if (piece) {
        sentences.push(piece);
      }
      start = end;
    }
    const tail = trimmed.slice(start).trim();
    if (tail) {
      sentences.push(tail);
    }
  }
  return sentences.length ? sentences : text.trim() ? [text.trim()] : [];
}

function prepareSentence(text, index) {
  const tokens = tokenizeSurface(text);
  const encoded = encodeSentence(tokens.map((token) => token.text));
  return {
    index,
    text,
    tokens,
    ...encoded,
  };
}

function tokenizeSurface(text) {
  return [...text.matchAll(TOKEN_RE)].map((match) => ({
    text: match[0].normalize("NFC"),
    start: match.index || 0,
    end: (match.index || 0) + match[0].length,
    isWord: hasLetter(match[0]),
  }));
}

function encodeSentence(words) {
  const inputIds = [runtime.bosId];
  const firstSubword = Array(words.length).fill(-1);
  const lastSubword = Array(words.length).fill(-1);
  for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
    const tokenIds = normalizeIds(
      runtime.tokenizer.encode(` ${words[wordIndex]}`, { add_special_tokens: false }),
    );
    if (!tokenIds.length) {
      continue;
    }
    if (inputIds.length + tokenIds.length + 1 > MAX_SUBWORDS) {
      continue;
    }
    firstSubword[wordIndex] = inputIds.length;
    inputIds.push(...tokenIds);
    lastSubword[wordIndex] = inputIds.length - 1;
  }
  inputIds.push(runtime.eosId);
  return {
    inputIds,
    firstSubword,
    lastSubword,
    subwordLength: inputIds.length,
  };
}

function normalizeIds(ids) {
  if (ids && Array.isArray(ids)) {
    return ids.map(Number);
  }
  if (ids?.data) {
    return Array.from(ids.data, Number);
  }
  return [];
}

function adaptiveTokenBudget(tokenBudget, memory = updateMemoryStatus()) {
  if (
    memory.wasmMemoryCount > 0 &&
    memory.wasmRatio !== null &&
    memory.wasmRatio > WASM_HIGH_WATER_RATIO
  ) {
    return Math.max(MIN_TOKEN_BUDGET, Math.floor(tokenBudget / 2));
  }
  return tokenBudget;
}

function buildBatches(sentences, tokenBudget) {
  const sorted = [...sentences].sort((left, right) => {
    const byLength = right.subwordLength - left.subwordLength;
    return byLength || left.index - right.index;
  });
  const batches = [];
  let current = [];
  let currentMax = 0;

  for (const sentence of sorted) {
    const nextMax = Math.max(currentMax, sentence.subwordLength);
    const nextCost = nextMax * (current.length + 1);
    if (current.length && nextCost > tokenBudget) {
      batches.push(current);
      current = [];
      currentMax = 0;
    }
    current.push(sentence);
    currentMax = Math.max(currentMax, sentence.subwordLength);
  }
  if (current.length) {
    batches.push(current);
  }
  return batches;
}

async function runBatch(batch, state) {
  let feeds = null;
  let outputs = null;
  try {
    feeds = makeFeeds(batch, state);
    outputs = await runSession(state, feeds);
    return decodeBatch(batch, outputs, state);
  } finally {
    await disposeOrtValues(outputs);
    await disposeOrtValues(feeds);
  }
}

async function runSession(state, feeds) {
  if (memoryTestOverrides.forceAllocationFailure) {
    throw new Error("WebAssembly.Memory allocation failed (forced pilot ceiling test)");
  }
  window.__pilotRunPhase = "session";
  try {
    return await state.session.run(feeds);
  } finally {
    window.__pilotRunPhase = "idle";
  }
}

async function disposeOrtValues(values) {
  if (!values) {
    return;
  }
  for (const value of Object.values(values)) {
    const dispose = value?.dispose;
    if (typeof dispose !== "function") {
      continue;
    }
    try {
      await dispose.call(value);
    } catch (error) {
      console.debug("ORT value dispose failed; continuing.", error);
    }
  }
}

function isMemoryAllocationError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return (
    message.includes("out of memory") ||
    message.includes("allocation failed") ||
    message.includes("failed to allocate") ||
    message.includes("could not allocate") ||
    message.includes("cannot enlarge memory") ||
    message.includes("array buffer allocation") ||
    (message.includes("wasm") && message.includes("memory"))
  );
}

function makeFeeds(batch, state) {
  const batchSize = batch.length;
  const subwords = Math.max(...batch.map((sentence) => sentence.inputIds.length));
  const words = Math.max(...batch.map((sentence) => sentence.tokens.length), 1);
  const maxChars = state.maxChars;
  const inputIds = new Array(batchSize * subwords).fill(state.padId);
  const attentionMask = new Array(batchSize * subwords).fill(0);
  const firstSubword = new Array(batchSize * words).fill(-1);
  const lastSubword = new Array(batchSize * words).fill(-1);
  const charIds = new Array(batchSize * words * maxChars).fill(0);

  for (let row = 0; row < batchSize; row += 1) {
    const sentence = batch[row];
    for (let col = 0; col < sentence.inputIds.length; col += 1) {
      const offset = row * subwords + col;
      inputIds[offset] = sentence.inputIds[col];
      attentionMask[offset] = 1;
    }
    for (let word = 0; word < sentence.tokens.length; word += 1) {
      firstSubword[row * words + word] = sentence.firstSubword[word];
      lastSubword[row * words + word] = sentence.lastSubword[word];
      const keyChars = Array.from(wordKey(sentence.tokens[word].text));
      for (let char = 0; char < Math.min(keyChars.length, maxChars); char += 1) {
        charIds[(row * words + word) * maxChars + char] =
          state.charVocab[keyChars[char]] ?? 1;
      }
    }
  }

  return {
    input_ids: int64Tensor(inputIds, [batchSize, subwords]),
    attention_mask: int64Tensor(attentionMask, [batchSize, subwords]),
    first_subword: int64Tensor(firstSubword, [batchSize, words]),
    last_subword: int64Tensor(lastSubword, [batchSize, words]),
    char_ids: int64Tensor(charIds, [batchSize, words, maxChars]),
  };
}

function decodeBatch(batch, outputs, state) {
  const posLogits = outputs.pos_logits;
  const stressLogits = outputs.stress_logits;
  const noStressLogits = outputs.no_stress_logits;
  const labelCount = state.labels.length;
  const markCount = state.marks.length;
  const maxChars = state.maxChars;
  const maxWords = posLogits.dims[1];
  return batch.map((sentence, row) => ({
    index: sentence.index,
    tokens: sentence.tokens.map((surface, word) => {
      const predicted = sentence.firstSubword[word] >= 0 && word < maxWords;
      if (!predicted) {
        return {
          accented: surface.text,
          className: "",
          predicted: false,
          pos: [],
          noStress: false,
        };
      }
      const posOffset = (row * maxWords + word) * labelCount;
      const pos = decodePos(
        posLogits.data,
        posOffset,
        state.labels,
        state.labelBridgeCache,
      );
      const stressOffset = (row * maxWords + word) * maxChars * markCount;
      const noStressOffset = row * maxWords + word;
      const stress = decodeStress(
        stressLogits.data,
        noStressLogits.data[noStressOffset],
        stressOffset,
        surface.text,
        state.marks,
        maxChars,
      );
      return {
        accented:
          stress && !stress.noStress
            ? applyStress(surface.text, stress.pos, stress.mark)
            : surface.text,
        className: tokenClassFor(pos, Boolean(stress?.noStress)),
        predicted: true,
        pos,
        noStress: Boolean(stress?.noStress),
      };
    }),
  }));
}

function decodePos(logits, offset, labels, labelBridgeCache) {
  let max = -Infinity;
  for (let i = 0; i < labels.length; i += 1) {
    max = Math.max(max, logits[offset + i]);
  }
  let denom = 0;
  for (let i = 0; i < labels.length; i += 1) {
    denom += Math.exp(logits[offset + i] - max);
  }
  const merged = new Map();
  for (let i = 0; i < labels.length; i += 1) {
    const probability = Math.exp(logits[offset + i] - max) / denom;
    if (probability <= POS_PROB_CUT) {
      continue;
    }
    const mi = labelBridgeCache.get(labels[i]) || labels[i];
    merged.set(mi, (merged.get(mi) || 0) + probability);
  }
  return Array.from(merged, ([label, probability]) => [label, probability])
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, MAX_POPOVER_ROWS);
}

function decodeStress(logits, noStressLogit, offset, word, marks, maxChars) {
  const key = wordKey(word);
  const chars = Array.from(key);
  let bestLogit = noStressLogit;
  let best = null;
  for (let pos = 0; pos < Math.min(chars.length, maxChars); pos += 1) {
    for (let markIndex = 0; markIndex < marks.length; markIndex += 1) {
      const mark = marks[markIndex];
      if (!validTarget(chars, pos, mark)) {
        continue;
      }
      const value = logits[offset + pos * marks.length + markIndex];
      if (value > bestLogit) {
        bestLogit = value;
        best = { pos, mark, noStress: false };
      }
    }
  }
  return best || { noStress: true };
}

function validTarget(chars, pos, mark) {
  if (!(pos >= 0 && pos < chars.length)) {
    return false;
  }
  const ch = chars[pos];
  const prev = pos > 0 ? chars[pos - 1] : "";
  const next = pos + 1 < chars.length ? chars[pos + 1] : "";
  if (SONORANTS.has(ch)) {
    return mark === TILDE && VOWELS.has(prev);
  }
  if (!VOWELS.has(ch)) {
    return false;
  }
  if (LONG_VOWELS.has(ch)) {
    return mark !== GRAVE;
  }
  if ((ch === "i" || ch === "u") && !VOWELS.has(prev) && !VOWELS.has(next)) {
    if (SONORANTS.has(next)) {
      return mark === GRAVE;
    }
    if (ch === "i") {
      return mark === GRAVE;
    }
    return mark !== ACUTE;
  }
  return true;
}

function applyStress(word, pos, mark) {
  const plain = stripAccents(normalizeLt(word));
  const chars = Array.from(plain);
  if (pos < 0 || pos >= chars.length) {
    return word;
  }
  chars.splice(pos + 1, 0, mark);
  return normalizeNotation(chars.join("").normalize("NFC"));
}

function normalizeLt(text) {
  if (!text) {
    return "";
  }
  const out = [];
  let lastBase = "";
  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch)) {
      if (ch === COMBINING_DOT_ABOVE && I_DOT_BASES.has(lastBase)) {
        continue;
      }
      out.push(ch);
    } else {
      lastBase = ch;
      out.push(ch);
    }
  }
  return out.join("").normalize("NFC");
}

function stripAccents(text) {
  if (!text) {
    return "";
  }
  const out = [];
  let lastBase = "";
  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch)) {
      if (STRESS_MARKS.has(ch)) {
        continue;
      }
      if (ch === COMBINING_DOT_ABOVE && I_DOT_BASES.has(lastBase)) {
        continue;
      }
      out.push(ch);
    } else {
      lastBase = ch;
      out.push(ch);
    }
  }
  return out.join("").normalize("NFC");
}

function wordKey(text) {
  return stripAccents(normalizeLt(text)).toLowerCase();
}

function normalizeNotation(text) {
  if (!text || !hasStress(text)) {
    return text;
  }
  const clusters = graphemeClusters(text);
  const moves = [];
  for (let i = 0; i < clusters.length; i += 1) {
    const base = plainBase(clusters[i]);
    if (!base) {
      continue;
    }
    const next = i + 1 < clusters.length ? plainBase(clusters[i + 1]) : null;
    const prev = i > 0 ? plainBase(clusters[i - 1]) : null;
    const after = i + 2 < clusters.length ? clusters[i + 2][0].toLowerCase() : "";
    if (clusters[i].includes(TILDE)) {
      if (
        next &&
        PURE_DIPHTHONGS.has(base + next) &&
        !PURE_DIPHTHONGS.has(next + after)
      ) {
        moves.push([i, i + 1, TILDE, TILDE]);
      } else if (
        "aeiu".includes(base) &&
        next &&
        SONORANTS.has(next) &&
        !VOWEL_BASES.has(after) &&
        !(prev && PURE_DIPHTHONGS.has(prev + base))
      ) {
        moves.push([i, i + 1, TILDE, TILDE]);
      }
    } else if (
      clusters[i].includes(ACUTE) &&
      SONORANTS.has(base) &&
      prev &&
      VOWEL_BASES.has(prev)
    ) {
      moves.push([i, i, ACUTE, TILDE]);
    }
  }
  for (const [src, dst, drop, add] of moves) {
    if (src !== dst && clusters[dst].some((mark) => STRESS_MARKS.has(mark))) {
      continue;
    }
    const dropIndex = clusters[src].indexOf(drop);
    if (dropIndex >= 0) {
      clusters[src].splice(dropIndex, 1);
      clusters[dst].push(add);
    }
  }
  return clusters.flat().join("").normalize("NFC");
}

function graphemeClusters(text) {
  const clusters = [];
  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch) && clusters.length) {
      clusters[clusters.length - 1].push(ch);
    } else {
      clusters.push([ch]);
    }
  }
  return clusters;
}

function plainBase(cluster) {
  if (cluster.slice(1).every((mark) => STRESS_MARKS.has(mark))) {
    return cluster[0].toLowerCase();
  }
  return null;
}

function hasStress(text) {
  return Array.from(text.normalize("NFD")).some((ch) => STRESS_MARKS.has(ch));
}

function isCombining(ch) {
  return /\p{M}/u.test(ch);
}

function hasLetter(text) {
  return /\p{L}/u.test(text);
}

function buildLabelBridgeCache(bridge, labels) {
  const miVocab = Array.isArray(bridge?.mi_vocab) ? bridge.mi_vocab : [];
  const modelLabels = bridge?.model_labels || {};
  const cache = new Map();
  for (const label of labels) {
    cache.set(label, bestMiForSlots(modelLabels[label] || {}, miVocab));
  }
  return cache;
}

function bestMiForSlots(contextSlots, miVocab) {
  let best = null;
  for (const candidate of miVocab) {
    const score = scoreTags(candidate.slots || {}, contextSlots || {});
    const spurious = spuriousSlots(candidate.slots || {}, contextSlots || {});
    if (
      !best ||
      score > best.score ||
      (score === best.score && spurious < best.spurious) ||
      (
        score === best.score &&
        spurious === best.spurious &&
        candidate.label.length < best.label.length
      )
    ) {
      best = { label: candidate.label, score, spurious };
    }
  }
  return best?.label || "";
}

function scoreTags(variantTags, contextTags) {
  let score = 0;
  if (variantTags.pos && contextTags.pos) {
    score += variantTags.pos === contextTags.pos ? 4 : -3;
  }
  for (const slot of SCORING_SLOTS) {
    const variantValue = variantTags[slot];
    const contextValue = contextTags[slot];
    if (!variantValue || !contextValue) {
      continue;
    }
    score += variantValue === contextValue ? 2 : -2;
  }
  return score;
}

function spuriousSlots(variantTags, contextTags) {
  let count = 0;
  for (const slot of Object.keys(variantTags)) {
    if (!(slot in contextTags)) {
      count += 1;
    }
  }
  return count;
}

function renderPlaceholders(sentences) {
  dom.result.classList.remove("is-empty");
  dom.result.textContent = "";
  const fragment = document.createDocumentFragment();
  for (const sentence of sentences) {
    const node = document.createElement("span");
    node.className = "sentence is-pending";
    node.dataset.index = String(sentence.index);
    node.textContent = sentence.index + 1 < sentences.length ? `${sentence.text} ` : sentence.text;
    fragment.append(node);
  }
  dom.result.append(fragment);
  dom.result.dataset.sentenceCount = String(sentences.length);
  dom.result.dataset.renderedCount = "0";
}

function renderSentence(sentence, decodedTokens) {
  const node = dom.result.querySelector(`[data-index="${sentence.index}"]`);
  if (!node) {
    return;
  }
  node.textContent = "";
  node.classList.remove("is-pending");
  const fragment = document.createDocumentFragment();
  let last = 0;
  for (let i = 0; i < sentence.tokens.length; i += 1) {
    const surface = sentence.tokens[i];
    const decoded = decodedTokens[i];
    if (surface.start > last) {
      fragment.append(document.createTextNode(sentence.text.slice(last, surface.start)));
    }
    if (surface.isWord && decoded?.predicted) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = decoded.className;
      button.textContent = decoded.accented;
      button.dataset.pos = JSON.stringify(decoded.pos);
      button.dataset.word = decoded.accented;
      button.dataset.noStress = String(Boolean(decoded.noStress));
      if (decoded.noStress) {
        button.title = PILOT_UI[lang].noStressTitle;
      }
      button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", () => showPopover(button));
      fragment.append(button);
    } else {
      fragment.append(document.createTextNode(decoded?.accented ?? surface.text));
    }
    last = surface.end;
  }
  if (last < sentence.text.length) {
    fragment.append(document.createTextNode(sentence.text.slice(last)));
  }
  if (sentence.index + 1 < Number(dom.result.dataset.sentenceCount || "0")) {
    fragment.append(document.createTextNode(" "));
  }
  node.append(fragment);
  dom.result.dataset.renderedCount = String(
    Number(dom.result.dataset.renderedCount || "0") + 1,
  );
}

function tokenClassFor(pos, noStress) {
  if (noStress) {
    return "token token-unknown";
  }
  const rowsAboveCut = pos.filter((row) => row[1] > POS_PROB_CUT);
  if (rowsAboveCut.length >= 2) {
    return "token token-unresolved";
  }
  if (pos.length === 1 && pos[0][1] >= RESOLVED_PROB) {
    return "token token-resolved";
  }
  return "token token-plain";
}

function showPopover(button) {
  const rows = normalizePopoverRows(JSON.parse(button.dataset.pos || "[]"));
  const word = button.dataset.word || button.textContent || "";
  dom.popover.textContent = "";
  const title = document.createElement("p");
  title.className = "pos-title";
  title.textContent = word;
  dom.popover.append(title);
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "pos-row";
    empty.textContent = PILOT_UI[lang].popoverEmpty;
    dom.popover.append(empty);
  }
  for (const item of rows) {
    const row = document.createElement("div");
    row.className = "pos-row";
    const label = document.createElement("span");
    label.className = "pos-label";
    label.textContent = item.label;
    const prob = document.createElement("span");
    prob.className = "pos-prob";
    prob.textContent = `${(item.probability * 100).toFixed(1)}%`;
    const gloss = document.createElement("span");
    gloss.className = "pos-gloss";
    appendMorphologyInfo(gloss, morphologySegments(item.label, lang));
    row.append(label, prob, gloss);
    dom.popover.append(row);
  }
  positionPopover(dom.popover, button);
}

function normalizePopoverRows(rows) {
  if (!Array.isArray(rows)) {
    return [];
  }
  return rows
    .map((row) => {
      if (Array.isArray(row)) {
        return { label: String(row[0] || ""), probability: Number(row[1] || 0) };
      }
      return {
        label: String(row?.label || ""),
        probability: Number(row?.probability || 0),
      };
    })
    .filter((row) => row.label)
    .slice(0, MAX_POPOVER_ROWS);
}

function hidePopover() {
  dom.popover.hidden = true;
}

function positionPopover(popover, anchor) {
  popover.hidden = false;
  const rect = anchor.getBoundingClientRect();
  const popRect = popover.getBoundingClientRect();
  const top = window.scrollY + rect.bottom + 8;
  const left = Math.min(
    Math.max(window.scrollX + rect.left, 12),
    window.scrollX + window.innerWidth - popRect.width - 12,
  );
  popover.style.top = `${top}px`;
  popover.style.left = `${left}px`;
}

function appendMorphologyInfo(container, segments) {
  segments.forEach((segment) => {
    if (!segment.lt) {
      container.append(document.createTextNode(segment.text));
      return;
    }
    const ruby = document.createElement("ruby");
    ruby.append(document.createTextNode(segment.lt));
    const rt = document.createElement("rt");
    rt.textContent = segment.text;
    ruby.append(rt);
    container.append(ruby);
  });
}

function collectRenderedText(sentences, decodedBySentence) {
  return sentences
    .map((sentence) => {
      let out = "";
      let last = 0;
      const decodedTokens = decodedBySentence[sentence.index] || [];
      for (let index = 0; index < sentence.tokens.length; index += 1) {
        const surface = sentence.tokens[index];
        const decoded = decodedTokens[index];
        out += sentence.text.slice(last, surface.start);
        out += decoded?.accented ?? surface.text;
        last = surface.end;
      }
      out += sentence.text.slice(last);
      return out;
    })
    .join(" ");
}

function releaseSentenceInferenceInputs(sentence) {
  sentence.inputIds = null;
  sentence.firstSubword = null;
  sentence.lastSubword = null;
  sentence.subwordLength = 0;
}

function int64Tensor(values, dims) {
  const data = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    data[i] = BigInt(values[i]);
  }
  return new ort.Tensor("int64", data, dims);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchOptionalJson(url) {
  try {
    return await fetchJson(url);
  } catch {
    return null;
  }
}

async function headContentLength(url) {
  try {
    const response = await fetch(url, { method: "HEAD" });
    return Number(response.headers.get("content-length") || 0) || null;
  } catch {
    return null;
  }
}

async function cacheHit(url) {
  if (!("caches" in window)) {
    return false;
  }
  const cache = await openModelCache();
  if (!cache) {
    return false;
  }
  return Boolean(
    (await cache.match(cacheRequest(url))) ||
      (await cache.match(cacheChunkMetadataRequest(url))),
  );
}

async function openModelCache() {
  if (!("caches" in window)) {
    return null;
  }
  try {
    return await caches.open(CACHE_NAME);
  } catch (error) {
    logCacheWriteIssueOnce(error);
    return null;
  }
}

function cacheModelUrl(url) {
  return new URL(url, location.href).href;
}

function cacheRequest(url) {
  return new Request(cacheModelUrl(url), { credentials: "same-origin" });
}

function cacheChunkMetadataRequest(url) {
  return new Request(`${cacheModelUrl(url)}?pilot-cache=chunks`, {
    credentials: "same-origin",
  });
}

function cacheChunkRequest(url, index) {
  return new Request(`${cacheModelUrl(url)}?pilot-cache=chunk-${index}`, {
    credentials: "same-origin",
  });
}

async function readChunkedCacheBytes(cache, url, expectedBytes) {
  const metadataResponse = await cache.match(cacheChunkMetadataRequest(url));
  if (!metadataResponse) {
    return null;
  }
  let metadata = null;
  try {
    metadata = await metadataResponse.json();
  } catch {
    return null;
  }
  const total = Number(metadata?.bytes || 0);
  const chunkCount = Number(metadata?.chunks || 0);
  if (
    metadata?.version !== 1 ||
    !Number.isFinite(total) ||
    total <= 0 ||
    !Number.isInteger(chunkCount) ||
    chunkCount <= 0 ||
    (expectedBytes && total !== expectedBytes)
  ) {
    return null;
  }

  const chunks = [];
  let received = 0;
  for (let index = 0; index < chunkCount; index += 1) {
    const chunkResponse = await cache.match(cacheChunkRequest(url, index));
    if (!chunkResponse) {
      return null;
    }
    const chunk = new Uint8Array(await chunkResponse.arrayBuffer());
    chunks.push(chunk);
    received += chunk.byteLength;
    setModelStatusState({
      type: "transfer",
      cached: true,
      received,
      total,
    });
  }
  if (received !== total) {
    return null;
  }
  return assembleModelBytes(chunks, received);
}

async function estimateCacheWriteHeadroom(modelBytes) {
  const bytes = Number(modelBytes);
  const required =
    Number.isFinite(bytes) && bytes > 0
      ? bytes * CACHE_HEADROOM_MULTIPLIER
      : null;
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
      reason: "storage-estimate-failed",
      usage: null,
      quota: null,
      headroom: null,
      required,
      error: error.message,
    };
  }
}

function makeCacheWriteState(url) {
  return {
    url,
    status: "miss",
    attempted: false,
    completed: false,
    skipped: false,
    watchdogFired: false,
    estimate: null,
    reason: null,
    error: null,
    startedAt: performance.now(),
    completedAt: null,
  };
}

function updateCacheWriteState(state, updates) {
  Object.assign(state, updates);
  publishCacheWriteState(state);
}

function publishCacheWriteState(state) {
  window.__pilotCacheWrite = {
    url: state.url,
    status: state.status,
    attempted: state.attempted,
    completed: state.completed,
    skipped: state.skipped,
    watchdogFired: state.watchdogFired,
    estimate: state.estimate,
    reason: state.reason,
    error: state.error,
    elapsedMs:
      state.completedAt == null
        ? performance.now() - state.startedAt
        : state.completedAt - state.startedAt,
  };
}

function startCacheWriteAfterMainRead(cache, url, eligibility) {
  const state = makeCacheWriteState(url);
  publishCacheWriteState(state);
  const decision =
    eligibility ?? {
      ok: false,
      reason: "storage-estimate-not-started",
      usage: null,
      quota: null,
      headroom: null,
      required: null,
    };
  updateCacheWriteState(state, { estimate: decision });

  if (cacheWriteClosedUrls.has(url)) {
    updateCacheWriteState(state, {
      completed: true,
      skipped: true,
      reason: "already-handled-this-load",
      completedAt: performance.now(),
    });
    return state;
  }
  cacheWriteClosedUrls.add(url);

  if (!decision.ok) {
    updateCacheWriteState(state, {
      status: "unavailable",
      completed: true,
      skipped: true,
      reason: decision.reason,
      completedAt: performance.now(),
    });
    return state;
  }

  state.start = () => {
    if (state.promise) {
      return state.promise;
    }
    state.promise = (async () => {
      updateCacheWriteState(state, { attempted: true });
      try {
        await writeCacheFromIndependentFetch(cache, url, state);
        updateCacheWriteState(state, {
          status: "stored",
          completed: true,
          completedAt: performance.now(),
        });
      } catch (error) {
        updateCacheWriteState(state, {
          status: "failed",
          completed: true,
          reason: state.watchdogFired ? "watchdog-timeout" : "cache-write-failed",
          error: error.message,
          completedAt: performance.now(),
        });
        logCacheWriteIssueOnce(error);
      }
      return state.status;
    })();
    state.promise.catch(() => {});
    return state.promise;
  };
  return state;
}

async function writeCacheFromIndependentFetch(cache, url, state) {
  const controller = new AbortController();
  const modelRequest = cacheRequest(url);
  let cacheResponse = null;
  let timeoutId = 0;
  let rejectWatchdog = null;
  const timeout = new Promise((_, reject) => {
    rejectWatchdog = reject;
  });
  const touchWatchdog = () => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => {
      updateCacheWriteState(state, { watchdogFired: true });
      cancelCacheWriteBranch(controller, cacheResponse);
      rejectWatchdog(new Error("Cache API write watchdog timed out"));
    }, CACHE_WRITE_WATCHDOG_MS);
  };
  touchWatchdog();
  const operation = (async () => {
    cacheResponse = await fetch(modelRequest, { signal: controller.signal });
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

async function writeChunkedCacheResponse(cache, url, response, touchWatchdog) {
  await cache.delete(cacheChunkMetadataRequest(url));
  const total = Number(response.headers.get("content-length") || 0) || null;
  const chunks = [];
  let received = 0;
  let chunkIndex = 0;
  let buffer = new Uint8Array(CACHE_CHUNK_BYTES);
  let offset = 0;

  const writeChunk = async (bytes) => {
    await cache.put(
      cacheChunkRequest(url, chunkIndex),
      new Response(bytes, {
        headers: {
          "content-type": "application/octet-stream",
          "x-pilot-cache-chunk": String(chunkIndex),
        },
      }),
    );
    chunks.push(bytes.byteLength);
    chunkIndex += 1;
    touchWatchdog();
  };

  const appendBytes = async (value) => {
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

function cancelCacheWriteBranch(controller, response) {
  try {
    controller.abort();
  } catch {
    // Best effort; the branch is independent from the model bytes already read.
  }
  if (response?.body && !response.body.locked) {
    response.body.cancel().catch(() => {});
  }
}

function logCacheWriteIssueOnce(error) {
  if (cacheWriteIssueLogged) {
    return;
  }
  cacheWriteIssueLogged = true;
  console.info("Cache API store failed; continuing without cached model.", error);
}

async function fetchWithCache(url, expectedBytes) {
  const cache = await openModelCache();
  let response = cache ? await cache.match(cacheRequest(url)) : null;
  const cached = Boolean(response);
  let cacheStatus = cache ? (cached ? "hit" : "miss") : "unavailable";
  let cacheWriteState = null;
  let cacheEligibility = null;
  if (!response && cache) {
    const bytes = await readChunkedCacheBytes(cache, url, expectedBytes);
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
  const bytes = await readResponseBytes(response, expectedBytes, cached);
  if (cache && !cached) {
    cacheWriteState = startCacheWriteAfterMainRead(cache, url, cacheEligibility);
    cacheStatus = cacheWriteState.status;
  }
  return { bytes, cacheStatus, cacheWriteState };
}

async function readResponseBytes(response, expectedBytes, cached) {
  const total = Number(response.headers.get("content-length") || expectedBytes || 0);
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    const out = new Uint8Array(buffer);
    setModelStatusState({ type: "session", bytes: out.byteLength });
    return out;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    received += value.byteLength;
    if (total) {
      setModelStatusState({
        type: "transfer",
        cached,
        received,
        total,
      });
    }
  }
  return assembleModelBytes(chunks, received);
}

function assembleModelBytes(chunks, received) {
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  chunks.length = 0;
  setModelStatusState({ type: "session", bytes: out.byteLength });
  return out;
}

async function measureAfterLoadBufferRelease() {
  for (let i = 0; i < 3; i += 1) {
    await nextFrame();
    if (typeof window.gc === "function") {
      window.gc();
    }
  }
  return updateMemoryStatus();
}

async function measureBeforeSessionCreate() {
  await nextFrame();
  if (typeof window.gc === "function") {
    window.gc();
  }
  return updateMemoryStatus();
}

function setLanguage(nextLang, options = { persist: true }) {
  lang = nextLang;
  document.documentElement.lang = lang;
  dom.textarea.lang = lang;
  dom.result.lang = lang;
  if (options.persist) {
    localStorage.setItem("lang", lang);
  }
  hidePopover();
  renderUi();
  if (dom.result.classList.contains("is-empty")) {
    setResultEmpty(UI[lang].resultEmpty);
  }
}

function renderUi() {
  const strings = UI[lang];
  const pilot = PILOT_UI[lang];
  document.title = `${pilot.title} · bundled weights pilot`;
  dom.metaDescription?.setAttribute(
    "content",
    `${strings.tagline} ${pilot.subtitle}`,
  );
  dom.appTitle.textContent = pilot.title;
  dom.pilotEyebrow.textContent = pilot.eyebrow;
  dom.heroTagline.textContent = strings.tagline;
  dom.pilotSubtitle.textContent = pilot.subtitle;
  dom.inputLabel.textContent = strings.inputLabel;
  dom.tokenBudgetLabel.textContent = pilot.tokenBudgetLabel;
  dom.button.textContent = strings.accentButton;
  dom.copyButton.textContent = copied ? strings.copied : strings.copyButton;
  dom.resultHeading.textContent = strings.resultHeading;
  dom.legend.setAttribute("aria-label", strings.legendLabel);
  dom.legendLabel.textContent = strings.legendLabel;
  dom.legendResolved.textContent = strings.legendResolved;
  dom.legendAmbiguous.textContent = strings.legendAmbiguous;
  dom.legendUnknown.textContent = strings.legendUnknown;
  dom.siteFooter.textContent = pilot.footer;
  renderPrimer(strings);
  dom.languageButtons.forEach((button) => {
    const buttonLang = parseLang(button.dataset.lang);
    const isCurrent = buttonLang === lang;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });
  updateCounter();
  renderModelStatus();
  renderRunStatus();
  renderMemoryStatus();
}

function renderPrimer(strings) {
  dom.primerLink.textContent = strings.primerLink;
  dom.primerTitle.textContent = strings.primerTitle;
  dom.primerIntro.textContent = strings.primerIntro;
  dom.primerGraveName.textContent = strings.primerGraveName;
  dom.primerGraveDesc.textContent = strings.primerGraveDesc;
  dom.primerGraveEx.textContent = strings.primerGraveEx;
  dom.primerAcuteName.textContent = strings.primerAcuteName;
  dom.primerAcuteDesc.textContent = strings.primerAcuteDesc;
  dom.primerAcuteEx.textContent = strings.primerAcuteEx;
  dom.primerTildeName.textContent = strings.primerTildeName;
  dom.primerTildeDesc.textContent = strings.primerTildeDesc;
  dom.primerTildeEx.textContent = strings.primerTildeEx;
  renderTextWithLtWords(dom.primerMixed, strings.primerMixed, PRIMER_MIXED_WORDS);
  renderTextWithLtWords(dom.primerPair, strings.primerPair, PRIMER_PAIR_WORDS);
  dom.primerMore.href = VLKK_PRIMER_URL;
  dom.primerMore.textContent = strings.primerMore;
}

function renderTextWithLtWords(container, text, ltWords) {
  container.replaceChildren();
  let cursor = 0;
  while (cursor < text.length) {
    const next = findNextLtWord(text, ltWords, cursor);
    if (!next) {
      container.append(document.createTextNode(text.slice(cursor)));
      return;
    }
    if (next.index > cursor) {
      container.append(document.createTextNode(text.slice(cursor, next.index)));
    }
    const word = document.createElement("span");
    word.lang = "lt";
    word.textContent = next.word;
    container.append(word);
    cursor = next.index + next.word.length;
  }
}

function findNextLtWord(text, ltWords, cursor) {
  let next = null;
  ltWords.forEach((word) => {
    const index = text.indexOf(word, cursor);
    if (index !== -1 && (!next || index < next.index)) {
      next = { index, word };
    }
  });
  return next;
}

function getPrimerMixedWords() {
  const pieces = UI.lt.primerMixed.split(": ");
  const examples = pieces[pieces.length - 1]?.replace(/\.$/, "") ?? "";
  return examples
    .split(", ")
    .map((word) => word.trim())
    .filter(Boolean);
}

function getPrimerPairWords() {
  const match = UI.lt.primerPair.match(/: ([^(]+)\s\([^)]*\) ir ([^(]+)\s\(/);
  return match ? [match[1].trim(), match[2].trim()] : [];
}

function openPrimer() {
  hidePopover();
  dom.primerBackdrop.hidden = false;
  document.body.classList.add("has-primer-open");
  dom.primerDialog.focus({ preventScroll: true });
}

function closePrimer() {
  if (!isPrimerOpen()) {
    return;
  }
  dom.primerBackdrop.hidden = true;
  document.body.classList.remove("has-primer-open");
  dom.primerLink.focus({ preventScroll: true });
}

function isPrimerOpen() {
  return !dom.primerBackdrop.hidden;
}

function trapPrimerFocus(event) {
  const focusable = Array.from(
    dom.primerDialog.querySelectorAll(FOCUSABLE_SELECTOR),
  ).filter((element) => element.getClientRects().length > 0);
  if (focusable.length === 0) {
    event.preventDefault();
    dom.primerDialog.focus({ preventScroll: true });
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
    return;
  }
  if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function setResultEmpty(text) {
  dom.result.classList.add("is-empty");
  dom.result.textContent = text;
}

function readMemoryStatus() {
  const wasm = wasmMemoryTracker.read();
  const wasmMaxBytes =
    memoryTestOverrides.wasmMaxBytes ?? wasm.wasmMaxBytes;
  const js = performance.memory
    ? {
        jsHeapBytes: performance.memory.usedJSHeapSize,
        jsHeapLimitBytes: performance.memory.jsHeapSizeLimit,
      }
    : { jsHeapBytes: null, jsHeapLimitBytes: null };
  return {
    ...wasm,
    wasmMaxBytes,
    ...js,
    wasmRatio:
      wasmMaxBytes > 0 ? wasm.wasmBytes / wasmMaxBytes : null,
  };
}

function updateMemoryStatus() {
  memoryStatusState = readMemoryStatus();
  window.__pilotMemoryStatus = memoryStatusState;
  renderMemoryStatus();
  return memoryStatusState;
}

function renderMemoryStatus() {
  const pilot = PILOT_UI[lang];
  const session = SESSION_UI[lang] || SESSION_UI.en;
  const wasm = formatMemoryMb(memoryStatusState.wasmBytes);
  const js = formatMemoryMb(memoryStatusState.jsHeapBytes);
  const mode = runtimeExecutionMode
    ? `${session.modeLabel}: ${executionModeLabel(runtimeExecutionMode)} / `
    : "";
  dom.memoryStatus.textContent =
    `${pilot.memoryLabel}: ${mode}${pilot.wasmMemoryLabel} ${wasm} / ` +
    `${pilot.jsHeapMemoryLabel} ${js}`;
}

function setModelStatusState(state) {
  modelStatusState = state;
  renderModelStatus();
}

function renderModelStatus() {
  const pilot = PILOT_UI[lang];
  const state = modelStatusState;
  switch (state.type) {
    case "metadata":
      dom.modelStatus.textContent = pilot.modelMetadata;
      break;
    case "loadFailed":
      dom.modelStatus.textContent = `${pilot.modelLoadFailed}: ${state.message}`;
      break;
    case "modelInfo":
      dom.modelStatus.textContent =
        `${pilot.modelLabel}: ${formatBytes(state.expectedBytes)} · ` +
        `${pilot.cacheLabel}: ${state.cacheState ? pilot.cachePresent : pilot.cacheWillFill} · ` +
        `${pilot.wasmThreads}: ${state.threads}`;
      break;
    case "transfer":
      dom.modelStatus.textContent =
        `${state.cached ? pilot.readingCache : pilot.downloading} · ` +
        `${formatBytes(state.received)}/${formatBytes(state.total)}`;
      break;
    case "session":
      dom.modelStatus.textContent =
        `${sessionStatusText(state.mode)} · ${formatBytes(state.bytes)}.`;
      break;
    case "readyModel":
      dom.modelStatus.textContent =
        `${pilot.ready} · ${state.modelFile} · ${formatBytes(state.bytes)} · ` +
        `${executionModeLabel(state.executionMode)} · ` +
        `${pilot.cacheLabel}: ${cacheLabel(state.cacheStatus)} · ` +
        `${pilot.wasmThreads}: ${state.threads}`;
      break;
    default:
      dom.modelStatus.textContent = pilot.modelLoading;
  }
}

function sessionStatusText(mode) {
  const session = SESSION_UI[lang] || SESSION_UI.en;
  switch (mode) {
    case "worker":
      return session.sessionWorker;
    case "fallback":
      return session.sessionFallback;
    case "main":
      return session.sessionMain;
    default:
      return PILOT_UI[lang].creatingSession;
  }
}

function executionModeLabel(mode) {
  const session = SESSION_UI[lang] || SESSION_UI.en;
  if (mode === "worker") {
    return session.workerMode;
  }
  if (mode === "main") {
    return session.mainMode;
  }
  return PILOT_UI[lang].unknownSize;
}

function setRunStatusState(state) {
  runStatusState = state;
  renderRunStatus();
}

function renderRunStatus() {
  const pilot = PILOT_UI[lang];
  const state = runStatusState;
  switch (state.type) {
    case "copied":
      dom.runStatus.textContent = UI[lang].copied;
      break;
    case "running":
      dom.runStatus.textContent =
        `${pilot.sentences}: ${state.sentences} · ` +
        `${pilot.batches}: ${state.batches} · ${pilot.running}`;
      break;
    case "batch":
      dom.runStatus.textContent =
        `${pilot.sentences}: ${state.renderedSentences}/${state.sentences} · ` +
        `${pilot.batch}: ${state.batch}/${state.batches} · ` +
        `${state.tokensPerSecond.toFixed(1)} ${pilot.tokensPerSecond}`;
      break;
    case "done":
      dom.runStatus.textContent =
        `${pilot.done} · ${state.sentences} ${pilot.sentenceShort} · ` +
        `${state.inferredTokens}/${state.totalTokens} ${pilot.tokens} · ` +
        `${state.tokensPerSecond.toFixed(1)} ${pilot.tokensPerSecond} · ` +
        `${(state.elapsedMs / 1000).toFixed(2)} ${pilot.secondsShort}`;
      break;
    case "error":
      dom.runStatus.textContent = `${pilot.errorPrefix}: ${state.message}`;
      break;
    case "memoryLimit":
      dom.runStatus.textContent = pilot.memoryLimitReached;
      break;
    default:
      dom.runStatus.textContent = `${pilot.ready}.`;
  }
}

function updateCounter() {
  dom.counter.textContent = `${dom.textarea.value.length} ${PILOT_UI[lang].charsSuffix}`;
}

function setProgress(value) {
  dom.progressBar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
}

function clampInt(value, min, max, fallback) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, number));
}

function formatBytes(bytes) {
  if (!bytes) {
    return PILOT_UI[lang].unknownSize;
  }
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit + 1 < units.length) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function formatMemoryMb(bytes) {
  if (!bytes) {
    return PILOT_UI[lang].unknownSize;
  }
  return `${(Number(bytes) / 1024 / 1024).toFixed(1)} MB`;
}

function cacheLabel(status) {
  const pilot = PILOT_UI[lang];
  switch (status) {
    case "hit":
      return pilot.cachePresent;
    case "stored":
      return pilot.cacheStored;
    case "failed":
      return pilot.cacheFailed;
    case "unavailable":
      return pilot.cacheUnavailable;
    default:
      return pilot.cacheMiss;
  }
}

function parseLang(value) {
  return LANGS.find((candidate) => candidate === value) ?? null;
}

function getElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }
  return element;
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}
