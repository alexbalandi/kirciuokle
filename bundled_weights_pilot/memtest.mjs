import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_URL = "http://127.0.0.1:8788/";
const REPRO_TEXT =
  "81-erių vilnietė pardavė butą ir nusikaltėliui atidavė 114 tūkst. eurų. Tuo metu valstybės institucijos, nevyriausybinės organizacijos ir verslininkai suka galvas, kaip dar padėti žmonėms nepakliūti į sukčių pinkles. Pavyzdžiui, verslai ant kvitų spausdina patarimus bei numerį, kuriuo reikėtų skambinti įtarus, kad susiduria su nusikaltėliu. Vis dėlto ekspertai pabrėžia, kad svarbiausia vadovautis kritiniu mąstymu ir elgtis atsakingai, jog tokie atvejai nebepasikartotų.";
const RESPONSIVENESS_TEXT =
  "Vilniaus universiteto mokslininkai aiškina, kaip lietuvių kalbos kirčiavimo taisyklės keičiasi gyvoje vartosenoje. " +
  "Studentai skaito ilgus tekstus, žymi neaiškias formas ir tikrina, ar modelis išlaiko nuoseklų ritmą. " +
  "Redaktoriai pastebi, kad oficialiuose pranešimuose dažnai susitinka seni terminai, nauji skolininiai ir sudėtingi linksniai. " +
  "Tyrėjai prašo sistemos ramiai apdoroti kiekvieną sakinį, o sąsaja tuo metu turi likti pasiekiama. " +
  "Jeigu naudotojas pakeičia kalbą, mygtukai turi reaguoti iš karto, net kai naršyklėje vyksta skaičiavimai. " +
  "Tokie bandymai padeda atskirti tikrą spartą nuo gražiai atrodančių, bet trumpų demonstracijų. " +
  "Kiekviena partija turi parodyti dalinį rezultatą, kad ilgesnis straipsnis neatrodytų sustingęs. " +
  "Programuotojai matuoja delsą, atmintį ir tokenų per sekundę skaičių, nes visi trys rodikliai svarbūs. " +
  "Vartotojui svarbiausia, kad tekstas būtų tvarkingas, o puslapis neignoruotų paspaudimų. " +
  "Todėl šis sakinys užbaigia ilgą įvestį, skirtą patikrinti darbą su proxy vykdymu.";

const rawArgs = process.argv.slice(2);
const quotaStressFlag = rawArgs.includes("--quota-stress");
const noWorkerFlag = rawArgs.includes("--no-worker");
const positionalArgs = rawArgs.filter(
  (arg) => arg !== "--quota-stress" && arg !== "--no-worker",
);
const firstArgIsIterations = /^\d+$/u.test(positionalArgs[0] || "");
const phaseKind = quotaStressFlag
  ? "quota-stress"
  : noWorkerFlag && (positionalArgs.length === 0 || firstArgIsIterations)
    ? "no-worker"
    : positionalArgs[0] || "curve";
const phase = noWorkerFlag && phaseKind !== "no-worker" ? `${phaseKind}-no-worker` : phaseKind;
const noWorkerMode = noWorkerFlag || phaseKind === "no-worker";
const iterationArg = quotaStressFlag
  ? positionalArgs[0]
  : noWorkerFlag && phaseKind === "no-worker"
    ? positionalArgs[0]
    : positionalArgs[1];
const defaultIterations = new Set([
  "cold-blocked",
  "no-worker",
  "normal-cache",
  "quota-stress",
  "responsive",
  "worker-mode",
]).has(phaseKind)
  ? "1"
  : "25";
const iterations = Number.parseInt(
  iterationArg || process.env.ITERATIONS || defaultIterations,
  10,
);
const url = process.env.PILOT_URL || DEFAULT_URL;
const runtimeTimeoutMs = Number.parseInt(process.env.RUNTIME_TIMEOUT_MS || "300000", 10);
const runTimeoutMs = Number.parseInt(process.env.RUN_TIMEOUT_MS || "120000", 10);
const cacheWriteTimeoutMs = Number.parseInt(
  process.env.CACHE_WRITE_TIMEOUT_MS || "60000",
  10,
);
const PROXY_WATCHDOG_MS = 45_000;
const WASM_FLAT_TOLERANCE_MB = 64;
const JS_HEAP_FLAT_TOLERANCE_MB = 128;
const WASM_RUNTIME_RESOURCE_RE = /\/model\/runtime\/ort-wasm.*\.(?:mjs|wasm)(?:[?#]|$)/u;
const STRESS_MARK_RE = /[\u0300\u0301\u0303áàãéèẽė̃íìĩýỳỹúùũóòõÁÀÃÉÈẼÍÌĨÝỲỸÚÙŨÓÒÕ]/u;
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

const recentConsole = [];

function loadPlaywright() {
  try {
    return require("playwright");
  } catch {
    const pathEntries = (process.env.PATH || "").split(path.delimiter);
    for (const entry of pathEntries) {
      const normalized = entry.replaceAll("/", path.sep).toLowerCase();
      if (!normalized.endsWith(`${path.sep}node_modules${path.sep}.bin`)) {
        continue;
      }
      const candidate = path.join(path.dirname(entry), "playwright");
      if (fs.existsSync(path.join(candidate, "package.json"))) {
        return require(candidate);
      }
    }
    throw new Error(
      "Playwright is not resolvable. Run with: npx.cmd --yes --package playwright node bundled_weights_pilot\\memtest.mjs",
    );
  }
}

function mb(bytes) {
  return bytes == null ? "n/a" : (Number(bytes) / 1024 / 1024).toFixed(1);
}

function compactStatus(status) {
  return JSON.stringify(status, (_key, value) => {
    if (typeof value === "string" && value.length > 700) {
      return `${value.slice(0, 700)}...`;
    }
    return value;
  });
}

function safeName(label) {
  return label.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "");
}

function isLocalRequest(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" && LOCAL_HOSTS.has(parsed.hostname);
  } catch {
    return false;
  }
}

function isWasmRuntimeRequest(url) {
  return WASM_RUNTIME_RESOURCE_RE.test(url);
}

function hasStressMarks(text) {
  return STRESS_MARK_RE.test(text || "");
}

function rangeMb(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (numeric.length < 2) {
    return null;
  }
  return (Math.max(...numeric) - Math.min(...numeric)) / 1024 / 1024;
}

function fmtRange(value) {
  return value == null ? "n/a" : value.toFixed(1);
}

function sessionWarningFields(sessionCreate) {
  const warnings = sessionCreate?.warnings || [];
  return [
    `warnings=${warnings.length}`,
    `firstWarning=${JSON.stringify(warnings[0]?.message || "")}`,
  ];
}

async function installNoWorkerMock(page) {
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    function BlockedWorker() {
      throw new Error("Worker construction blocked by bundled_weights_pilot memtest --no-worker");
    }
    if (NativeWorker?.prototype) {
      BlockedWorker.prototype = NativeWorker.prototype;
    }
    Object.defineProperty(window, "__pilotNoWorkerHarness", {
      configurable: true,
      value: true,
    });
    Object.defineProperty(window, "Worker", {
      configurable: true,
      writable: true,
      value: BlockedWorker,
    });
  });
}

async function installLocalhostOnlyRoutes(page) {
  const blocked = [];
  await page.route("**/*", async (route) => {
    const request = route.request();
    if (isLocalRequest(request.url())) {
      await route.continue();
      return;
    }
    blocked.push(request.url());
    console.log(`BLOCKED phase=${phase} url=${JSON.stringify(request.url())}`);
    await route.abort("blockedbyclient");
  });
  return blocked;
}

async function installQuotaStressMock(context) {
  await context.addInitScript(() => {
    window.__pilotQuotaStress = {
      openCalls: 0,
      matchCalls: 0,
      putCalls: 0,
      putUrls: [],
      putStartedAt: null,
    };

    const nativeOpen = caches.open.bind(caches);
    Object.defineProperty(caches, "open", {
      configurable: true,
      value: async (name) => {
        window.__pilotQuotaStress.openCalls += 1;
        const nativeCache = await nativeOpen(name);
        return new Proxy(nativeCache, {
          get(target, property) {
            if (property === "match") {
              return (...args) => {
                window.__pilotQuotaStress.matchCalls += 1;
                return target.match(...args);
              };
            }
            if (property === "put") {
              return (request, response) => {
                const requestUrl =
                  typeof request === "string" ? request : request?.url || String(request);
                if (!/\.onnx(?:[?#]|$)/u.test(requestUrl)) {
                  return target.put(request, response);
                }
                window.__pilotQuotaStress.putCalls += 1;
                window.__pilotQuotaStress.putUrls.push(requestUrl);
                window.__pilotQuotaStress.putStartedAt = performance.now();
                return new Promise(() => {});
              };
            }
            const value = target[property];
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
      },
    });
  });
}

function cacheWriteFields(cacheWrite) {
  return [
    `cacheStatus=${cacheWrite?.status ?? "n/a"}`,
    `cacheAttempted=${cacheWrite?.attempted ?? "n/a"}`,
    `cacheCompleted=${cacheWrite?.completed ?? "n/a"}`,
    `cacheSkipped=${cacheWrite?.skipped ?? "n/a"}`,
    `cacheWatchdog=${cacheWrite?.watchdogFired ?? "n/a"}`,
    `cacheReason=${cacheWrite?.reason ?? "n/a"}`,
    `estimateReason=${cacheWrite?.estimate?.reason ?? "n/a"}`,
    `estimateHeadroomMB=${mb(cacheWrite?.estimate?.headroom)}`,
    `estimateRequiredMB=${mb(cacheWrite?.estimate?.required)}`,
  ];
}

async function waitForCacheWriteSettled(page, label) {
  await waitForOrDump(
    page,
    label,
    () => {
      const state = window.__pilotCacheWrite;
      return Boolean(state?.completed);
    },
    null,
    cacheWriteTimeoutMs,
  );
  return await collectPageStatus(page);
}

async function collectPageStatus(page) {
  try {
    return await page.evaluate(() => {
      const text = (id) => document.getElementById(id)?.textContent || "";
      const button = document.getElementById("accent-button");
      const result = document.getElementById("result-output");
      return {
        url: location.href,
        ready: Boolean(window.__pilotRuntimeReady),
        crossOriginIsolated: window.crossOriginIsolated,
        modelStatus: text("model-status"),
        runStatus: text("run-status"),
        memoryLine: text("memory-status"),
        memory: window.__pilotMemoryStatus || null,
        loadMemory: window.__pilotLoadMemory || null,
        runtimeConfig: window.__pilotRuntimeConfig || null,
        sessionCreate: window.__pilotSessionCreate || null,
        cacheWrite: window.__pilotCacheWrite || null,
        quotaStress: window.__pilotQuotaStress || null,
        noWorkerHarness: Boolean(window.__pilotNoWorkerHarness),
        runPhase: window.__pilotRunPhase || null,
        lastRun: window.__pilotLastRun || null,
        buttonDisabled: button ? button.disabled : null,
        resultLength: result ? result.textContent.length : null,
        resultSnippet: result ? result.textContent.slice(0, 500) : "",
        performanceMemory: performance.memory
          ? {
              usedJSHeapSize: performance.memory.usedJSHeapSize,
              totalJSHeapSize: performance.memory.totalJSHeapSize,
              jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
            }
          : null,
      };
    });
  } catch (error) {
    return { evaluateError: error.message };
  }
}

async function dumpWaitFailure(page, label, error) {
  const status = await collectPageStatus(page);
  const screenshot = path.join(SCRIPT_DIR, `memtest-timeout-${phase}-${safeName(label)}.png`);
  console.log(`TIMEOUT phase=${phase} label=${label} error=${JSON.stringify(error.message)}`);
  console.log(`PAGE_STATUS phase=${phase} label=${label} ${compactStatus(status)}`);
  try {
    await page.screenshot({ path: screenshot, fullPage: true, timeout: 15000 });
    console.log(`SCREENSHOT phase=${phase} label=${label} path=${screenshot}`);
  } catch (screenshotError) {
    console.log(
      `SCREENSHOT_FAILED phase=${phase} label=${label} error=${JSON.stringify(
        screenshotError.message,
      )}`,
    );
  }
}

async function waitForOrDump(page, label, predicate, arg, timeout) {
  try {
    return await page.waitForFunction(predicate, arg, { timeout });
  } catch (error) {
    await dumpWaitFailure(page, label, error);
    throw error;
  }
}

async function readMeasurement(page) {
  return await page.evaluate(async () => {
    if (typeof window.gc === "function") {
      window.gc();
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    const memory = window.__pilotMemoryStatus || {};
    const perf = performance.memory || {};
    let userAgentSpecific = null;
    if (typeof performance.measureUserAgentSpecificMemory === "function") {
      try {
        userAgentSpecific = await performance.measureUserAgentSpecificMemory();
      } catch (error) {
        userAgentSpecific = { error: error.message };
      }
    }
    return {
      wasmBytes: memory.wasmBytes ?? null,
      wasmMaxBytes: memory.wasmMaxBytes ?? null,
      wasmMemoryCount: memory.wasmMemoryCount ?? null,
      wasmRatio: memory.wasmRatio ?? null,
      jsHeapBytes: perf.usedJSHeapSize ?? memory.jsHeapBytes ?? null,
      jsHeapLimitBytes: perf.jsHeapSizeLimit ?? memory.jsHeapLimitBytes ?? null,
      userAgentSpecificBytes: userAgentSpecific?.bytes ?? null,
      userAgentSpecific,
      statusText: document.getElementById("run-status")?.textContent || "",
      memoryText: document.getElementById("memory-status")?.textContent || "",
      lastRun: window.__pilotLastRun || null,
      loadMemory: window.__pilotLoadMemory || null,
      runtimeConfig: window.__pilotRuntimeConfig || null,
      sessionCreate: window.__pilotSessionCreate || null,
      cacheWrite: window.__pilotCacheWrite || null,
      quotaStress: window.__pilotQuotaStress || null,
      modelStatusText: document.getElementById("model-status")?.textContent || "",
    };
  });
}

async function waitForRuntimeReady(page) {
  await waitForOrDump(
    page,
    "runtime-ready",
    () =>
      Boolean(window.__pilotRuntimeReady) &&
      !document.getElementById("accent-button")?.disabled,
    null,
    runtimeTimeoutMs,
  );
}

async function logLoadMeasurement(page) {
  const measurement = await readMeasurement(page);
  const loadMemory = measurement.loadMemory || {};
  const beforeSession = loadMemory.beforeSession || {};
  const before = loadMemory.beforeRelease || {};
  const after = loadMemory.afterRelease || {};
  const beforeSessionJs = beforeSession.jsHeapBytes ?? null;
  const beforeJs = before.jsHeapBytes ?? null;
  const afterJs = after.jsHeapBytes ?? null;
  const deltaJs =
    beforeSessionJs == null || afterJs == null ? null : afterJs - beforeSessionJs;
  console.log(
    [
      `LOAD phase=${phase}`,
      `wasmMB=${mb(after.wasmBytes ?? measurement.wasmBytes)}`,
      `jsHeapBeforeSessionMB=${mb(beforeSessionJs)}`,
      `jsHeapBeforeReleaseMB=${mb(beforeJs)}`,
      `jsHeapAfterReleaseMB=${mb(afterJs)}`,
      `jsHeapDeltaMB=${mb(deltaJs)}`,
      `proxy=${measurement.runtimeConfig?.proxy ?? "n/a"}`,
      `mode=${measurement.runtimeConfig?.mode ?? "n/a"}`,
      `threads=${measurement.runtimeConfig?.threads ?? "n/a"}`,
      `fallback=${measurement.runtimeConfig?.proxyFallback ?? "n/a"}`,
      `wasmRuntimeFetchObserved=${measurement.runtimeConfig?.wasmRuntimeFetchObserved ?? "n/a"}`,
      `sessionCreateMs=${measurement.sessionCreate?.elapsedMs?.toFixed?.(1) ?? "n/a"}`,
      `cacheStatus=${measurement.cacheWrite?.status ?? "n/a"}`,
      `cacheAttempted=${measurement.cacheWrite?.attempted ?? "n/a"}`,
      `wasmPaths=${JSON.stringify(measurement.runtimeConfig?.wasmPaths ?? null)}`,
    ].join(" "),
  );
  return measurement;
}

async function runPilotText(page, text, label, iteration, total) {
  await page.evaluate((nextText) => {
    window.__pilotLastRun = null;
    const textarea = document.getElementById("source-text");
    textarea.value = nextText;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }, text);

  const startedAt = Date.now();
  await page.click("#accent-button", { timeout: 15000 });
  await waitForOrDump(
    page,
    label,
    () => {
      const lastRun = window.__pilotLastRun;
      const runStatus = document.getElementById("run-status")?.textContent || "";
      return Boolean(lastRun) || /^Error[: ]/.test(runStatus);
    },
    null,
    runTimeoutMs,
  );
  const measurement = await readMeasurement(page);
  measurement.iteration = iteration;
  measurement.wallMs = Date.now() - startedAt;

  console.log(
    [
      `ITER phase=${phase}`,
      `i=${iteration}/${total}`,
      `wasmMB=${mb(measurement.wasmBytes)}`,
      `wasmMaxMB=${mb(measurement.wasmMaxBytes)}`,
      `wasmCount=${measurement.wasmMemoryCount ?? "n/a"}`,
      `wasmRatio=${measurement.wasmRatio == null ? "n/a" : measurement.wasmRatio.toFixed(3)}`,
      `jsHeapMB=${mb(measurement.jsHeapBytes)}`,
      `uaSpecificMB=${mb(measurement.userAgentSpecificBytes)}`,
      `mode=${measurement.runtimeConfig?.mode ?? "n/a"}`,
      `baseBudget=${measurement.lastRun?.baseTokenBudget ?? "n/a"}`,
      `effectiveBudget=${measurement.lastRun?.effectiveTokenBudget ?? "n/a"}`,
      `wallMs=${measurement.wallMs}`,
      `runMs=${measurement.lastRun?.elapsedMs?.toFixed?.(1) ?? "n/a"}`,
      `tokensPerSecond=${measurement.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
      `status=${JSON.stringify(measurement.statusText)}`,
    ].join(" "),
  );

  if (measurement.lastRun?.memoryLimitReached || /^Error[: ]/.test(measurement.statusText)) {
    throw new Error(`Run failed during ${label}: ${measurement.statusText}`);
  }
  return measurement;
}

async function runColdBlockedCheck(page, blockedRequests) {
  const measurement = await runPilotText(page, REPRO_TEXT, "cold-blocked-run", 1, 1);
  const status = await collectPageStatus(page);
  const ok =
    Boolean(status.ready) &&
    Boolean(status.lastRun) &&
    !status.buttonDisabled &&
    status.resultLength > 20 &&
    blockedRequests.length === 0;
  console.log(
    [
      `COLD_BLOCKED phase=${phase}`,
      `ok=${ok}`,
      `blocked=${blockedRequests.length}`,
      `resultLength=${status.resultLength ?? "n/a"}`,
      `tokensPerSecond=${measurement.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(`Cold blocked check failed: ${compactStatus(status)}`);
  }
}

async function runNoWorkerCheck(page) {
  const measurement = await runPilotText(page, REPRO_TEXT, "no-worker-run", 1, 1);
  const status = await collectPageStatus(page);
  const sessionCreate = status.sessionCreate || {};
  const proxyAttempt = sessionCreate.attempts?.find((attempt) => attempt.mode === "worker");
  const fallbackWithinWatchdog =
    Number(proxyAttempt?.elapsedMs || Number.POSITIVE_INFINITY) <= PROXY_WATCHDOG_MS + 5000;
  const accentuated = hasStressMarks(status.resultSnippet);
  const ok =
    Boolean(status.ready) &&
    Boolean(status.lastRun) &&
    status.noWorkerHarness === true &&
    status.runtimeConfig?.mode === "main" &&
    status.runtimeConfig?.proxyFallback === true &&
    Boolean(proxyAttempt) &&
    ["rejected", "watchdog"].includes(proxyAttempt.status) &&
    fallbackWithinWatchdog &&
    !status.buttonDisabled &&
    status.resultLength > 20 &&
    accentuated;
  console.log(
    [
      `NO_WORKER phase=${phase}`,
      `ok=${ok}`,
      `mode=${status.runtimeConfig?.mode ?? "n/a"}`,
      `fallback=${status.runtimeConfig?.proxyFallback ?? "n/a"}`,
      `proxyAttemptStatus=${proxyAttempt?.status ?? "n/a"}`,
      `proxyAttemptMs=${proxyAttempt?.elapsedMs?.toFixed?.(1) ?? "n/a"}`,
      `fallbackWithinWatchdog=${fallbackWithinWatchdog}`,
      `accentuated=${accentuated}`,
      ...sessionWarningFields(sessionCreate),
      `resultLength=${status.resultLength ?? "n/a"}`,
      `tokensPerSecond=${measurement.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
      `memoryStatus=${JSON.stringify(status.memoryLine)}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(`No-worker fallback check failed: ${compactStatus(status)}`);
  }
}

async function runWorkerModeCheck(page, wasmRuntimeRequests) {
  const measurement = await runPilotText(page, REPRO_TEXT, "worker-mode-run", 1, 1);
  const status = await collectPageStatus(page);
  const appObserved = Boolean(status.runtimeConfig?.wasmRuntimeFetchObserved);
  const fetchObserved = appObserved || wasmRuntimeRequests.length > 0;
  const ok =
    Boolean(status.ready) &&
    Boolean(status.lastRun) &&
    status.runtimeConfig?.mode === "worker" &&
    status.runtimeConfig?.proxy === true &&
    fetchObserved &&
    !status.buttonDisabled &&
    status.resultLength > 20;
  console.log(
    [
      `WORKER_MODE phase=${phase}`,
      `ok=${ok}`,
      `mode=${status.runtimeConfig?.mode ?? "n/a"}`,
      `proxy=${status.runtimeConfig?.proxy ?? "n/a"}`,
      `wasmRuntimeFetches=${wasmRuntimeRequests.length}`,
      `appObserved=${appObserved}`,
      `accentuated=${hasStressMarks(status.resultSnippet)}`,
      ...sessionWarningFields(status.sessionCreate),
      `resultLength=${status.resultLength ?? "n/a"}`,
      `tokensPerSecond=${measurement.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
      `memoryStatus=${JSON.stringify(status.memoryLine)}`,
      `modelStatus=${JSON.stringify(status.modelStatus)}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(`Worker mode check failed: ${compactStatus(status)}`);
  }
}

async function runNormalCacheCheck(page) {
  let measurement = null;
  for (let i = 1; i <= iterations; i += 1) {
    measurement = await runPilotText(page, REPRO_TEXT, "normal-cache-run", i, iterations);
  }

  const status = await waitForCacheWriteSettled(page, "normal-cache-write");
  const cacheWrite = status.cacheWrite;
  const ok =
    Boolean(status.ready) &&
    Boolean(status.lastRun) &&
    !status.buttonDisabled &&
    status.resultLength > 20 &&
    cacheWrite?.attempted === true &&
    cacheWrite?.completed === true &&
    cacheWrite?.status === "stored";
  console.log(
    [
      `NORMAL_CACHE phase=${phase}`,
      `ok=${ok}`,
      ...cacheWriteFields(cacheWrite),
      `resultLength=${status.resultLength ?? "n/a"}`,
      `tokensPerSecond=${measurement?.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(`Normal cache check failed: ${compactStatus(status)}`);
  }
}

async function runQuotaStressCheck(page) {
  let measurement = null;
  for (let i = 1; i <= iterations; i += 1) {
    measurement = await runPilotText(page, REPRO_TEXT, "quota-stress-run", i, iterations);
  }

  let status = await collectPageStatus(page);
  if (status.cacheWrite?.attempted && !status.cacheWrite.completed) {
    status = await waitForCacheWriteSettled(page, "quota-stress-cache-write");
  }

  const cacheWrite = status.cacheWrite;
  const quotaStress = status.quotaStress || {};
  const preflightSkipOk =
    cacheWrite?.status === "unavailable" &&
    cacheWrite?.skipped === true &&
    cacheWrite?.attempted === false &&
    Number(quotaStress.putCalls || 0) === 0;
  const watchdogOk =
    cacheWrite?.status === "failed" &&
    cacheWrite?.watchdogFired === true &&
    cacheWrite?.attempted === true &&
    Number(quotaStress.putCalls || 0) >= 1;
  const ok =
    Boolean(status.ready) &&
    Boolean(status.lastRun) &&
    !status.buttonDisabled &&
    status.resultLength > 20 &&
    (preflightSkipOk || watchdogOk);
  console.log(
    [
      `QUOTA_STRESS phase=${phase}`,
      `ok=${ok}`,
      `mode=${preflightSkipOk ? "preflight-skip" : watchdogOk ? "watchdog-cancel" : "unexpected"}`,
      ...cacheWriteFields(cacheWrite),
      `mockOpenCalls=${quotaStress.openCalls ?? "n/a"}`,
      `mockMatchCalls=${quotaStress.matchCalls ?? "n/a"}`,
      `mockPutCalls=${quotaStress.putCalls ?? "n/a"}`,
      `resultLength=${status.resultLength ?? "n/a"}`,
      `tokensPerSecond=${measurement?.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(`Quota stress check failed: ${compactStatus(status)}`);
  }
}

async function runResponsivenessCheck(page) {
  await page.evaluate((text) => {
    window.__pilotLastRun = null;
    const budget = document.getElementById("token-budget");
    budget.value = "256";
    budget.dispatchEvent(new Event("input", { bubbles: true }));
    const textarea = document.getElementById("source-text");
    textarea.value = text;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  }, RESPONSIVENESS_TEXT);

  const startedAt = Date.now();
  await page.click("#accent-button", { timeout: 15000 });
  await waitForOrDump(
    page,
    "responsiveness-session",
    () => window.__pilotRunPhase === "session",
    null,
    runTimeoutMs,
  );

  const interaction = await page.evaluate(async () => {
    const button = document.querySelector('button[data-lang="en"]');
    const before = performance.now();
    button.click();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await new Promise((resolve) => requestAnimationFrame(resolve));
    return {
      elapsedMs: performance.now() - before,
      lang: document.documentElement.lang,
      ariaPressed: button.getAttribute("aria-pressed"),
      runPhase: window.__pilotRunPhase || null,
    };
  });
  console.log(
    [
      `RESPONSIVENESS phase=${phase}`,
      `interactionMs=${interaction.elapsedMs.toFixed(1)}`,
      `lang=${interaction.lang}`,
      `ariaPressed=${interaction.ariaPressed}`,
      `runPhase=${interaction.runPhase}`,
    ].join(" "),
  );
  if (
    interaction.elapsedMs >= 200 ||
    interaction.lang !== "en" ||
    interaction.ariaPressed !== "true"
  ) {
    throw new Error(`Responsiveness check failed: ${JSON.stringify(interaction)}`);
  }

  await waitForOrDump(
    page,
    "responsiveness-run",
    () => {
      const lastRun = window.__pilotLastRun;
      const runStatus = document.getElementById("run-status")?.textContent || "";
      return Boolean(lastRun) || /^Error[: ]/.test(runStatus);
    },
    null,
    runTimeoutMs,
  );
  const measurement = await readMeasurement(page);
  measurement.iteration = 1;
  measurement.wallMs = Date.now() - startedAt;
  console.log(
    [
      `ITER phase=${phase}`,
      "i=1/1",
      `wasmMB=${mb(measurement.wasmBytes)}`,
      `jsHeapMB=${mb(measurement.jsHeapBytes)}`,
      `wallMs=${measurement.wallMs}`,
      `runMs=${measurement.lastRun?.elapsedMs?.toFixed?.(1) ?? "n/a"}`,
      `tokensPerSecond=${measurement.lastRun?.tokensPerSecond?.toFixed?.(1) ?? "n/a"}`,
      `status=${JSON.stringify(measurement.statusText)}`,
    ].join(" "),
  );
  if (measurement.lastRun?.memoryLimitReached || /^Error[: ]/.test(measurement.statusText)) {
    throw new Error(`Responsiveness run failed: ${measurement.statusText}`);
  }
}

function assertSpec42Flat(curve) {
  const wasmRange = rangeMb(curve.map((item) => item.wasmBytes));
  const jsHeapRange = rangeMb(curve.map((item) => item.jsHeapBytes));
  const status = curve[curve.length - 1];
  const expectedMode = noWorkerMode ? "main" : "worker";
  const modeOk = status?.runtimeConfig?.mode === expectedMode;
  const wasmOk = wasmRange == null || wasmRange <= WASM_FLAT_TOLERANCE_MB;
  const jsHeapOk = jsHeapRange == null || jsHeapRange <= JS_HEAP_FLAT_TOLERANCE_MB;
  const runOk = curve.every(
    (item) => !item.lastRun?.memoryLimitReached && !/^Error[: ]/.test(item.statusText),
  );
  const ok = modeOk && wasmOk && jsHeapOk && runOk;
  console.log(
    [
      `SPEC42_FLAT phase=${phase}`,
      `ok=${ok}`,
      `mode=${status?.runtimeConfig?.mode ?? "n/a"}`,
      `expectedMode=${expectedMode}`,
      `iterations=${curve.length}`,
      `wasmRangeMB=${fmtRange(wasmRange)}`,
      `wasmToleranceMB=${WASM_FLAT_TOLERANCE_MB}`,
      `jsHeapRangeMB=${fmtRange(jsHeapRange)}`,
      `jsHeapToleranceMB=${JS_HEAP_FLAT_TOLERANCE_MB}`,
    ].join(" "),
  );
  if (!ok) {
    throw new Error(
      `SPEC42 flat check failed: modeOk=${modeOk} wasmOk=${wasmOk} jsHeapOk=${jsHeapOk} runOk=${runOk}`,
    );
  }
}

async function main() {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error(`Invalid iteration count: ${process.argv[3]}`);
  }

  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--js-flags=--expose-gc",
      "--enable-precise-memory-info",
      "--disable-dev-shm-usage",
      "--disk-cache-size=2147483647",
      "--unlimited-storage",
      "--no-sandbox",
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
  });
  const wasmRuntimeRequests = [];
  context.on("request", (request) => {
    if (isWasmRuntimeRequest(request.url())) {
      wasmRuntimeRequests.push(request.url());
    }
  });
  if (phaseKind === "quota-stress") {
    await installQuotaStressMock(context);
  }
  const page = await context.newPage();
  page.on("console", (message) => {
    recentConsole.push(`${message.type()}: ${message.text()}`);
    if (recentConsole.length > 20) {
      recentConsole.shift();
    }
  });
  page.on("pageerror", (error) => {
    recentConsole.push(`pageerror: ${error.message}`);
    if (recentConsole.length > 20) {
      recentConsole.shift();
    }
  });

  let blockedRequests = [];

  try {
    if (noWorkerMode) {
      await installNoWorkerMock(page);
    }
    if (phaseKind === "cold-blocked") {
      blockedRequests = await installLocalhostOnlyRoutes(page);
    }
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: runtimeTimeoutMs });
    await waitForRuntimeReady(page);
    await logLoadMeasurement(page);

    if (phaseKind === "cold-blocked") {
      await runColdBlockedCheck(page, blockedRequests);
      return;
    }

    if (phaseKind === "no-worker") {
      await runNoWorkerCheck(page);
      return;
    }

    if (phaseKind === "worker-mode") {
      await runWorkerModeCheck(page, wasmRuntimeRequests);
      return;
    }

    if (phaseKind === "normal-cache") {
      await runNormalCacheCheck(page);
      return;
    }

    if (phaseKind === "quota-stress") {
      await runQuotaStressCheck(page);
      return;
    }

    if (phaseKind === "responsive") {
      await runResponsivenessCheck(page);
      return;
    }

    const curve = [];
    for (let i = 1; i <= iterations; i += 1) {
      const measurement = await runPilotText(
        page,
        REPRO_TEXT,
        `iteration-${i}`,
        i,
        iterations,
      );
      curve.push(measurement);
    }

    console.log(
      `CURVE_JSON phase=${phase} ${JSON.stringify(
        curve.map((item) => ({
          i: item.iteration,
          wasmMB: item.wasmBytes == null ? null : Number(mb(item.wasmBytes)),
          wasmMaxMB: item.wasmMaxBytes == null ? null : Number(mb(item.wasmMaxBytes)),
          wasmCount: item.wasmMemoryCount,
          wasmRatio: item.wasmRatio,
          jsHeapMB: item.jsHeapBytes == null ? null : Number(mb(item.jsHeapBytes)),
          uaSpecificMB:
            item.userAgentSpecificBytes == null ? null : Number(mb(item.userAgentSpecificBytes)),
          baseBudget: item.lastRun?.baseTokenBudget ?? null,
          effectiveBudget: item.lastRun?.effectiveTokenBudget ?? null,
          wallMs: item.wallMs,
          runMs: item.lastRun?.elapsedMs ?? null,
          tokensPerSecond: item.lastRun?.tokensPerSecond ?? null,
          memoryLimitReached: Boolean(item.lastRun?.memoryLimitReached),
          status: item.statusText,
        })),
      )}`,
    );
    if (phaseKind === "spec42-flat") {
      assertSpec42Flat(curve);
    }
  } catch (error) {
    console.log(`RECENT_CONSOLE phase=${phase} ${JSON.stringify(recentConsole)}`);
    throw error;
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
