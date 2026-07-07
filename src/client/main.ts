import type {
  AccentResponse,
  ErrorResponse,
  Part,
  Variant,
  WordResponse,
} from "../shared/types";
import { parseMi, scoreTags } from "../shared/tags";
import { formatProbability } from "./format";
import {
  type SpellcheckContext,
  type SpellcheckSuggestion,
} from "./spellcheck";
import { suggestBatch } from "./spellcheckClient";
import {
  buildEditedSentenceSpans,
  rebuildRenderedPartsWithFragments,
  retileRenderedParts,
  tokenizeForPreview,
  type RenderedPartCore,
  type TextEdit,
} from "./preview";
import {
  formatBytes,
  hasCachedLocalModel,
  loadModelTierInfo,
  LOCAL_DEFAULT_MODEL_TIER,
  LOCAL_MODEL_SIZE_FALLBACK,
  purgeCachedLocalModel,
  updateToCurrentModel,
  type LocalModelTierInfo,
} from "./local/assets";
import {
  createLocalDownloadGate,
  type LocalDownloadGateState,
} from "./local/consent";
import type { LocalAccentEngine } from "./local/engine";
import type {
  CacheStatus,
  ExecutionMode,
  LocalModelTier,
  LocalModelStatus,
  LocalRunStatus,
  LocalModelUpdateInfo,
  LocalStats,
} from "./local/types";
import {
  detectLang,
  LANGS,
  parallelMorphologyLines,
  UI,
  type Lang,
  type UiStrings,
} from "./i18n";
import "./style.css";

const MAX_TEXT_LENGTH = 20_000;
const MODE_STORAGE_KEY = "accent-mode";
const DISPLAY_STORAGE_KEY = "accent-display";
const LOCAL_TIER_STORAGE_KEY = "accent-local-tier";
const LOCAL_READY_TIERS_STORAGE_KEY = "accent-local-ready-tiers-v1";
const DETAILS_COLLAPSED_STORAGE_KEY = "accent-details-collapsed";
const PREVIEW_SPELLCHECK_DEBOUNCE_MS = 600;
const VLKK_PRIMER_URL =
  "https://www.vlkk.lt/aktualiausios-temos/tartis-ir-kirciavimas";
const FOCUSABLE_SELECTOR =
  'a[href], button:not(:disabled), textarea:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])';
const PRIMER_MIXED_WORDS = getPrimerMixedWords();
const PRIMER_PAIR_WORDS = getPrimerPairWords();

type MessageKey = Extract<
  keyof UiStrings,
  "errEmpty" | "errTooLong" | "errUpstream" | "errUnexpected"
>;

type RenderedPart = RenderedPartCore & {
  spelling?: SpellcheckSuggestion;
  userChosen?: boolean;
};

type AccentMode = "web" | "local";
type DisplayMode = "top" | "all";

class AccentRequestError extends Error {
  constructor(readonly status: number) {
    super(`Accent request failed with status ${status}`);
  }
}

const languageSwitcher = getElement<HTMLDivElement>("language-switcher");
const languageButtons = Array.from(
  languageSwitcher.querySelectorAll<HTMLButtonElement>("button[data-lang]"),
);
const heroTagline = getElement<HTMLParagraphElement>("hero-tagline");
const form = getElement<HTMLFormElement>("accent-form");
const inputLabel = getElement<HTMLLabelElement>("input-label");
const textarea = getElement<HTMLTextAreaElement>("source-text");
const sourceOverlay = getElement<HTMLDivElement>("source-overlay");
const charCounter = getElement<HTMLSpanElement>("char-counter");
const modeLabel = getElement<HTMLSpanElement>("mode-label");
const modeSwitch = getElement<HTMLSpanElement>("mode-switch");
const modeButtons = Array.from(
  modeSwitch.querySelectorAll<HTMLButtonElement>("button[data-mode]"),
);
const modeExplainer = getElement<HTMLParagraphElement>("mode-explainer");
const localTierControls = getElement<HTMLDivElement>("local-tier-controls");
const localTierLabel = getElement<HTMLSpanElement>("local-tier-label");
const localTierSwitch = getElement<HTMLSpanElement>("local-tier-switch");
const localTierButtons = Array.from(
  localTierSwitch.querySelectorAll<HTMLButtonElement>("button[data-tier]"),
);
const localStatusLine = getElement<HTMLDivElement>("local-status");
const localUpdateControl = getElement<HTMLDivElement>("local-update");
const panelExtras = getElement<HTMLDivElement>("panel-extras");
const detailsToggle = getElement<HTMLButtonElement>("details-toggle");
const inputActions = form.querySelector<HTMLDivElement>(".input-actions")!;
const fixAllButton = getElement<HTMLButtonElement>("fix-all-button");
const accentButton = getElement<HTMLButtonElement>("accent-button");
const copyButton = getElement<HTMLButtonElement>("copy-button");
const message = getElement<HTMLParagraphElement>("form-message");
const resultHeading = getElement<HTMLHeadingElement>("result-heading");
const displayLabel = getElement<HTMLSpanElement>("display-label");
const displaySwitch = getElement<HTMLSpanElement>("display-switch");
const displayButtons = Array.from(
  displaySwitch.querySelectorAll<HTMLButtonElement>("button[data-display]"),
);
const localStatsButton = getElement<HTMLButtonElement>("local-stats-button");
const resultOutput = getElement<HTMLDivElement>("result-output");
const taggerNotice = getElement<HTMLDivElement>("tagger-notice");
const taggerNoticeText = getElement<HTMLSpanElement>("tagger-notice-text");
const taggerNoticeClose = getElement<HTMLButtonElement>("tagger-notice-close");
const legend = getElement<HTMLDivElement>("legend");
const legendLabel = getElement<HTMLSpanElement>("legend-label");
const legendResolved = getElement<HTMLSpanElement>("legend-resolved");
const legendAmbiguous = getElement<HTMLSpanElement>("legend-ambiguous");
const legendUser = getElement<HTMLSpanElement>("legend-user");
const legendUnknown = getElement<HTMLSpanElement>("legend-unknown");
const inputLegend = getElement<HTMLDivElement>("input-legend");
const legendAutofix = getElement<HTMLSpanElement>("legend-autofix");
const legendClickfix = getElement<HTMLSpanElement>("legend-clickfix");
const primerLink = getElement<HTMLButtonElement>("primer-link");
const primerBackdrop = getElement<HTMLDivElement>("primer-backdrop");
const primerDialog = getElement<HTMLDivElement>("primer-dialog");
const primerClose = getElement<HTMLButtonElement>("primer-close");
const primerTitle = getElement<HTMLHeadingElement>("primer-title");
const primerIntro = getElement<HTMLParagraphElement>("primer-intro");
const primerGraveName = getElement<HTMLHeadingElement>("primer-grave-name");
const primerGraveDesc = getElement<HTMLParagraphElement>("primer-grave-desc");
const primerGraveEx = getElement<HTMLSpanElement>("primer-grave-ex");
const primerAcuteName = getElement<HTMLHeadingElement>("primer-acute-name");
const primerAcuteDesc = getElement<HTMLParagraphElement>("primer-acute-desc");
const primerAcuteEx = getElement<HTMLSpanElement>("primer-acute-ex");
const primerTildeName = getElement<HTMLHeadingElement>("primer-tilde-name");
const primerTildeDesc = getElement<HTMLParagraphElement>("primer-tilde-desc");
const primerTildeEx = getElement<HTMLSpanElement>("primer-tilde-ex");
const primerMixed = getElement<HTMLParagraphElement>("primer-mixed");
const primerPair = getElement<HTMLParagraphElement>("primer-pair");
const primerMore = getElement<HTMLAnchorElement>("primer-more");
const siteFooter = getElement<HTMLElement>("site-footer");
const metaDescription = document.querySelector<HTMLMetaElement>(
  'meta[name="description"]',
);

let lang: Lang = detectLang();
let accentMode: AccentMode = readStoredMode();
let displayMode: DisplayMode = readStoredDisplayMode();
let localTier: LocalModelTier = readStoredLocalTier();
let localTierInfo: Partial<Record<LocalModelTier, LocalModelTierInfo>> = {};
let renderedParts: RenderedPart[] = [];
let leftTokens: RenderedPart[] = [];
let renderedSourceText = "";
let activePopover: HTMLDivElement | null = null;
let activeStatsPopover: HTMLDivElement | null = null;
let isLoading = false;
let spellcheckRequestId = 0;
let leftSpellcheckRequestId = 0;
let leftSpellcheckTimer: number | undefined;
let messageKey: MessageKey | null = null;
let copyResetTimer: number | undefined;
let copied = false;
let localEngine: LocalAccentEngine | null = null;
let localEnginePromise: Promise<LocalAccentEngine> | null = null;
let localEngineTier: LocalModelTier | null = null;
let localEngineRequestId = 0;
let localModelStatus: LocalModelStatus = { type: "idle" };
let localRunStatus: LocalRunStatus = { type: "ready" };
let localStats: LocalStats | null = null;
let localExpectedBytes: number = LOCAL_MODEL_SIZE_FALLBACK;
let localDownloadGateState: LocalDownloadGateState = "inactive";
let isLocalModelUpdating = false;
let detailsCollapsed = localStorage.getItem(DETAILS_COLLAPSED_STORAGE_KEY) === "true";

const localDownloadGate = createLocalDownloadGate({
  hasCachedModel: () => hasCachedLocalModel(localTier),
  ensureEngine: ensureLocalEngine,
  isEngineReady: () => Boolean(localEngine && localEngineTier === localTier),
  wasPreviouslyReady: () => wasLocalTierPreviouslyReady(localTier),
  markReady: () => markLocalTierReady(localTier),
  onState: (state) => {
    localDownloadGateState = state;
    renderUi();
  },
});

languageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextLang = parseLang(button.dataset.lang);
    if (nextLang) {
      setLanguage(nextLang);
    }
  });
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextMode = parseMode(button.dataset.mode);
    if (nextMode) {
      void setAccentMode(nextMode);
    }
  });
});

localTierButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextTier = parseLocalTier(button.dataset.tier);
    if (nextTier) {
      void setLocalTier(nextTier);
    }
  });
});

displayButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextDisplay = parseDisplayMode(button.dataset.display);
    if (nextDisplay && accentMode === "web") {
      setDisplayMode(nextDisplay);
    }
  });
});

textarea.addEventListener("input", () => {
  syncBoxHeights();
  updateCounter();
  updateFixAllButtonState();
  primeLeftOverlayForCurrentText();
  scheduleLeftSpellcheck();
});

textarea.addEventListener("paste", () => {
  requestAnimationFrame(() => {
    window.clearTimeout(leftSpellcheckTimer);
    leftSpellcheckTimer = undefined;
    void runLeftSpellcheck();
  });
});

// Keep the two boxes the same height when the viewport (and thus wrapping)
// changes; rAF-coalesced so a resize drag doesn't thrash layout.
let resizeRaf = 0;
window.addEventListener("resize", () => {
  if (resizeRaf) {
    return;
  }
  resizeRaf = requestAnimationFrame(() => {
    resizeRaf = 0;
    syncBoxHeights();
  });
});

textarea.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

detailsToggle.addEventListener("click", () => {
  detailsCollapsed = !detailsCollapsed;
  localStorage.setItem(DETAILS_COLLAPSED_STORAGE_KEY, String(detailsCollapsed));
  renderDetailsToggle();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitText();
});

fixAllButton.addEventListener("click", () => {
  void fixAllRestores();
});

copyButton.addEventListener("click", () => {
  void copyResult();
});

localStatsButton.addEventListener("click", (event) => {
  event.stopPropagation();
  toggleStatsPopover();
});

taggerNoticeClose.addEventListener("click", () => {
  taggerNotice.hidden = true;
});

primerLink.addEventListener("click", () => {
  openPrimer();
});

primerClose.addEventListener("click", () => {
  closePrimer();
});

primerBackdrop.addEventListener("click", (event) => {
  if (event.target === primerBackdrop) {
    closePrimer();
  }
});

primerDialog.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    trapPrimerFocus(event);
  }
});

// Keep the input and the result scrolled together. The two boxes hold
// character-parallel text at identical line positions and are locked to the
// same height, so mirror the scroll offset 1:1 — that glues line N to line N
// the whole way down. (Proportional mapping drifts a few px mid-scroll whenever
// the two scroll heights differ even slightly.) The induced scroll event on the
// synced element is swallowed via the ignore set (no timers — they stall in
// background tabs).
const scrollIgnore = new Set<HTMLElement>();

function syncScroll(source: HTMLElement, target: HTMLElement): void {
  if (scrollIgnore.delete(source)) {
    return;
  }

  let changed = false;
  const targetMaxTop = target.scrollHeight - target.clientHeight;
  const nextTop = targetMaxTop > 0 ? Math.min(source.scrollTop, targetMaxTop) : 0;
  if (Math.abs(target.scrollTop - nextTop) >= 1) {
    changed = true;
  }

  const targetMaxLeft = target.scrollWidth - target.clientWidth;
  const nextLeft =
    targetMaxLeft > 0 ? Math.min(source.scrollLeft, targetMaxLeft) : 0;
  if (Math.abs(target.scrollLeft - nextLeft) >= 1) {
    changed = true;
  }

  if (!changed) {
    return;
  }
  scrollIgnore.add(target);
  target.scrollTop = nextTop;
  target.scrollLeft = nextLeft;
}

function syncLeftOverlayScroll(): void {
  sourceOverlay.scrollTop = textarea.scrollTop;
  sourceOverlay.scrollLeft = textarea.scrollLeft;
}

textarea.addEventListener("scroll", () => {
  syncLeftOverlayScroll();
  syncScroll(textarea, resultOutput);
});
resultOutput.addEventListener("scroll", () => {
  syncScroll(resultOutput, textarea);
  syncLeftOverlayScroll();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (isPrimerOpen()) {
      event.preventDefault();
      closePrimer();
      return;
    }

    closePopover();
    closeStatsPopover();
  }
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (
    target instanceof Node &&
    activeStatsPopover &&
    !activeStatsPopover.contains(target) &&
    !(target instanceof HTMLElement && target.closest("#local-stats-button"))
  ) {
    closeStatsPopover();
  }

  if (
    target instanceof Node &&
    activePopover &&
    !activePopover.contains(target) &&
    !(
      target instanceof HTMLElement &&
      target.closest(
        ".token-ambiguous, .token-plain, .token-correctable, .spell-underline",
      )
    )
  ) {
    closePopover();
  }
});

setLanguage(lang, { persist: false });
syncBoxHeights();
updateCounter();
if (accentMode === "local") {
  void enterLocalMode();
}

function scheduleLeftSpellcheck(): void {
  window.clearTimeout(leftSpellcheckTimer);
  leftSpellcheckTimer = window.setTimeout(() => {
    leftSpellcheckTimer = undefined;
    void runLeftSpellcheck();
  }, PREVIEW_SPELLCHECK_DEBOUNCE_MS);
}

async function runLeftSpellcheck(): Promise<void> {
  window.clearTimeout(leftSpellcheckTimer);
  leftSpellcheckTimer = undefined;

  const requestId = ++leftSpellcheckRequestId;
  const text = textarea.value;
  closePopover();
  closeStatsPopover();

  if (text.trim().length === 0) {
    leftTokens = [];
    renderLeftOverlay();
    return;
  }

  const tokens = tokenizeForPreview(text) as RenderedPart[];
  leftTokens = tokens;
  renderLeftOverlay();

  const targets = tokens
    .map((part, index) => ({ index, part }))
    .filter((item) => item.part.type === "word" && !item.part.numeralFragment);
  const words = targets.map(({ index, part }) => {
    const context = spellingContextForIndex(index, tokens);
    return { word: part.text, prev: context?.prev, next: context?.next };
  });

  let suggestions: SpellcheckSuggestion[] = [];
  try {
    suggestions = await suggestBatch(words);
  } catch {
    suggestions = [];
  }

  if (requestId !== leftSpellcheckRequestId || textarea.value !== text) {
    return;
  }

  const suggestionsByIndex = new Map(
    targets.map(({ index }, position) => [
      index,
      suggestions[position] ??
        ({ status: "unknown", candidates: [] } satisfies SpellcheckSuggestion),
    ]),
  );
  leftTokens = tokens.map((part, index) => {
    const spelling = suggestionsByIndex.get(index);
    if (!spelling) {
      return part;
    }

    return {
      ...part,
      spelling,
    };
  });
  renderLeftOverlay();
}

async function submitText(): Promise<void> {
  spellcheckRequestId += 1;
  const text = textarea.value;
  closePopover();
  closeStatsPopover();
  setMessage(null);

  if (text.trim().length === 0) {
    setMessage("errEmpty");
    return;
  }

  if (text.length > MAX_TEXT_LENGTH) {
    setMessage("errTooLong");
    return;
  }

  setLoading(true);

  try {
    const payload =
      accentMode === "local" ? await accentTextLocal(text) : await accentTextWeb(text);
    applyAccentResponse(payload);
  } catch (error) {
    renderedParts = [];
    renderedSourceText = "";
    spellcheckRequestId += 1;
    showTaggerNotice(false);
    renderResult();
    copyButton.disabled = true;
    setMessage(
      error instanceof AccentRequestError
        ? getMessageKeyForStatus(error.status)
        : "errUnexpected",
    );
  } finally {
    setLoading(false);
  }
}

const UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process";
const UDPIPE_MODEL = "lithuanian-alksnis";
const UDPIPE_TIMEOUT_MS = 10_000;

// Tag the text against UDPipe directly from the browser, so the request
// egresses from the user's own IP rather than the Worker's shared one. Returns
// the raw CoNLL-U for the Worker to align; on any failure returns undefined and
// the Worker falls back to a server-side UDPipe call. This is a CORS "simple
// request" (form-encoded + safelisted headers), so it needs no preflight.
async function fetchUdpipeTags(text: string): Promise<string | undefined> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), UDPIPE_TIMEOUT_MS);
  try {
    const response = await fetch(UDPIPE_URL, {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        accept: "application/json",
      },
      body: new URLSearchParams({
        tokenizer: "",
        tagger: "",
        model: UDPIPE_MODEL,
        data: text,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      return undefined;
    }
    const payload = (await response.json().catch(() => null)) as { result?: unknown } | null;
    return typeof payload?.result === "string" ? payload.result : undefined;
  } catch {
    return undefined;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function accentTextWeb(text: string): Promise<AccentResponse> {
  const tags = await fetchUdpipeTags(text);
  const response = await fetch("/api/accent", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(tags ? { text, tags } : { text }),
  });

  const payload = (await response.json().catch(() => null)) as
    | AccentResponse
    | ErrorResponse
    | null;
  if (!response.ok || !payload || "error" in payload) {
    throw new AccentRequestError(response.status);
  }

  return payload;
}

async function accentTextLocal(text: string): Promise<AccentResponse> {
  if (
    !localEngine &&
    localDownloadGateState !== "loading" &&
    localDownloadGateState !== "ready"
  ) {
    throw new Error("Local model download has not been approved.");
  }

  const engine = await ensureLocalEngine();
  const result = await engine.accent(text, (status) => {
    localRunStatus = status;
    renderLocalStatus();
  });
  localStats = result.stats;
  return {
    source: "local",
    tagger: "ok",
    parts: result.parts,
  };
}

function applyAccentResponse(payload: AccentResponse): void {
  renderedSourceText = textarea.value;
  spellcheckRequestId += 1;
  renderedParts = retileRenderedParts(payload.parts) as RenderedPart[];

  showTaggerNotice(payload.tagger === "unavailable");
  renderResult();
  void annotateUnknownWordsWithSpellcheck();
  updateCopyButtonState();
}

async function setAccentMode(nextMode: AccentMode): Promise<void> {
  if (accentMode === nextMode) {
    return;
  }

  accentMode = nextMode;
  localStorage.setItem(MODE_STORAGE_KEY, accentMode);
  closePopover();
  closeStatsPopover();
  renderUi();

  if (accentMode === "local") {
    await enterLocalMode();
  } else {
    localDownloadGate.leaveLocalMode();
  }
}

async function enterLocalMode(): Promise<void> {
  try {
    await refreshLocalTierInfo();
    await localDownloadGate.enterLocalMode();
  } catch {
    setMessage("errUnexpected");
  }
}

async function consentToLocalDownload(): Promise<void> {
  try {
    await refreshLocalTierInfo();
    if (localDownloadGateState === "needs-redownload") {
      await disposeLocalEngine();
    }
    await localDownloadGate.consentToDownload();
  } catch {
    setMessage("errUnexpected");
  }
}

async function setLocalTier(nextTier: LocalModelTier): Promise<void> {
  if (localTier === nextTier) {
    return;
  }

  localTier = nextTier;
  localStorage.setItem(LOCAL_TIER_STORAGE_KEY, localTier);
  closeStatsPopover();
  await disposeLocalEngine();
  await refreshLocalTierInfo();
  renderUi();

  if (accentMode === "local") {
    await enterLocalMode();
  }
}

async function forceRedownloadLocalModel(): Promise<void> {
  if (accentMode !== "local") {
    return;
  }

  closeStatsPopover();
  setMessage(null);
  try {
    await refreshLocalTierInfo();
    await disposeLocalEngine();
    await purgeCachedLocalModel(localTier);
    markLocalTierReady(localTier);
    await localDownloadGate.consentToDownload();
  } catch {
    setMessage("errUnexpected");
  }
}

async function updateCurrentLocalModel(): Promise<void> {
  if (
    accentMode !== "local" ||
    isLocalModelUpdating ||
    !currentLocalUpdateInfo()
  ) {
    return;
  }

  closeStatsPopover();
  setMessage(null);
  isLocalModelUpdating = true;
  localRunStatus = { type: "ready" };
  renderUi();

  const requestedTier = localTier;
  try {
    const info = await updateToCurrentModel(requestedTier, (status) => {
      localModelStatus = status;
      if ("expectedBytes" in status && status.expectedBytes) {
        localExpectedBytes = status.expectedBytes;
      }
      renderUi();
    });
    localTierInfo = {
      ...localTierInfo,
      [requestedTier]: info,
    };

    if (requestedTier !== localTier) {
      return;
    }

    await disposeLocalEngine();
    markLocalTierReady(requestedTier);
    await localDownloadGate.consentToDownload();
  } catch (error) {
    localModelStatus = { type: "failed", message: errorMessage(error) };
    setMessage("errUnexpected");
  } finally {
    isLocalModelUpdating = false;
    renderUi();
  }
}

function setDisplayMode(nextDisplay: DisplayMode): void {
  displayMode = nextDisplay;
  localStorage.setItem(DISPLAY_STORAGE_KEY, displayMode);
  closePopover();
  renderUi();
}

async function refreshLocalTierInfo(): Promise<void> {
  const [light, heavy] = await Promise.all([
    loadModelTierInfo("light"),
    loadModelTierInfo("heavy"),
  ]);
  const info = localTier === "light" ? light : heavy;
  localTierInfo = {
    light,
    heavy,
  };
  localExpectedBytes = info.bytes ?? LOCAL_MODEL_SIZE_FALLBACK;
  renderUi();
}

async function disposeLocalEngine(): Promise<void> {
  localEngineRequestId += 1;
  const engine = localEngine;
  localEngine = null;
  localEnginePromise = null;
  localEngineTier = null;
  localStats = null;
  localModelStatus = { type: "idle" };
  localRunStatus = { type: "ready" };
  window.__localAccentReady = false;
  window.__localAccentStats = undefined;
  await engine?.dispose();
}

async function ensureLocalEngine(): Promise<LocalAccentEngine> {
  if (localEngine && localEngineTier === localTier) {
    return localEngine;
  }

  if (localEnginePromise && localEngineTier === localTier) {
    return localEnginePromise;
  }

  await disposeLocalEngine();
  const requestedTier = localTier;
  const requestId = ++localEngineRequestId;
  localEngineTier = requestedTier;
  localRunStatus = { type: "ready" };
  localEnginePromise = import("./local/engine")
    .then(({ LocalAccentEngine: Engine }) =>
      Engine.create(requestedTier, (status) => {
        localModelStatus = status;
        if ("expectedBytes" in status && status.expectedBytes) {
          localExpectedBytes = status.expectedBytes;
        }
        if ("bytes" in status && status.bytes) {
          localExpectedBytes = status.bytes;
        }
        renderUi();
      }),
    )
    .then((engine) => {
      if (requestId !== localEngineRequestId || requestedTier !== localTier) {
        void engine.dispose();
        throw new Error("Local model tier changed while loading.");
      }
      localEngine = engine;
      localEngineTier = requestedTier;
      localStats = engine.getStats();
      localModelStatus =
        localModelStatus.type === "ready" ? localModelStatus : { type: "idle" };
      renderUi();
      return engine;
    })
    .catch((error: unknown) => {
      localEnginePromise = null;
      localEngineTier = null;
      localModelStatus = { type: "failed", message: errorMessage(error) };
      renderUi();
      throw error;
    });

  renderUi();
  return localEnginePromise;
}

function primeLeftOverlayForCurrentText(): void {
  leftSpellcheckRequestId += 1;
  const text = textarea.value;
  leftTokens =
    text.trim().length === 0 ? [] : (tokenizeForPreview(text) as RenderedPart[]);
  renderLeftOverlay();
}

function renderLeftOverlay(): void {
  sourceOverlay.replaceChildren();
  // Fix-all reflects the left overlay's autofixable restores — refresh it here so
  // the button lights up as soon as spellcheck annotates the tokens (before any
  // accentuation), covering every early-return path below.
  updateFixAllButtonState();

  if (
    leftTokens.length === 0 ||
    leftTokens.map((part) => part.text).join("") !== textarea.value
  ) {
    inputLegend.hidden = true;
    syncLeftOverlayScroll();
    return;
  }

  let underlineCount = 0;
  leftTokens.forEach((part, index) => {
    const spelling = part.spelling;
    if (part.type === "word" && isCorrectableSpelling(spelling)) {
      underlineCount += 1;
      const span = document.createElement("span");
      // Red = fix-all will apply it unattended (a restore with an autofix);
      // muted = ambiguous, needs a click to choose.
      const autofixable = spelling.status === "restore" && Boolean(spelling.autofix);
      span.className = autofixable
        ? "spell-underline spell-underline--auto"
        : "spell-underline";
      span.textContent = part.text;
      span.dataset.index = String(index);
      span.addEventListener("click", (event) => {
        event.stopPropagation();
        openCorrectionPopover(span, index, leftTokens, applyLeftSpellingCorrection);
      });
      sourceOverlay.append(span);
      return;
    }

    sourceOverlay.append(document.createTextNode(part.text));
  });

  inputLegend.hidden = underlineCount === 0;
  syncLeftOverlayScroll();
}

function renderResult(): void {
  resultOutput.replaceChildren();

  if (renderedParts.length === 0) {
    resultOutput.classList.add("is-empty");
    resultOutput.textContent = UI[lang].resultEmpty;
    updateFixAllButtonState();
    updateCopyButtonState();
    syncBoxHeights();
    return;
  }

  resultOutput.classList.remove("is-empty");

  renderedParts.forEach((part, index) => {
    const visibleText = getVisibleText(part);

    if (part.type === "sep") {
      resultOutput.append(document.createTextNode(visibleText));
      return;
    }

    if (shouldShowCorrection(part)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "token token-unknown token-correctable";
      button.title = UI[lang].correctionHeading;
      button.textContent = visibleText;
      button.dataset.index = String(index);
      button.dataset.correctable = part.spelling!.status;
      button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        openCorrectionPopover(button, index);
      });
      resultOutput.append(button);
      return;
    }

    if (part.ambiguous) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = [
        "token",
        "token-ambiguous",
        part.userChosen
          ? "token-user"
          : part.resolvedBy
            ? "token-resolved"
            : "token-unresolved",
      ].join(" ");
      button.textContent = visibleText;
      button.dataset.index = String(index);
      button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        openVariantPopover(button, index);
      });
      resultOutput.append(button);
      return;
    }

    if (part.unknown) {
      const span = document.createElement("span");
      span.className = "token token-unknown";
      span.title = UI[lang].unknownTitle;
      span.textContent = visibleText;
      resultOutput.append(span);
      return;
    }

    if (part.numeralFragment) {
      const span = document.createElement("span");
      span.className = "token token-numeral";
      span.textContent = visibleText;
      resultOutput.append(span);
      return;
    }

    if (part.accented || (part.variants && part.variants.length > 0)) {
      // Plain resolved word: clickable for morphology info, but not
      // underlined — it is not a choice. Readings not shipped with the
      // response (long-text fallback) are fetched on first click.
      const button = document.createElement("button");
      button.type = "button";
      button.className = "token token-plain";
      button.textContent = visibleText;
      button.dataset.index = String(index);
      button.setAttribute("aria-haspopup", "dialog");
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        openVariantPopover(button, index);
      });
      resultOutput.append(button);
      return;
    }

    resultOutput.append(document.createTextNode(visibleText));
  });

  updateFixAllButtonState();
  updateCopyButtonState();
  syncBoxHeights();
}

async function annotateUnknownWordsWithSpellcheck(): Promise<void> {
  const requestId = spellcheckRequestId;
  const targets = renderedParts
    .map((part, index) => ({ index, part }))
    .filter(
      (item) => item.part.type === "word" && !item.part.numeralFragment,
    );

  if (targets.length === 0) {
    updateFixAllButtonState();
    renderResult();
    return;
  }

  // One batched round-trip to the spellcheck Web Worker: all words are checked
  // off the main thread, so the ~580k engine's first build never freezes the UI.
  const words = targets.map(({ index, part }) => {
    const context = spellingContextForIndex(index, renderedParts);
    return { word: part.text, prev: context?.prev, next: context?.next };
  });

  let suggestions: SpellcheckSuggestion[];
  try {
    suggestions = await suggestBatch(words);
  } catch {
    suggestions = [];
  }

  if (requestId !== spellcheckRequestId) {
    return;
  }

  const results = targets.map(({ index, part }, position) => ({
    index,
    spelling:
      suggestions[position] ??
      ({ status: "unknown", candidates: [] } satisfies SpellcheckSuggestion),
    text: part.text,
  }));

  let changed = false;
  results.forEach(({ index, spelling, text }) => {
    const part = renderedParts[index];
    if (!part || part.text !== text) {
      return;
    }

    renderedParts[index] = { ...part, spelling };
    changed = true;
  });

  if (changed) {
    renderResult();
  } else {
    updateFixAllButtonState();
  }
}

function spellingContextForIndex(
  index: number,
  parts: readonly RenderedPart[],
): SpellcheckContext | undefined {
  const prev = nearestWordText(index, -1, parts);
  const next = nearestWordText(index, 1, parts);
  return prev || next ? { prev, next } : undefined;
}

function nearestWordText(
  index: number,
  direction: -1 | 1,
  parts: readonly RenderedPart[],
): string | undefined {
  for (
    let cursor = index + direction;
    cursor >= 0 && cursor < parts.length;
    cursor += direction
  ) {
    const part = parts[cursor];
    if (part?.type === "word") {
      return part.text;
    }
  }

  return undefined;
}

function isCorrectableSpelling(
  spelling: SpellcheckSuggestion | undefined,
): spelling is SpellcheckSuggestion {
  return (
    spelling !== undefined &&
    (spelling.status === "restore" || spelling.status === "typo") &&
    spelling.candidates.length > 0
  );
}

// Whether to flag a word as a spelling mistake, even if the accentuator produced
// something for it. A pure-ASCII diacritic drop ("restore") always wins — that's
// the "as" → "aš" case, clearly wrong in real LT text. An edit-distance typo is
// only surfaced when the accentuator itself didn't know the word, to avoid
// false positives on valid forms the lexicon happens to miss.
function shouldShowCorrection(part: RenderedPart): boolean {
  const spelling = part.spelling;
  if (!spelling || spelling.candidates.length === 0) {
    return false;
  }
  if (spelling.status === "restore") {
    return true;
  }
  if (spelling.status === "typo") {
    return Boolean(part.unknown);
  }
  return false;
}

function openCorrectionPopover(
  anchor: HTMLElement,
  index: number,
  parts: readonly RenderedPart[] = renderedParts,
  applyCorrection: (index: number, candidate: string) => void | Promise<void> =
    applySpellingCorrection,
): void {
  closePopover();

  const part = parts[index];
  if (!part) {
    return;
  }

  const popover = document.createElement("div");
  popover.className = "variant-popover correction-popover";
  popover.role = "dialog";
  const body = document.createElement("div");
  body.className = "variant-popover-scroll";
  popover.append(body);
  document.body.append(popover);
  activePopover = popover;

  const heading = document.createElement("div");
  heading.className = "variant-row variant-row--static correction-heading";
  heading.textContent = UI[lang].correctionHeading;
  body.append(heading);

  const candidates = isCorrectableSpelling(part.spelling)
    ? part.spelling.candidates
    : [];

  if (candidates.length === 0) {
    const status = document.createElement("div");
    status.className = "variant-row variant-row--static variant-status";
    status.textContent = UI[lang].correctionNone;
    body.append(status);
    positionVariantPopover(popover, anchor);
    return;
  }

  candidates.forEach((candidate) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "variant-row correction-option";

    const formLine = document.createElement("span");
    formLine.className = "variant-headword-line";

    const formText = document.createElement("span");
    formText.className = "variant-headword";
    formText.lang = "lt";
    formText.textContent = candidate.normalize("NFC");
    formLine.append(formText);

    option.append(formLine);
    option.addEventListener("click", () => {
      void applyCorrection(index, candidate);
    });
    body.append(option);
  });

  positionVariantPopover(popover, anchor);
}

async function applySpellingCorrection(index: number, candidate: string): Promise<void> {
  const part = renderedParts[index];
  if (
    !part ||
    typeof part.sourceStart !== "number" ||
    typeof part.sourceEnd !== "number" ||
    !canRewriteRenderedSource()
  ) {
    return;
  }

  const edits = [
    {
      start: part.sourceStart,
      end: part.sourceEnd,
      text: candidate.normalize("NFC"),
    },
  ];
  replaceTextareaRanges(edits);
  closePopover();
  void runLeftSpellcheck();
  await reaccentuateEdits(edits);
}

async function applyLeftSpellingCorrection(
  index: number,
  candidate: string,
): Promise<void> {
  const part = leftTokens[index];
  if (
    !part ||
    typeof part.sourceStart !== "number" ||
    typeof part.sourceEnd !== "number"
  ) {
    return;
  }

  replaceTextareaRanges([
    {
      start: part.sourceStart,
      end: part.sourceEnd,
      text: candidate.normalize("NFC"),
    },
  ]);
  closePopover();
  await runLeftSpellcheck();
}

async function fixAllRestores(): Promise<void> {
  const replacements = autofixableRestores();
  if (replacements.length === 0) {
    updateFixAllButtonState();
    return;
  }

  // Fix-all works on the LEFT overlay (before accentuation too) and does NOT
  // accentuate — consistent with left-side single fixes.
  closePopover();
  replaceTextareaRanges(replacements);
  await runLeftSpellcheck();
}

// Restores in the left overlay that "fix all" can apply unattended — each token's
// engine-chosen `autofix` (unambiguous restore). Guarded against stale offsets by
// requiring the tokens to still tile the current textarea value exactly.
function autofixableRestores(): Array<{ start: number; end: number; text: string }> {
  if (leftTokens.map((part) => part.text).join("") !== textarea.value) {
    return [];
  }

  return leftTokens.flatMap((part) => {
    const autofix = part.spelling?.status === "restore" ? part.spelling.autofix : undefined;
    if (
      !autofix ||
      typeof part.sourceStart !== "number" ||
      typeof part.sourceEnd !== "number"
    ) {
      return [];
    }

    return [{ start: part.sourceStart, end: part.sourceEnd, text: autofix.normalize("NFC") }];
  });
}

function replaceTextareaRanges(
  replacements: Array<{ start: number; end: number; text: string }>,
): void {
  let nextText = textarea.value;
  const sorted = [...replacements].sort((left, right) => right.start - left.start);

  sorted.forEach(({ start, end, text }) => {
    nextText = `${nextText.slice(0, start)}${text}${nextText.slice(end)}`;
  });

  textarea.value = nextText;
  updateCounter();
  updateFixAllButtonState();
  syncBoxHeights();
}

function canRewriteRenderedSource(): boolean {
  return Boolean(renderedSourceText) && textarea.value === renderedSourceText;
}

async function reaccentuateEdits(edits: TextEdit[]): Promise<void> {
  const oldText = renderedSourceText;
  const oldParts = renderedParts;

  if (accentMode === "web" || !oldText || oldParts.length === 0) {
    await submitText();
    return;
  }

  let scopedLoading = false;
  const fallbackToSubmit = async (): Promise<void> => {
    if (scopedLoading) {
      scopedLoading = false;
      setLoading(false);
    }
    await submitText();
  };

  try {
    const spans = buildEditedSentenceSpans(oldText, oldParts, edits, textarea.value);
    if (!spans || spans.length === 0) {
      await fallbackToSubmit();
      return;
    }

    setLoading(true);
    scopedLoading = true;
    const fragments: Part[][] = [];
    for (const span of spans) {
      fragments.push(
        await accentFragment(textarea.value.slice(span.newStart, span.newEnd)),
      );
    }

    const rebuilt = rebuildRenderedPartsWithFragments(
      oldParts,
      spans,
      fragments,
      textarea.value,
    );
    if (!rebuilt) {
      await fallbackToSubmit();
      return;
    }

    renderedParts = rebuilt as RenderedPart[];
    renderedSourceText = textarea.value;
    spellcheckRequestId += 1;
    showTaggerNotice(false);
    renderResult();
    await annotateUnknownWordsWithSpellcheck();
    updateCopyButtonState();
  } catch {
    await fallbackToSubmit();
    return;
  } finally {
    if (scopedLoading) {
      setLoading(false);
    }
  }
}

async function accentFragment(text: string): Promise<Part[]> {
  if (!text) {
    return [];
  }

  const engine = await ensureLocalEngine();
  const result = await engine.accent(text, (status) => {
    localRunStatus = status;
    renderLocalStatus();
  });
  localStats = result.stats;
  return result.parts;
}

function openVariantPopover(anchor: HTMLElement, index: number): void {
  closePopover();

  const part = renderedParts[index];
  if (!part) {
    return;
  }

  const popover = document.createElement("div");
  popover.className = "variant-popover";
  popover.role = "dialog";
  const body = document.createElement("div");
  body.className = "variant-popover-scroll";
  popover.append(body);
  document.body.append(popover);
  activePopover = popover;

  const variants = part.variants ?? [];

  if (variants.length === 0) {
    if (accentMode === "web" && part.type === "word" && !part.unknown) {
      renderPopoverStatus(popover, UI[lang].variantsLoading);
      positionVariantPopover(popover, anchor);
      void loadWordInfo(popover, anchor, index);
      return;
    }

    renderPopoverStatus(popover, UI[lang].variantsNone);
    positionVariantPopover(popover, anchor);
    return;
  }

  const interactive = Boolean(part.ambiguous);
  const chosenMi = bestReadingMi(part);

  // Display order: the variant that is actually shown in the result (or the
  // one carrying the tagger-matched reading) comes first, so the relevant
  // group is always at the top. Original indices are kept for selection.
  const order = variants.map((_, variantIndex) => variantIndex);
  let first = typeof part.chosen === "number" ? part.chosen : -1;
  if ((first < 0 || first >= order.length) && chosenMi) {
    first = variants.findIndex((variant) =>
      variant.info.split("; ").some((reading) => readingMatchesMi(reading, chosenMi)),
    );
  }
  if (first > 0 && first < order.length) {
    order.splice(order.indexOf(first), 1);
    order.unshift(first);
  }

  const populate = (forceAll: boolean): void => {
    body.replaceChildren();
    const topOnly =
      !forceAll && accentMode === "web" && displayMode === "top" && order.length > 1;
    const visibleOrder = topOnly ? order.slice(0, 1) : order;

  visibleOrder.forEach((variantIndex) => {
    const variant = variants[variantIndex]!;
    const readings = splitVariantReadings(variant.info);
    const matchedReadings =
      topOnly && chosenMi
        ? readings.filter((reading) => readingMatchesMi(reading, chosenMi))
        : [];
    const displayReadings =
      topOnly && chosenMi
        ? matchedReadings.length > 0
          ? matchedReadings
          : readings.slice(0, 1)
        : readings;

    displayReadings.forEach((reading, readingIndex) => {
      const option = document.createElement(interactive ? "button" : "div");
      option.className = "variant-row";

      if (interactive) {
        (option as HTMLButtonElement).type = "button";
      } else {
        option.classList.add("variant-row--static");
      }

      const selected = isSelectedReading(
        part,
        variantIndex,
        readingIndex,
        reading,
        chosenMi,
        order,
      );
      if (selected) {
        option.classList.add("is-selected");
        option.setAttribute("aria-current", "true");
      }

      const formLine = document.createElement("span");
      formLine.className = "variant-headword-line";

      const formText = document.createElement("span");
      formText.className = "variant-headword";
      formText.lang = "lt";
      formText.textContent = variant.form.normalize("NFC");
      formLine.append(formText);

      const metaText = variantMetaText(variant, selected);
      if (metaText) {
        const meta = document.createElement("span");
        meta.className =
          accentMode === "local"
            ? "variant-meta variant-probability"
            : "variant-meta variant-check";
        meta.textContent = metaText;
        formLine.append(meta);
      }

      option.append(formLine);

      const lines = parallelMorphologyLines(reading, lang);
      if (lines.morphology) {
        const morphology = document.createElement("span");
        morphology.className = "variant-morphology";
        morphology.textContent = lines.morphology;
        option.append(morphology);
      }
      if (lang !== "lt" && lines.gloss) {
        const gloss = document.createElement("span");
        gloss.className = "variant-gloss";
        gloss.textContent = lines.gloss;
        option.append(gloss);
      }

      if (interactive) {
        option.addEventListener("click", () => {
          const selectedMi = readingMi(reading);
          renderedParts[index] = {
            ...part,
            chosen: variantIndex,
            ...(selectedMi
              ? { chosenMi: selectedMi, tokenTags: parseMi(selectedMi) }
              : {}),
            current: matchCase(variant.form.normalize("NFC"), part.text).normalize(
              "NFC",
            ),
            userChosen: true,
          };
          closePopover();
          renderResult();
        });
      }

      body.append(option);
    });
  });

  if (topOnly) {
    const showAll = document.createElement("button");
    showAll.type = "button";
    showAll.className = "variant-show-all";
    showAll.textContent = UI[lang].displayShowAll;
    showAll.addEventListener("click", (event) => {
      // Expand in place. stopPropagation so the document-level click handler
      // (which closes the popover) doesn't fire after populate() detaches this
      // button from the DOM — that detach is exactly what closed it before.
      event.stopPropagation();
      populate(true);
      positionVariantPopover(popover, anchor);
    });
    body.append(showAll);
  }
  };

  populate(false);
  positionVariantPopover(popover, anchor);
}

function renderPopoverStatus(popover: HTMLDivElement, text: string): void {
  const body = variantPopoverBody(popover);
  const row = document.createElement("div");
  row.className = "variant-row variant-row--static variant-status";
  row.textContent = text;
  body.replaceChildren(row);
}

function variantPopoverBody(popover: HTMLElement): HTMLElement {
  return popover.querySelector<HTMLElement>(".variant-popover-scroll") ?? popover;
}

function splitVariantReadings(info: string): string[] {
  const readings = info
    .split("; ")
    .map((reading) => reading.trim())
    .filter(Boolean);
  return readings.length > 0 ? readings : [""];
}

function isSelectedReading(
  part: RenderedPart,
  variantIndex: number,
  readingIndex: number,
  reading: string,
  chosenMi: string | undefined,
  order: number[],
): boolean {
  if (part.userChosen) {
    if (chosenMi) {
      return part.chosen === variantIndex && readingMatchesMi(reading, chosenMi);
    }
    return part.chosen === variantIndex && readingIndex === 0;
  }

  if (chosenMi && readingMatchesMi(reading, chosenMi)) {
    return true;
  }

  if (typeof part.chosen === "number") {
    return part.chosen === variantIndex && readingIndex === 0;
  }

  return accentMode === "local" && variantIndex === order[0] && readingIndex === 0;
}

function variantMetaText(variant: Variant, selected: boolean): string {
  if (accentMode === "local" && typeof variant.p === "number") {
    return formatProbability(variant.p);
  }

  return accentMode === "web" && selected ? "✓" : "";
}

function readingMi(reading: string): string | undefined {
  const mi = reading.split(" - ")[0]?.trim();
  return mi || undefined;
}

function bestReadingMi(part: RenderedPart): string | undefined {
  if (part.chosenMi) {
    return part.chosenMi;
  }
  if (!part.tokenTags) {
    return undefined;
  }

  // Readings fetched after the accent response: score them here with the
  // token tags the server shipped alongside the word.
  let best: { mi: string; score: number } | null = null;
  for (const variant of part.variants ?? []) {
    for (const reading of variant.info.split("; ")) {
      const mi = reading.split(" - ")[0]?.trim();
      if (!mi) {
        continue;
      }
      const score = scoreTags(parseMi(mi), part.tokenTags);
      if (!best || score > best.score) {
        best = { mi, score };
      }
    }
  }

  return best && best.score > 0 ? best.mi : undefined;
}

function readingMatchesMi(reading: string, mi: string): boolean {
  const trimmed = reading.trim();
  return trimmed === mi || trimmed.startsWith(`${mi} - `);
}

async function loadWordInfo(
  popover: HTMLDivElement,
  anchor: HTMLElement,
  index: number,
): Promise<void> {
  const part = renderedParts[index];
  if (!part) {
    return;
  }

  try {
    const response = await fetch(`/api/word?w=${encodeURIComponent(part.text)}`);
    const payload = (await response.json().catch(() => null)) as
      | WordResponse
      | ErrorResponse
      | null;
    if (!response.ok || !payload || "error" in payload) {
      throw new Error(`Word info request failed (${response.status})`);
    }

    const variants = payload.variants ?? [];
    renderedParts[index] = { ...part, variants };

    if (activePopover !== popover) {
      return; // closed or replaced while loading
    }

    if (variants.length === 0) {
      renderPopoverStatus(popover, UI[lang].variantsNone);
      positionVariantPopover(popover, anchor);
      return;
    }

    closePopover();
    openVariantPopover(anchor, index);
  } catch {
    if (activePopover === popover) {
      renderPopoverStatus(popover, UI[lang].variantsError);
      positionVariantPopover(popover, anchor);
    }
  }
}

function closePopover(): void {
  activePopover?.remove();
  activePopover = null;
}

function toggleStatsPopover(): void {
  if (activeStatsPopover) {
    closeStatsPopover();
    return;
  }

  openStatsPopover();
}

function openStatsPopover(): void {
  closePopover();
  closeStatsPopover();

  const popover = document.createElement("div");
  popover.className = "stats-popover";
  popover.role = "dialog";
  document.body.append(popover);
  activeStatsPopover = popover;

  renderStatsPopover(popover);
  positionPopover(popover, localStatsButton);
}

function renderStatsPopover(popover: HTMLDivElement): void {
  const strings = UI[lang];
  const stats = localStats ?? localEngine?.getStats() ?? null;
  const title = document.createElement("strong");
  title.className = "stats-title";
  title.textContent = strings.statsTitle;
  popover.append(title);

  appendStatsRow(
    popover,
    strings.statsInferenceMode,
    stats ? executionModeLabel(stats.executionMode) : strings.statsModeUnknown,
  );
  appendStatsRow(popover, strings.statsMemory, formatMemoryUsage(stats));
  appendStatsRow(
    popover,
    strings.statsLastRun,
    stats?.lastRun
      ? `${stats.lastRun.tokensPerSecond.toFixed(1)} ${
          strings.localTokensPerSecond
        } · ${stats.lastRun.batches}`
      : strings.statsLastRunEmpty,
  );
  appendStatsRow(
    popover,
    strings.statsModel,
    stats?.modelFile ? stats.modelFile : strings.statsModeUnknown,
  );
  appendStatsRow(
    popover,
    strings.statsModelVersion,
    stats?.modelVersion || strings.statsModeUnknown,
  );
  appendStatsRow(popover, strings.statsCache, cacheLabel(stats?.cacheStatus ?? null));

  const redownload = document.createElement("button");
  redownload.type = "button";
  redownload.className = "stats-redownload";
  redownload.textContent = strings.statsRedownload;
  redownload.disabled = localDownloadGateState === "loading";
  redownload.addEventListener("click", () => {
    void forceRedownloadLocalModel();
  });
  popover.append(redownload);
}

function appendStatsRow(container: HTMLElement, label: string, value: string): void {
  const row = document.createElement("p");
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("span");
  valueNode.textContent = value;
  row.append(labelNode, valueNode);
  container.append(row);
}

function closeStatsPopover(): void {
  activeStatsPopover?.remove();
  activeStatsPopover = null;
}

function executionModeLabel(mode: ExecutionMode | null): string {
  const strings = UI[lang];
  if (mode === "worker") {
    return strings.statsModeWorker;
  }
  if (mode === "main") {
    return strings.statsModeMain;
  }
  return strings.statsModeUnknown;
}

function formatMemory(bytes: number | null): string {
  if (!bytes) {
    return UI[lang].unknownSize;
  }
  return `${(Number(bytes) / 1024 / 1024).toFixed(1)} MB`;
}

// When the model runs in ORT's proxy worker (the fast path), the WASM heap
// lives in that worker and is invisible to the main thread — so wasmBytes is 0.
// Show the WASM figure only when it's actually tracked (main-thread fallback),
// otherwise fall back to the JS heap so the row isn't a bare "unknown".
function formatMemoryUsage(stats: LocalStats | null): string {
  if (!stats) {
    return UI[lang].statsModeUnknown;
  }
  const memory = stats.memory;
  const js = memory.jsHeapBytes ? `JS ${formatMemory(memory.jsHeapBytes)}` : null;
  if (memory.wasmMemoryCount > 0 && memory.wasmBytes) {
    const wasm = `WASM ${formatMemory(memory.wasmBytes)}`;
    return js ? `${wasm} / ${js}` : wasm;
  }
  return js ?? UI[lang].statsModeUnknown;
}

function openPrimer(): void {
  closePopover();
  closeStatsPopover();
  primerBackdrop.hidden = false;
  document.body.classList.add("has-primer-open");
  primerDialog.focus({ preventScroll: true });
}

function closePrimer(): void {
  if (!isPrimerOpen()) {
    return;
  }

  primerBackdrop.hidden = true;
  document.body.classList.remove("has-primer-open");
  primerLink.focus({ preventScroll: true });
}

function isPrimerOpen(): boolean {
  return !primerBackdrop.hidden;
}

function trapPrimerFocus(event: KeyboardEvent): void {
  const focusable = Array.from(
    primerDialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter((element) => element.getClientRects().length > 0);

  if (focusable.length === 0) {
    event.preventDefault();
    primerDialog.focus({ preventScroll: true });
    return;
  }

  const first = focusable[0]!;
  const last = focusable[focusable.length - 1]!;

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

function positionVariantPopover(popover: HTMLElement, anchor: HTMLElement): void {
  const rect = anchor.getBoundingClientRect();
  const margin = 8;
  const caret = 10;
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const viewportHeight = window.innerHeight;
  const width = popover.offsetWidth;
  const height = popover.offsetHeight;
  const anchorCenterX = rect.left + rect.width / 2;

  const minLeft = window.scrollX + margin;
  const maxLeft = window.scrollX + viewportWidth - width - margin;
  const unclampedLeft = window.scrollX + anchorCenterX - width / 2;
  const left = Math.max(minLeft, Math.min(unclampedLeft, Math.max(minLeft, maxLeft)));

  const roomBelow = viewportHeight - rect.bottom;
  const roomAbove = rect.top;
  const fitsBelow = roomBelow >= height + caret + margin;
  const fitsAbove = roomAbove >= height + caret + margin;
  const placeAbove = !fitsBelow && (fitsAbove || roomAbove > roomBelow);
  const rawTop = placeAbove
    ? window.scrollY + rect.top - height - caret
    : window.scrollY + rect.bottom + caret;
  const minTop = window.scrollY + margin;
  const maxTop = window.scrollY + viewportHeight - height - margin;
  const top = Math.max(minTop, Math.min(rawTop, Math.max(minTop, maxTop)));
  const caretLeft = window.scrollX + anchorCenterX - left;

  popover.classList.toggle("is-above", placeAbove);
  popover.classList.toggle("is-below", !placeAbove);
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
  popover.style.setProperty(
    "--popover-caret-left",
    `${Math.max(12, Math.min(caretLeft, width - 12))}px`,
  );
}

function positionPopover(popover: HTMLElement, anchor: HTMLElement): void {
  const rect = anchor.getBoundingClientRect();
  const gap = 8;
  const maxLeft = window.innerWidth - popover.offsetWidth - 12;
  const left = Math.max(12, Math.min(rect.left + window.scrollX, maxLeft));
  const top = rect.bottom + window.scrollY + gap;

  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
}

async function copyResult(): Promise<void> {
  const text = renderedParts.map(getVisibleText).join("").normalize("NFC");
  if (!text) {
    return;
  }

  await navigator.clipboard.writeText(text);
  copied = true;
  renderUi();

  if (copyResetTimer) {
    window.clearTimeout(copyResetTimer);
  }
  copyResetTimer = window.setTimeout(() => {
    copied = false;
    renderUi();
  }, 1400);
}

function getVisibleText(part: RenderedPart): string {
  return (part.current ?? part.accented ?? part.text).normalize("NFC");
}

function setLoading(nextLoading: boolean): void {
  window.clearTimeout(copyResetTimer);
  copyResetTimer = undefined;
  isLoading = nextLoading;
  copied = false;
  updateAccentButtonState();
  updateFixAllButtonState();
  updateCopyButtonState();
  renderIconButtonLabel(
    accentButton,
    nextLoading ? UI[lang].accentButtonLoading : UI[lang].stressAllLabel,
    UI[lang].stressAllLabel,
  );
}

function updateAccentButtonState(): void {
  accentButton.disabled =
    isLoading || (accentMode === "local" && isLocalModelUnavailableBeforeReady());
}

function updateFixAllButtonState(): void {
  const disabled =
    isLoading ||
    (accentMode === "local" && isLocalModelUnavailableBeforeReady()) ||
    autofixableRestores().length === 0;
  fixAllButton.disabled = disabled;
  // When there are corrections to apply, light up with the exact same accent
  // fill as the Accentuate button (borrow its classes so every theme matches).
  fixAllButton.classList.toggle("primary-button", !disabled);
  fixAllButton.classList.toggle("accent-icon-button", !disabled);
}

function updateCopyButtonState(): void {
  copyButton.disabled = isLoading || renderedParts.length === 0;
}

function renderIconButtonLabel(
  button: HTMLButtonElement,
  ariaLabel: string,
  title: string,
): void {
  button.setAttribute("aria-label", ariaLabel);
  button.title = title;
}

function updateCounter(): void {
  const count = textarea.value.length;
  charCounter.textContent = `${count} / ${MAX_TEXT_LENGTH}`;
  charCounter.classList.toggle("is-over", count > MAX_TEXT_LENGTH);
}

// Lock the input and result boxes to one shared height so the two columns stay
// identical and back-to-back regardless of how much text or surrounding chrome
// each panel has. Measure each box's natural content height, take the larger,
// clamp to the CSS min (240px) / max (62vh), and apply to both.
function syncBoxHeights(): void {
  const minPx = 240;
  const maxPx = Math.round(window.innerHeight * 0.62);
  textarea.style.height = "auto";
  resultOutput.style.height = "auto";
  const content = Math.max(textarea.scrollHeight, resultOutput.scrollHeight);
  const height = Math.max(minPx, Math.min(content, maxPx));
  textarea.style.height = `${height}px`;
  resultOutput.style.height = `${height}px`;
  syncLeftOverlayScroll();

  // Lock the two panel footers (input actions / result legend) to one height so
  // that with the extras collapsed the input card's border lines up exactly with
  // the result card — the legend can wrap taller than the actions row. Only when
  // the panels sit side by side; stacked (mobile) they don't need to match.
  inputActions.style.minHeight = "";
  legend.style.minHeight = "";
  if (window.matchMedia("(min-width: 821px)").matches) {
    const footerHeight = Math.max(inputActions.offsetHeight, legend.offsetHeight);
    inputActions.style.minHeight = `${footerHeight}px`;
    legend.style.minHeight = `${footerHeight}px`;
  }
}

function setMessage(key: MessageKey | null): void {
  messageKey = key;
  message.textContent = key ? UI[lang][key] : "";
}

function showTaggerNotice(show: boolean): void {
  taggerNotice.hidden = !show;
}

function setLanguage(
  nextLang: Lang,
  options: { persist: boolean } = { persist: true },
): void {
  lang = nextLang;
  document.documentElement.lang = lang;

  if (options.persist) {
    localStorage.setItem("lang", lang);
  }

  closePopover();
  renderUi();
  renderResult();
}

function renderUi(): void {
  const strings = UI[lang];

  metaDescription?.setAttribute("content", strings.tagline);
  heroTagline.textContent = strings.tagline;
  inputLabel.textContent = strings.inputLabel;
  modeLabel.textContent = strings.modeLabel;
  modeButtons.forEach((button) => {
    const buttonMode = parseMode(button.dataset.mode);
    const isCurrent = buttonMode === accentMode;
    button.textContent = buttonMode === "local" ? strings.modeLocal : strings.modeWeb;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });
  modeExplainer.textContent =
    accentMode === "local"
      ? strings.modeLocalExplainer.replace("{size}", formatBytes(localExpectedBytes))
      : strings.modeWebExplainer;
  renderLocalTierControl(strings);
  renderIconButtonLabel(
    accentButton,
    isLoading ? strings.accentButtonLoading : strings.stressAllLabel,
    strings.stressAllLabel,
  );
  renderIconButtonLabel(fixAllButton, strings.fixAllLabel, strings.fixAllLabel);
  updateAccentButtonState();
  updateFixAllButtonState();
  updateCopyButtonState();
  copyButton.textContent = copied ? strings.copied : strings.copyButton;
  resultHeading.textContent = strings.resultHeading;
  renderDisplayControl(strings);
  renderLocalStatus();
  renderLocalUpdateControl(strings);
  localStatsButton.hidden = accentMode !== "local" || !localEngine;
  localStatsButton.setAttribute("aria-label", strings.statsButtonLabel);
  localStatsButton.title = strings.statsButtonLabel;
  taggerNoticeText.textContent = strings.taggerNotice;
  legend.setAttribute("aria-label", strings.legendLabel);
  legendLabel.textContent = strings.legendLabel;
  legendResolved.textContent = strings.legendResolved;
  legendAmbiguous.textContent = strings.legendAmbiguous;
  legendUser.textContent = strings.legendUser;
  legendUnknown.textContent = strings.legendUnknown;
  legendAutofix.textContent = strings.legendAutofix;
  legendClickfix.textContent = strings.legendClickfix;
  message.textContent = messageKey ? strings[messageKey] : "";
  renderPrimer(strings);

  languageButtons.forEach((button) => {
    const buttonLang = parseLang(button.dataset.lang);
    const isCurrent = buttonLang === lang;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });

  renderFooter(strings);
  renderDetailsToggle();
}

// The extras block (explainer + tier selector + local status) can be truncated
// so the input card's outer border lines up with the result card. Collapsing is
// only offered once nothing below needs the user's attention: in Web mode, or in
// Local mode once the model is ready. While the model still needs a download the
// block stays open and the toggle is hidden so the consent/progress card shows.
function renderDetailsToggle(): void {
  const strings = UI[lang];
  const canCollapse =
    accentMode === "web" ||
    (accentMode === "local" && localDownloadGateState === "ready");
  const collapsed = detailsCollapsed && canCollapse;
  panelExtras.dataset.collapsed = collapsed ? "true" : "false";
  detailsToggle.hidden = !canCollapse;
  detailsToggle.setAttribute("aria-expanded", String(!collapsed));
  const label = collapsed ? strings.detailsShow : strings.detailsHide;
  detailsToggle.setAttribute("aria-label", label);
  detailsToggle.title = label;
}

function renderLocalTierControl(strings: UiStrings): void {
  localTierControls.hidden = accentMode !== "local";
  if (accentMode !== "local") {
    return;
  }

  localTierLabel.textContent = strings.localTierLabel;
  localTierButtons.forEach((button) => {
    const buttonTier = parseLocalTier(button.dataset.tier);
    if (!buttonTier) {
      return;
    }

    const isCurrent = buttonTier === localTier;
    const label =
      buttonTier === "light" ? strings.localTierLight : strings.localTierHeavy;
    const bytes = localTierInfo[buttonTier]?.bytes;
    button.textContent = `${label} (${formatBytes(bytes)})`;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });
}

function renderDisplayControl(strings: UiStrings): void {
  const effectiveDisplay = effectiveDisplayMode();
  displayLabel.textContent = strings.displayLabel;
  displaySwitch.title = accentMode === "local" ? strings.displayLocalTooltip : "";

  displayButtons.forEach((button) => {
    const buttonDisplay = parseDisplayMode(button.dataset.display);
    const isCurrent = buttonDisplay === effectiveDisplay;
    button.textContent = buttonDisplay === "all" ? strings.displayAll : strings.displayTop;
    button.disabled = accentMode === "local";
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
    if (accentMode === "local") {
      button.title = strings.displayLocalTooltip;
    } else {
      button.removeAttribute("title");
    }
  });
}

function renderLocalStatus(): void {
  if (accentMode !== "local") {
    localStatusLine.hidden = true;
    localStatusLine.replaceChildren();
    return;
  }

  localStatusLine.hidden = false;
  localStatusLine.replaceChildren();

  if (
    localDownloadGateState === "needs-consent" ||
    localDownloadGateState === "needs-redownload"
  ) {
    localStatusLine.append(renderLocalConsentCard(UI[lang]));
    return;
  }

  const text =
    localDownloadGateState === "checking-cache"
      ? UI[lang].localCheckingCache
      : localRunStatusText() || localModelStatusText();
  localStatusLine.textContent = text;
}

function renderLocalUpdateControl(strings: UiStrings): void {
  localUpdateControl.replaceChildren();
  const update = currentLocalUpdateInfo();
  const canShow =
    accentMode === "local" &&
    localDownloadGateState === "ready" &&
    Boolean(localEngine) &&
    Boolean(update);
  localUpdateControl.hidden = !canShow;
  if (!canShow || !update) {
    return;
  }

  const label = document.createElement("span");
  label.textContent = template(strings.localUpdateAvailable, {
    size: formatBytes(update.bytes),
  });

  const button = document.createElement("button");
  button.type = "button";
  button.className = "secondary-button local-update-button";
  button.textContent = isLocalModelUpdating
    ? strings.localUpdating
    : template(strings.localUpdateButton, { size: formatBytes(update.bytes) });
  button.disabled = isLocalModelUpdating;
  button.addEventListener("click", () => {
    void updateCurrentLocalModel();
  });

  localUpdateControl.append(label, button);
}

function renderLocalConsentCard(strings: UiStrings): HTMLElement {
  const card = document.createElement("div");
  card.className = "local-consent-card";
  const isRedownload = localDownloadGateState === "needs-redownload";

  const copy = document.createElement("p");
  copy.textContent = isRedownload
    ? strings.localRedownloadText
    : template(strings.localConsentText, {
        size: formatBytes(localExpectedBytes),
      });

  const button = document.createElement("button");
  button.type = "button";
  button.className = "primary-button local-consent-button";
  button.append(createCloudArrowDownIcon());
  button.append(
    document.createTextNode(
      template(isRedownload ? strings.localRedownloadButton : strings.localConsentButton, {
        size: formatBytes(localExpectedBytes),
      }),
    ),
  );
  button.addEventListener("click", () => {
    void consentToLocalDownload();
  });

  card.append(copy, button);
  return card;
}

function createCloudArrowDownIcon(): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");

  [
    "M12 13v8",
    "m8 17 4 4 4-4",
    "M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3",
  ].forEach((d) => {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    svg.append(path);
  });

  return svg;
}

function localModelStatusText(): string {
  const strings = UI[lang];
  const status = localModelStatus;

  switch (status.type) {
    case "metadata":
      return strings.localMetadata;
    case "verify-runtime":
      return template(strings.localVerifyingRuntime, {
        file: status.file,
        done: formatBytes(status.received),
        total: formatBytes(status.total),
      });
    case "modelInfo":
      return template(strings.localModelInfo, {
        size: formatBytes(status.expectedBytes),
        cache: status.cacheState ? strings.cacheHit : strings.cacheMiss,
        threads: String(status.threads),
      });
    case "transfer":
      return template(status.cached ? strings.localReadingCache : strings.localDownloading, {
        done: formatBytes(status.received),
        total: formatBytes(status.total),
      });
    case "session":
      if (status.mode === "worker") {
        return strings.localSessionWorker;
      }
      if (status.mode === "fallback") {
        return strings.localSessionFallback;
      }
      return strings.localSessionMain;
    case "ready":
      return template(strings.localReady, {
        model: status.modelFile,
        size: formatBytes(status.bytes),
        cache: cacheLabel(status.cacheStatus),
      });
    case "failed":
      return template(strings.localFailed, { message: status.message });
    default:
      return strings.localIdle;
  }
}

function localRunStatusText(): string {
  const strings = UI[lang];
  const status = localRunStatus;

  switch (status.type) {
    case "running":
      return template(strings.localRunning, {
        sentences: String(status.sentences),
        batches: String(status.batches),
      });
    case "batch":
      return template(strings.localBatch, {
        done: String(status.renderedSentences),
        sentences: String(status.sentences),
        batch: String(status.batch),
        batches: String(status.batches),
        speed: status.tokensPerSecond.toFixed(1),
      });
    case "done":
      return template(strings.localDone, {
        tokens: String(status.inferredTokens),
        total: String(status.totalTokens),
        speed: status.tokensPerSecond.toFixed(1),
        seconds: (status.elapsedMs / 1000).toFixed(2),
      });
    case "memoryLimit":
      return strings.localMemoryLimit;
    case "error":
      return status.message;
    default:
      return "";
  }
}

function effectiveDisplayMode(): DisplayMode {
  return accentMode === "local" ? "top" : displayMode;
}

function currentLocalUpdateInfo(): LocalModelUpdateInfo | null {
  const stats = localStats ?? localEngine?.getStats() ?? null;
  if (stats?.updateAvailable && stats.update) {
    return stats.update;
  }

  if (localModelStatus.type === "ready" && localModelStatus.updateAvailable) {
    return localModelStatus.update;
  }

  return null;
}

function isLocalModelUnavailableBeforeReady(): boolean {
  return (
    isLocalModelUpdating ||
    !localEngine &&
    (localDownloadGateState === "checking-cache" ||
      localDownloadGateState === "needs-consent" ||
      localDownloadGateState === "needs-redownload" ||
      localDownloadGateState === "loading")
  );
}

function cacheLabel(status: CacheStatus | null): string {
  const strings = UI[lang];
  switch (status) {
    case "hit":
      return strings.cacheHit;
    case "stored":
      return strings.cacheStored;
    case "failed":
      return strings.cacheFailed;
    case "unavailable":
      return strings.cacheUnavailable;
    case "miss":
      return strings.cacheMiss;
    default:
      return strings.statsModeUnknown;
  }
}

function template(text: string, values: Record<string, string>): string {
  return Object.entries(values).reduce(
    (out, [key, value]) => out.split(`{${key}}`).join(value),
    text,
  );
}

function renderPrimer(strings: UiStrings): void {
  primerLink.textContent = strings.primerLink;
  primerTitle.textContent = strings.primerTitle;
  primerIntro.textContent = strings.primerIntro;
  primerGraveName.textContent = strings.primerGraveName;
  primerGraveDesc.textContent = strings.primerGraveDesc;
  primerGraveEx.textContent = strings.primerGraveEx;
  primerAcuteName.textContent = strings.primerAcuteName;
  primerAcuteDesc.textContent = strings.primerAcuteDesc;
  primerAcuteEx.textContent = strings.primerAcuteEx;
  primerTildeName.textContent = strings.primerTildeName;
  primerTildeDesc.textContent = strings.primerTildeDesc;
  primerTildeEx.textContent = strings.primerTildeEx;
  renderTextWithLtWords(primerMixed, strings.primerMixed, PRIMER_MIXED_WORDS);
  renderTextWithLtWords(primerPair, strings.primerPair, PRIMER_PAIR_WORDS);
  primerMore.href = VLKK_PRIMER_URL;
  primerMore.textContent = strings.primerMore;
}

function renderTextWithLtWords(
  container: HTMLElement,
  text: string,
  ltWords: string[],
): void {
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

function findNextLtWord(
  text: string,
  ltWords: string[],
  cursor: number,
): { index: number; word: string } | null {
  let next: { index: number; word: string } | null = null;

  ltWords.forEach((word) => {
    const index = text.indexOf(word, cursor);
    if (index !== -1 && (!next || index < next.index)) {
      next = { index, word };
    }
  });

  return next;
}

function getPrimerMixedWords(): string[] {
  const pieces = UI.lt.primerMixed.split(": ");
  const examples = pieces[pieces.length - 1]?.replace(/\.$/, "") ?? "";
  return examples
    .split(", ")
    .map((word) => word.trim())
    .filter(Boolean);
}

function getPrimerPairWords(): string[] {
  const match = UI.lt.primerPair.match(/: ([^(]+)\s\([^)]*\) ir ([^(]+)\s\(/);
  return match ? [match[1]!.trim(), match[2]!.trim()] : [];
}

function renderFooter(strings: UiStrings): void {
  // Attribution moved into the per-mode explainer (web mode names VDU +
  // UDPipe; local mode runs our own model). Footer holds only the small
  // project credit; kirtis.info reference removed.
  const repoLink = document.createElement("a");
  repoLink.href = "https://github.com/alexbalandi/kirciuokle";
  repoLink.rel = "noreferrer";
  repoLink.target = "_blank";
  repoLink.className = "footer-repo-link";
  // GitHub's own "mark" logo (Octocat silhouette) so the link visibly reads as
  // "this goes to GitHub". Inlined — a strict CSP blocks external assets.
  repoLink.insertAdjacentHTML(
    "afterbegin",
    '<svg class="github-mark" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path fill="currentColor" fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>',
  );
  const label = document.createElement("span");
  label.textContent = strings.footerProject;
  repoLink.append(label);
  siteFooter.replaceChildren(repoLink);
}

function parseLang(value: string | undefined): Lang | null {
  return LANGS.find((candidate) => candidate === value) ?? null;
}

function parseMode(value: string | undefined): AccentMode | null {
  return value === "web" || value === "local" ? value : null;
}

function parseLocalTier(value: string | undefined): LocalModelTier | null {
  return value === "light" || value === "heavy" ? value : null;
}

function parseDisplayMode(value: string | undefined): DisplayMode | null {
  return value === "top" || value === "all" ? value : null;
}

function readStoredMode(): AccentMode {
  return parseMode(localStorage.getItem(MODE_STORAGE_KEY) ?? undefined) ?? "web";
}

function readStoredLocalTier(): LocalModelTier {
  return (
    parseLocalTier(localStorage.getItem(LOCAL_TIER_STORAGE_KEY) ?? undefined) ??
    LOCAL_DEFAULT_MODEL_TIER
  );
}

function readStoredDisplayMode(): DisplayMode {
  return parseDisplayMode(localStorage.getItem(DISPLAY_STORAGE_KEY) ?? undefined) ?? "all";
}

function readReadyLocalTiers(): LocalModelTier[] {
  try {
    const value = JSON.parse(localStorage.getItem(LOCAL_READY_TIERS_STORAGE_KEY) ?? "[]");
    if (!Array.isArray(value)) {
      return [];
    }
    return value.filter((item): item is LocalModelTier => parseLocalTier(String(item)) !== null);
  } catch {
    return [];
  }
}

function wasLocalTierPreviouslyReady(tier: LocalModelTier): boolean {
  return readReadyLocalTiers().includes(tier);
}

function markLocalTierReady(tier: LocalModelTier): void {
  const ready = new Set(readReadyLocalTiers());
  ready.add(tier);
  localStorage.setItem(LOCAL_READY_TIERS_STORAGE_KEY, JSON.stringify([...ready]));
}

function getMessageKeyForStatus(status: number): MessageKey {
  switch (status) {
    case 400:
      return "errEmpty";
    case 413:
      return "errTooLong";
    case 502:
      return "errUpstream";
    default:
      return "errUnexpected";
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function matchCase(accented: string, original: string): string {
  if (original.length > 1 && original.toUpperCase() === original) {
    return accented.toUpperCase();
  }

  if (original[0] && original[0].toUpperCase() === original[0]) {
    return accented[0] ? accented[0].toUpperCase() + accented.slice(1) : accented;
  }

  return accented;
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}`);
  }

  return element as T;
}
