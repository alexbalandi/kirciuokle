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
  formatBytes,
  hasCachedLocalModel,
  LOCAL_MODEL_SIZE_FALLBACK,
} from "./local/assets";
import {
  createLocalDownloadGate,
  type LocalDownloadGateState,
} from "./local/consent";
import type { LocalAccentEngine } from "./local/engine";
import type {
  CacheStatus,
  ExecutionMode,
  LocalModelStatus,
  LocalRunStatus,
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

type RenderedPart = Part & {
  current?: string;
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
const charCounter = getElement<HTMLSpanElement>("char-counter");
const modeLabel = getElement<HTMLSpanElement>("mode-label");
const modeSwitch = getElement<HTMLSpanElement>("mode-switch");
const modeButtons = Array.from(
  modeSwitch.querySelectorAll<HTMLButtonElement>("button[data-mode]"),
);
const modeExplainer = getElement<HTMLParagraphElement>("mode-explainer");
const localStatusLine = getElement<HTMLDivElement>("local-status");
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
let renderedParts: RenderedPart[] = [];
let activePopover: HTMLDivElement | null = null;
let activeStatsPopover: HTMLDivElement | null = null;
let isLoading = false;
let messageKey: MessageKey | null = null;
let copyResetTimer: number | undefined;
let copied = false;
let localEngine: LocalAccentEngine | null = null;
let localEnginePromise: Promise<LocalAccentEngine> | null = null;
let localModelStatus: LocalModelStatus = { type: "idle" };
let localRunStatus: LocalRunStatus = { type: "ready" };
let localStats: LocalStats | null = null;
let localExpectedBytes: number = LOCAL_MODEL_SIZE_FALLBACK;
let localDownloadGateState: LocalDownloadGateState = "inactive";

const localDownloadGate = createLocalDownloadGate({
  hasCachedModel: hasCachedLocalModel,
  ensureEngine: ensureLocalEngine,
  isEngineReady: () => Boolean(localEngine),
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

displayButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextDisplay = parseDisplayMode(button.dataset.display);
    if (nextDisplay && accentMode === "web") {
      setDisplayMode(nextDisplay);
    }
  });
});

textarea.addEventListener("input", () => {
  resizeTextarea();
  updateCounter();
});

textarea.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitText();
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

// Keep the input and the result scrolled to the same relative position —
// the texts are identical, so proportional sync keeps the same passage
// visible on both sides. The induced scroll event on the synced element is
// swallowed via the ignore set (no timers — they stall in background tabs).
const scrollIgnore = new Set<HTMLElement>();

function syncScroll(source: HTMLElement, target: HTMLElement): void {
  if (scrollIgnore.delete(source)) {
    return;
  }
  const sourceMax = source.scrollHeight - source.clientHeight;
  const targetMax = target.scrollHeight - target.clientHeight;
  if (sourceMax <= 0 || targetMax <= 0) {
    return;
  }
  const next = (source.scrollTop / sourceMax) * targetMax;
  if (Math.abs(target.scrollTop - next) < 1) {
    return;
  }
  scrollIgnore.add(target);
  target.scrollTop = next;
}

textarea.addEventListener("scroll", () => syncScroll(textarea, resultOutput));
resultOutput.addEventListener("scroll", () => syncScroll(resultOutput, textarea));

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
    !(target instanceof HTMLElement && target.closest(".token-ambiguous, .token-plain"))
  ) {
    closePopover();
  }
});

setLanguage(lang, { persist: false });
resizeTextarea();
updateCounter();
if (accentMode === "local") {
  void enterLocalMode();
}

async function submitText(): Promise<void> {
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

async function accentTextWeb(text: string): Promise<AccentResponse> {
  const response = await fetch("/api/accent", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
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
  renderedParts = payload.parts.map((part) => ({
    ...part,
    text: part.text.normalize("NFC"),
    accented: part.accented?.normalize("NFC"),
    variants: part.variants?.map((variant) => ({
      ...variant,
      form: variant.form.normalize("NFC"),
    })),
    current: (part.accented ?? part.text).normalize("NFC"),
  }));

  showTaggerNotice(payload.tagger === "unavailable");
  renderResult();
  copyButton.disabled = renderedParts.length === 0;
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
    await localDownloadGate.enterLocalMode();
  } catch {
    setMessage("errUnexpected");
  }
}

async function consentToLocalDownload(): Promise<void> {
  try {
    await localDownloadGate.consentToDownload();
  } catch {
    setMessage("errUnexpected");
  }
}

function setDisplayMode(nextDisplay: DisplayMode): void {
  displayMode = nextDisplay;
  localStorage.setItem(DISPLAY_STORAGE_KEY, displayMode);
  closePopover();
  renderUi();
}

async function ensureLocalEngine(): Promise<LocalAccentEngine> {
  if (localEngine) {
    return localEngine;
  }

  if (localEnginePromise) {
    return localEnginePromise;
  }

  localRunStatus = { type: "ready" };
  localEnginePromise = import("./local/engine")
    .then(({ LocalAccentEngine: Engine }) =>
      Engine.create((status) => {
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
      localEngine = engine;
      localStats = engine.getStats();
      localModelStatus =
        localModelStatus.type === "ready" ? localModelStatus : { type: "idle" };
      renderUi();
      return engine;
    })
    .catch((error: unknown) => {
      localEnginePromise = null;
      localModelStatus = { type: "failed", message: errorMessage(error) };
      renderUi();
      throw error;
    });

  renderUi();
  return localEnginePromise;
}

function renderResult(): void {
  resultOutput.replaceChildren();

  if (renderedParts.length === 0) {
    resultOutput.classList.add("is-empty");
    resultOutput.textContent = UI[lang].resultEmpty;
    return;
  }

  resultOutput.classList.remove("is-empty");

  renderedParts.forEach((part, index) => {
    const visibleText = getVisibleText(part);

    if (part.type === "sep") {
      resultOutput.append(document.createTextNode(visibleText));
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

  const topOnly = accentMode === "web" && displayMode === "top" && order.length > 1;
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
    showAll.addEventListener("click", () => {
      setDisplayMode("all");
      closePopover();
      openVariantPopover(anchor, index);
    });
    body.append(showAll);
  }

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
  appendStatsRow(
    popover,
    strings.statsMemory,
    stats
      ? `WASM ${formatMemory(stats.memory.wasmBytes)} / JS ${formatMemory(
          stats.memory.jsHeapBytes,
        )}`
      : strings.statsModeUnknown,
  );
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
    stats?.modelFile
      ? `${stats.modelFile}${stats.modelVersion ? ` · ${stats.modelVersion}` : ""}`
      : strings.statsModeUnknown,
  );
  appendStatsRow(popover, strings.statsCache, cacheLabel(stats?.cacheStatus ?? null));
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
  accentButton.textContent = nextLoading
    ? UI[lang].accentButtonLoading
    : UI[lang].accentButton;
}

function updateAccentButtonState(): void {
  accentButton.disabled =
    isLoading || (accentMode === "local" && isLocalModelUnavailableBeforeReady());
}

function updateCounter(): void {
  const count = textarea.value.length;
  charCounter.textContent = `${count} / ${MAX_TEXT_LENGTH}`;
  charCounter.classList.toggle("is-over", count > MAX_TEXT_LENGTH);
}

function resizeTextarea(): void {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
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
  accentButton.textContent = isLoading
    ? strings.accentButtonLoading
    : strings.accentButton;
  updateAccentButtonState();
  copyButton.textContent = copied ? strings.copied : strings.copyButton;
  resultHeading.textContent = strings.resultHeading;
  renderDisplayControl(strings);
  renderLocalStatus();
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
  message.textContent = messageKey ? strings[messageKey] : "";
  renderPrimer(strings);

  languageButtons.forEach((button) => {
    const buttonLang = parseLang(button.dataset.lang);
    const isCurrent = buttonLang === lang;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });

  renderFooter(strings);
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

  if (!localEngine && localDownloadGateState === "needs-consent") {
    localStatusLine.append(renderLocalConsentCard(UI[lang]));
    return;
  }

  const text =
    localDownloadGateState === "checking-cache"
      ? UI[lang].localCheckingCache
      : localRunStatusText() || localModelStatusText();
  localStatusLine.textContent = text;
}

function renderLocalConsentCard(strings: UiStrings): HTMLElement {
  const card = document.createElement("div");
  card.className = "local-consent-card";

  const copy = document.createElement("p");
  copy.textContent = template(strings.localConsentText, {
    size: formatBytes(localExpectedBytes),
  });

  const button = document.createElement("button");
  button.type = "button";
  button.className = "primary-button local-consent-button";
  button.append(createCloudArrowDownIcon());
  button.append(
    document.createTextNode(
      template(strings.localConsentButton, {
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

function isLocalModelUnavailableBeforeReady(): boolean {
  return (
    !localEngine &&
    (localDownloadGateState === "checking-cache" ||
      localDownloadGateState === "needs-consent" ||
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
  repoLink.textContent = strings.footerProject;
  siteFooter.replaceChildren(repoLink);
}

function parseLang(value: string | undefined): Lang | null {
  return LANGS.find((candidate) => candidate === value) ?? null;
}

function parseMode(value: string | undefined): AccentMode | null {
  return value === "web" || value === "local" ? value : null;
}

function parseDisplayMode(value: string | undefined): DisplayMode | null {
  return value === "top" || value === "all" ? value : null;
}

function readStoredMode(): AccentMode {
  return parseMode(localStorage.getItem(MODE_STORAGE_KEY) ?? undefined) ?? "web";
}

function readStoredDisplayMode(): DisplayMode {
  return parseDisplayMode(localStorage.getItem(DISPLAY_STORAGE_KEY) ?? undefined) ?? "all";
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
