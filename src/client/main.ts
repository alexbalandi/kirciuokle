import type {
  AccentResponse,
  ErrorResponse,
  Part,
} from "../shared/types";
import {
  detectLang,
  LANGS,
  morphologySegments,
  UI,
  type Lang,
  type MorphSegment,
  type UiStrings,
} from "./i18n";
import "./style.css";

const MAX_TEXT_LENGTH = 20_000;

type MessageKey = Extract<
  keyof UiStrings,
  "errEmpty" | "errTooLong" | "errUpstream" | "errUnexpected"
>;

type RenderedPart = Part & {
  current?: string;
  userChosen?: boolean;
};

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
const accentButton = getElement<HTMLButtonElement>("accent-button");
const copyButton = getElement<HTMLButtonElement>("copy-button");
const message = getElement<HTMLParagraphElement>("form-message");
const resultHeading = getElement<HTMLHeadingElement>("result-heading");
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
const siteFooter = getElement<HTMLElement>("site-footer");
const metaDescription = document.querySelector<HTMLMetaElement>(
  'meta[name="description"]',
);

let lang: Lang = detectLang();
let renderedParts: RenderedPart[] = [];
let activePopover: HTMLDivElement | null = null;
let isLoading = false;
let messageKey: MessageKey | null = null;
let copyResetTimer: number | undefined;
let copied = false;

languageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextLang = parseLang(button.dataset.lang);
    if (nextLang) {
      setLanguage(nextLang);
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

taggerNoticeClose.addEventListener("click", () => {
  taggerNotice.hidden = true;
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closePopover();
  }
});

document.addEventListener("click", (event) => {
  const target = event.target;
  if (
    target instanceof Node &&
    activePopover &&
    !activePopover.contains(target) &&
    !(target instanceof HTMLElement && target.closest(".token-ambiguous"))
  ) {
    closePopover();
  }
});

setLanguage(lang, { persist: false });
resizeTextarea();
updateCounter();

async function submitText(): Promise<void> {
  const text = textarea.value;
  closePopover();
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
  document.body.append(popover);
  activePopover = popover;

  const variants = part.variants ?? [];

  if (variants.length === 0) {
    popover.textContent = UI[lang].variantsNone;
    positionPopover(popover, anchor);
    return;
  }

  variants.forEach((variant, variantIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "variant-option";

    if (part.chosen === variantIndex) {
      button.classList.add("is-selected");
      button.setAttribute("aria-current", "true");
    }

    const formText = document.createElement("strong");
    formText.textContent = variant.form.normalize("NFC");
    button.append(formText);

    if (variant.info) {
      const info = document.createElement("span");
      info.className = "variant-info";
      variant.info.split("; ").forEach((reading) => {
        const row = document.createElement("span");
        row.className = "variant-reading";
        appendMorphologyInfo(row, morphologySegments(reading, lang));
        info.append(row);
      });
      button.append(info);
    }

    button.addEventListener("click", () => {
      renderedParts[index] = {
        ...part,
        chosen: variantIndex,
        current: matchCase(variant.form.normalize("NFC"), part.text).normalize("NFC"),
        userChosen: true,
      };
      closePopover();
      renderResult();
    });

    popover.append(button);
  });

  positionPopover(popover, anchor);
}

function appendMorphologyInfo(
  container: HTMLElement,
  segments: MorphSegment[],
): void {
  segments.forEach((segment) => {
    if (!segment.lt) {
      container.append(document.createTextNode(segment.text));
      return;
    }

    // Lithuanian term is the primary text; the translation is the small
    // helper annotation underneath.
    const ruby = document.createElement("ruby");
    ruby.append(document.createTextNode(segment.lt));

    const rt = document.createElement("rt");
    rt.textContent = segment.text;
    ruby.append(rt);

    container.append(ruby);
  });
}

function closePopover(): void {
  activePopover?.remove();
  activePopover = null;
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
  accentButton.disabled = nextLoading;
  accentButton.textContent = nextLoading
    ? UI[lang].accentButtonLoading
    : UI[lang].accentButton;
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
  accentButton.textContent = isLoading
    ? strings.accentButtonLoading
    : strings.accentButton;
  copyButton.textContent = copied ? strings.copied : strings.copyButton;
  resultHeading.textContent = strings.resultHeading;
  taggerNoticeText.textContent = strings.taggerNotice;
  legend.setAttribute("aria-label", strings.legendLabel);
  legendLabel.textContent = strings.legendLabel;
  legendResolved.textContent = strings.legendResolved;
  legendAmbiguous.textContent = strings.legendAmbiguous;
  legendUser.textContent = strings.legendUser;
  legendUnknown.textContent = strings.legendUnknown;
  message.textContent = messageKey ? strings[messageKey] : "";

  languageButtons.forEach((button) => {
    const buttonLang = parseLang(button.dataset.lang);
    const isCurrent = buttonLang === lang;
    button.classList.toggle("is-active", isCurrent);
    button.setAttribute("aria-pressed", String(isCurrent));
  });

  renderFooter(strings);
}

function renderFooter(strings: UiStrings): void {
  const vduLink = document.createElement("a");
  vduLink.href = "https://kalbu.vdu.lt";
  vduLink.rel = "noreferrer";
  vduLink.target = "_blank";
  vduLink.textContent = "VDU kirčiuoklė";

  const kirtisLink = document.createElement("a");
  kirtisLink.href = "https://kirtis.info";
  kirtisLink.rel = "noreferrer";
  kirtisLink.target = "_blank";
  kirtisLink.textContent = "kirtis.info";

  siteFooter.replaceChildren(
    document.createTextNode(`${strings.footerData}: `),
    vduLink,
    document.createTextNode(` (kalbu.vdu.lt) · ${strings.footerInspired} `),
    kirtisLink,
  );
}

function parseLang(value: string | undefined): Lang | null {
  return LANGS.find((candidate) => candidate === value) ?? null;
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
