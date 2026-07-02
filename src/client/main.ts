import type {
  AccentResponse,
  ErrorResponse,
  Part,
} from "../shared/types";
import "./style.css";

const MAX_TEXT_LENGTH = 20_000;

type RenderedPart = Part & {
  current?: string;
  userChosen?: boolean;
};

const form = getElement<HTMLFormElement>("accent-form");
const textarea = getElement<HTMLTextAreaElement>("source-text");
const charCounter = getElement<HTMLSpanElement>("char-counter");
const accentButton = getElement<HTMLButtonElement>("accent-button");
const copyButton = getElement<HTMLButtonElement>("copy-button");
const message = getElement<HTMLParagraphElement>("form-message");
const resultOutput = getElement<HTMLDivElement>("result-output");
const taggerNotice = getElement<HTMLDivElement>("tagger-notice");
const taggerNoticeClose = getElement<HTMLButtonElement>("tagger-notice-close");

let renderedParts: RenderedPart[] = [];
let activePopover: HTMLDivElement | null = null;

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

resizeTextarea();
updateCounter();

async function submitText(): Promise<void> {
  const text = textarea.value;
  closePopover();
  setMessage("");

  if (text.trim().length === 0) {
    setMessage("Įveskite tekstą.");
    return;
  }

  if (text.length > MAX_TEXT_LENGTH) {
    setMessage("Tekstas per ilgas.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/accent", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const payload = (await response.json()) as AccentResponse | ErrorResponse;
    if (!response.ok || "error" in payload) {
      throw new Error("error" in payload ? payload.error : "Nepavyko sukirčiuoti.");
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
    setMessage(error instanceof Error ? error.message : "Nepavyko sukirčiuoti.");
  } finally {
    setLoading(false);
  }
}

function renderResult(): void {
  resultOutput.replaceChildren();

  if (renderedParts.length === 0) {
    resultOutput.classList.add("is-empty");
    resultOutput.textContent = "Rezultatas atsiras čia.";
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
        part.resolvedBy && !part.userChosen ? "token-resolved" : "token-unresolved",
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
      span.title = "Žodyne nerasta";
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
    popover.textContent = "Variantų nerasta.";
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
      info.textContent = variant.info;
      button.append(info);
    }

    if (part.chosen === variantIndex) {
      const marker = document.createElement("span");
      marker.className = "variant-current";
      marker.textContent = "Pasirinkta";
      button.append(marker);
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
  const previous = copyButton.textContent;
  copyButton.textContent = "Nukopijuota ✓";
  window.setTimeout(() => {
    copyButton.textContent = previous;
  }, 1400);
}

function getVisibleText(part: RenderedPart): string {
  return (part.current ?? part.accented ?? part.text).normalize("NFC");
}

function setLoading(isLoading: boolean): void {
  accentButton.disabled = isLoading;
  accentButton.textContent = isLoading ? "Kirčiuojama..." : "Sukirčiuoti";
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

function setMessage(text: string): void {
  message.textContent = text;
}

function showTaggerNotice(show: boolean): void {
  taggerNotice.hidden = !show;
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
