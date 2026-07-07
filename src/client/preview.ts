import type { Part, Variant } from "../shared/types";

export type RenderedPartCore = Part & {
  current?: string;
  preview?: true;
  sourceEnd?: number;
  sourceStart?: number;
};

export type TextEdit = {
  start: number;
  end: number;
  text: string;
};

export type EditedSpan = {
  oldStart: number;
  oldEnd: number;
  newStart: number;
  newEnd: number;
};

const LT_LETTER_RE = /[a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ]/u;
const INTERNAL_WORD_MARKS = new Set(["-", "'", "’"]);
const SENTENCE_TERMINATORS = new Set([".", "!", "?", "…", "\n"]);

export function tokenizeForPreview(text: string): RenderedPartCore[] {
  const parts: RenderedPartCore[] = [];
  let cursor = 0;

  while (cursor < text.length) {
    const sourceStart = cursor;
    if (isLtLetter(text[cursor])) {
      cursor += 1;
      while (cursor < text.length) {
        const char = text[cursor];
        if (isLtLetter(char)) {
          cursor += 1;
          continue;
        }
        if (
          isInternalWordMark(char) &&
          cursor > sourceStart &&
          isLtLetter(text[cursor - 1]) &&
          isLtLetter(text[cursor + 1])
        ) {
          cursor += 1;
          continue;
        }
        break;
      }

      parts.push(previewPart(text.slice(sourceStart, cursor), "word", sourceStart));
      continue;
    }

    cursor += 1;
    while (cursor < text.length && !isLtLetter(text[cursor])) {
      cursor += 1;
    }
    parts.push(previewPart(text.slice(sourceStart, cursor), "sep", sourceStart));
  }

  return parts;
}

export function buildEditedSentenceSpans(
  oldText: string,
  parts: readonly RenderedPartCore[],
  edits: readonly TextEdit[],
  newText: string,
): EditedSpan[] | null {
  const sortedEdits = [...edits].sort((left, right) => left.start - right.start);
  if (
    sortedEdits.some(
      (edit) =>
        edit.start < 0 ||
        edit.end < edit.start ||
        edit.end > oldText.length,
    )
  ) {
    return null;
  }

  const rawSpans: Array<{ oldStart: number; oldEnd: number }> = [];
  for (const edit of sortedEdits) {
    const sentence = sentenceBoundsForEdit(oldText, edit);
    const snapped = snapSpanToPartBoundaries(parts, sentence.oldStart, sentence.oldEnd);
    if (!snapped) {
      return null;
    }
    rawSpans.push(snapped);
  }

  const merged = mergeOldSpans(rawSpans);
  for (const edit of sortedEdits) {
    if (
      !merged.some(
        (span) => edit.start >= span.oldStart && edit.end <= span.oldEnd,
      )
    ) {
      return null;
    }
  }

  const spans = merged.map((span) => {
    const newStart = span.oldStart + deltaBefore(sortedEdits, span.oldStart);
    const newEnd = span.oldEnd + deltaThrough(sortedEdits, span.oldEnd);
    return {
      ...span,
      newStart,
      newEnd,
    };
  });

  if (
    spans.some(
      (span) =>
        span.newStart < 0 ||
        span.newEnd < span.newStart ||
        span.newEnd > newText.length,
    )
  ) {
    return null;
  }

  return spans;
}

export function rebuildRenderedPartsWithFragments(
  oldParts: readonly RenderedPartCore[],
  spans: readonly EditedSpan[],
  fragments: readonly (readonly Part[])[],
  newText: string,
): RenderedPartCore[] | null {
  if (spans.length !== fragments.length) {
    return null;
  }

  const rebuilt: RenderedPartCore[] = [];
  let partIndex = 0;

  spans.forEach((span, spanIndex) => {
    while (
      partIndex < oldParts.length &&
      (oldParts[partIndex]!.sourceEnd ?? 0) <= span.oldStart
    ) {
      rebuilt.push(oldParts[partIndex]!);
      partIndex += 1;
    }

    while (
      partIndex < oldParts.length &&
      (oldParts[partIndex]!.sourceStart ?? 0) < span.oldEnd
    ) {
      partIndex += 1;
    }

    rebuilt.push(...fragments[spanIndex]!);
  });

  while (partIndex < oldParts.length) {
    rebuilt.push(oldParts[partIndex]!);
    partIndex += 1;
  }

  const retiled = retileRenderedParts(rebuilt);
  return joinedText(retiled) === newText ? retiled : null;
}

export function retileRenderedParts(
  parts: readonly RenderedPartCore[],
): RenderedPartCore[] {
  let cursor = 0;

  return parts.map((part) => {
    const text = part.text.normalize("NFC");
    const accented = part.accented?.normalize("NFC");
    const variants = part.variants?.map(normalizeVariant);
    const sourceStart = cursor;
    cursor += text.length;

    return {
      ...part,
      text,
      accented,
      variants,
      current: (accented ?? text).normalize("NFC"),
      preview: part.preview,
      sourceStart,
      sourceEnd: cursor,
    };
  });
}

function previewPart(
  text: string,
  type: "word" | "sep",
  sourceStart: number,
): RenderedPartCore {
  return {
    text,
    type,
    current: text,
    preview: true,
    sourceStart,
    sourceEnd: sourceStart + text.length,
  };
}

function isLtLetter(char: string | undefined): boolean {
  return Boolean(char && LT_LETTER_RE.test(char));
}

function isInternalWordMark(char: string | undefined): boolean {
  return Boolean(char && INTERNAL_WORD_MARKS.has(char));
}

function sentenceBoundsForEdit(
  oldText: string,
  edit: TextEdit,
): { oldStart: number; oldEnd: number } {
  let oldStart = Math.min(edit.start, oldText.length);
  while (oldStart > 0 && !isSentenceTerminator(oldText[oldStart - 1])) {
    oldStart -= 1;
  }

  let oldEnd = Math.min(Math.max(edit.end, edit.start), oldText.length);
  while (oldEnd < oldText.length && !isSentenceTerminator(oldText[oldEnd])) {
    oldEnd += 1;
  }
  while (oldEnd < oldText.length && isSentenceTerminator(oldText[oldEnd])) {
    oldEnd += 1;
  }

  return { oldStart, oldEnd };
}

function isSentenceTerminator(char: string | undefined): boolean {
  return Boolean(char && SENTENCE_TERMINATORS.has(char));
}

function snapSpanToPartBoundaries(
  parts: readonly RenderedPartCore[],
  oldStart: number,
  oldEnd: number,
): { oldStart: number; oldEnd: number } | null {
  const first = parts.find(
    (part) =>
      typeof part.sourceStart === "number" &&
      typeof part.sourceEnd === "number" &&
      part.sourceEnd > oldStart &&
      part.sourceStart < oldEnd,
  );
  if (!first || typeof first.sourceStart !== "number") {
    return null;
  }

  const last = [...parts]
    .reverse()
    .find(
      (part) =>
        typeof part.sourceStart === "number" &&
        typeof part.sourceEnd === "number" &&
        part.sourceEnd > oldStart &&
        part.sourceStart < oldEnd,
    );
  if (!last || typeof last.sourceEnd !== "number") {
    return null;
  }

  return { oldStart: first.sourceStart, oldEnd: last.sourceEnd };
}

function mergeOldSpans(
  spans: Array<{ oldStart: number; oldEnd: number }>,
): Array<{ oldStart: number; oldEnd: number }> {
  const sorted = [...spans].sort(
    (left, right) => left.oldStart - right.oldStart || left.oldEnd - right.oldEnd,
  );
  const merged: Array<{ oldStart: number; oldEnd: number }> = [];

  for (const span of sorted) {
    const previous = merged[merged.length - 1];
    if (previous && span.oldStart <= previous.oldEnd) {
      previous.oldEnd = Math.max(previous.oldEnd, span.oldEnd);
      continue;
    }
    merged.push({ ...span });
  }

  return merged;
}

function deltaBefore(edits: readonly TextEdit[], position: number): number {
  return edits.reduce(
    (delta, edit) =>
      edit.end <= position ? delta + edit.text.length - (edit.end - edit.start) : delta,
    0,
  );
}

function deltaThrough(edits: readonly TextEdit[], position: number): number {
  return edits.reduce(
    (delta, edit) =>
      edit.start < position ? delta + edit.text.length - (edit.end - edit.start) : delta,
    0,
  );
}

function normalizeVariant(variant: Variant): Variant {
  return {
    ...variant,
    form: variant.form.normalize("NFC"),
  };
}

function joinedText(parts: readonly RenderedPartCore[]): string {
  return parts.map((part) => part.text).join("");
}
