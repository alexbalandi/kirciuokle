import { describe, expect, it } from "vitest";
import {
  buildEditedSentenceSpans,
  rebuildRenderedPartsWithFragments,
  retileRenderedParts,
  tokenizeForPreview,
} from "../src/client/preview";
import type { Part } from "../src/shared/types";

describe("tokenizeForPreview", () => {
  it("round-trips mixed punctuation, newlines, hyphenated words, and digits", () => {
    const text = "Labas, Jonai!\nEikime į kauną-vilnių 2026-07-07.";
    const parts = tokenizeForPreview(text);

    expect(parts.map((part) => part.text).join("")).toBe(text);
    expect(parts.every((part) => part.preview === true)).toBe(true);
    expect(parts.map((part) => [part.text, part.type])).toContainEqual([
      "kauną-vilnių",
      "word",
    ]);
    expect(parts.find((part) => part.text.includes("2026"))?.type).toBe("sep");
  });
});

describe("sentence-scoped rebuild helpers", () => {
  it("retile offsets and preserve reconstruction after editing the first sentence", () => {
    const oldText = "namas eina. Tu miegi.";
    const newText = "namelis eina. Tu miegi.";
    const edits = [{ start: 0, end: 5, text: "namelis" }];
    const oldParts = retileRenderedParts([
      { text: "namas", type: "word", accented: "nãmas" },
      { text: " ", type: "sep" },
      { text: "eina", type: "word", accented: "einà" },
      { text: ".", type: "sep" },
      { text: " ", type: "sep" },
      { text: "Tu", type: "word", accented: "Tù" },
      { text: " ", type: "sep" },
      { text: "miegi", type: "word", accented: "miẽgi" },
      { text: ".", type: "sep" },
    ]);
    const fragment: Part[] = [
      { text: "namelis", type: "word", accented: "namẽlis" },
      { text: " ", type: "sep" },
      { text: "eina", type: "word", accented: "einà" },
      { text: ".", type: "sep" },
    ];

    const spans = buildEditedSentenceSpans(oldText, oldParts, edits, newText);
    expect(spans).toEqual([{ oldStart: 0, oldEnd: 11, newStart: 0, newEnd: 13 }]);

    const rebuilt = rebuildRenderedPartsWithFragments(
      oldParts,
      spans ?? [],
      [fragment],
      newText,
    );

    expect(rebuilt?.map((part) => part.text).join("")).toBe(newText);
    expect(rebuilt?.find((part) => part.text === "Tu")?.sourceStart).toBe(14);
  });
});
