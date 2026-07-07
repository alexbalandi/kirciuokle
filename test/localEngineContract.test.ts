import { describe, expect, expectTypeOf, it } from "vitest";
import type { Part } from "../src/shared/types";
import { partsFromDecodedSentences, tokenizeSurface } from "../src/client/local/engine";
import type {
  DecodedSentence,
  DecodedToken,
  SurfaceToken,
} from "../src/client/local/types";

describe("local engine Part[] contract", () => {
  it("returns the shared Part[] shape with local probabilities on variants", () => {
    const sentence: DecodedSentence = {
      index: 0,
      start: 0,
      end: 6,
      tokens: [
        { text: "Labas", start: 0, end: 5, isWord: true },
        { text: "!", start: 5, end: 6, isWord: false },
      ],
      decodedTokens: [
        {
          accented: "Lãbas",
          predicted: true,
          noStress: false,
          pos: [{ label: "dkt.", probability: 0.93 }],
        },
        {
          accented: "!",
          predicted: false,
          noStress: false,
          pos: [],
        },
      ],
    };

    const parts = partsFromDecodedSentences("Labas!", new Map([[0, sentence]]));
    expectTypeOf(parts).toEqualTypeOf<Part[]>();
    expect(parts).toEqual([
      {
        text: "Labas",
        type: "word",
        accented: "Lãbas",
        variants: [{ form: "Lãbas", info: "dkt.", p: 0.93 }],
        chosen: 0,
        chosenMi: "dkt.",
        tokenTags: { pos: "NOUN" },
        resolvedBy: "context",
      },
      { text: "!", type: "sep" },
    ]);
  });

  it("reconstructs multi-sentence, multi-paragraph text without duplicating whitespace", () => {
    // Regression: the pre-sentence gap was emitted once, then re-emitted by the
    // first token's own leading gap — doubling every space between sentences and
    // every "\n\n" between paragraphs. That made the result taller than the input,
    // so the 1:1 left/right scroll mirror drifted on long text (Local mode only).
    const text = "Aš. Tu.\n\nJis.";
    const word = (t: string, s: number, e: number): SurfaceToken => ({
      text: t,
      start: s,
      end: e,
      isWord: true,
    });
    const punc = (t: string, s: number, e: number): SurfaceToken => ({
      text: t,
      start: s,
      end: e,
      isWord: false,
    });
    const dw = (accented: string): DecodedToken => ({
      accented,
      predicted: true,
      noStress: false,
      pos: [],
    });
    const dp: DecodedToken = { accented: ".", predicted: false, noStress: false, pos: [] };
    const decoded = new Map<number, DecodedSentence>([
      [0, { index: 0, start: 0, end: 3, tokens: [word("Aš", 0, 2), punc(".", 2, 3)], decodedTokens: [dw("Àš"), dp] }],
      [1, { index: 1, start: 4, end: 7, tokens: [word("Tu", 4, 6), punc(".", 6, 7)], decodedTokens: [dw("Tù"), dp] }],
      [2, { index: 2, start: 9, end: 13, tokens: [word("Jis", 9, 12), punc(".", 12, 13)], decodedTokens: [dw("Jìs"), dp] }],
    ]);

    const parts = partsFromDecodedSentences(text, decoded);
    // The source text (word surfaces + separators) must reconstruct the input exactly.
    expect(parts.map((part) => part.text).join("")).toBe(text);
  });

  it("keeps hyphenated numeral suffixes as one non-popover display token", () => {
    const text = "81-erių vilnietė";
    const tokens = tokenizeSurface(text, 0);

    expect(tokens).toEqual([
      {
        text: "81-erių",
        start: 0,
        end: 7,
        isWord: true,
        modelText: "erių",
        accentableStart: 3,
        accentableEnd: 7,
        numeralFragment: true,
      },
      {
        text: "vilnietė",
        start: 8,
        end: 16,
        isWord: true,
      },
    ]);

    const sentence: DecodedSentence = {
      index: 0,
      start: 0,
      end: text.length,
      tokens,
      decodedTokens: [
        {
          accented: "81-er̃ių".normalize("NFC"),
          predicted: true,
          noStress: false,
          pos: [{ label: "dkt.", probability: 0.97 }],
        },
        {
          accented: "vilniẽtė".normalize("NFC"),
          predicted: true,
          noStress: false,
          pos: [{ label: "dkt.", probability: 0.93 }],
        },
      ],
    };

    const parts = partsFromDecodedSentences(text, new Map([[0, sentence]]));

    expect(parts.filter((part) => part.type === "word")).toHaveLength(2);
    expect(parts[0]).toEqual({
      text: "81-erių",
      type: "word",
      accented: "81-er̃ių".normalize("NFC"),
      numeralFragment: true,
    });
    expect(parts[0]).not.toHaveProperty("variants");
    expect(parts[0]).not.toHaveProperty("ambiguous");
    expect(parts[0]).not.toHaveProperty("tokenTags");
  });
});
