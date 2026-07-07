import { describe, expect, it } from "vitest";
import { buildBatches, restoreSentenceOrder } from "../src/client/local/batching";

describe("local inference batching", () => {
  it("respects the padded token budget for multi-sentence batches", () => {
    const sentences = [
      { index: 0, subwordLength: 5 },
      { index: 1, subwordLength: 4 },
      { index: 2, subwordLength: 3 },
      { index: 3, subwordLength: 2 },
    ];
    const batches = buildBatches(sentences, 8);

    for (const batch of batches) {
      const cost = Math.max(...batch.map((sentence) => sentence.subwordLength)) * batch.length;
      expect(cost).toBeLessThanOrEqual(8);
    }
  });

  it("isolates a single over-budget sentence", () => {
    const batches = buildBatches(
      [
        { index: 0, subwordLength: 4 },
        { index: 1, subwordLength: 13 },
        { index: 2, subwordLength: 4 },
      ],
      8,
    );

    expect(batches[0]).toEqual([{ index: 1, subwordLength: 13 }]);
    expect(batches[0]!.length).toBe(1);
  });

  it("restores original sentence order", () => {
    const batches = buildBatches(
      [
        { index: 0, subwordLength: 3 },
        { index: 1, subwordLength: 9 },
        { index: 2, subwordLength: 5 },
      ],
      10,
    );
    const decoded = batches.flat().map((sentence) => ({
      index: sentence.index,
      value: `s${sentence.index}`,
    }));

    expect(restoreSentenceOrder(decoded).map((sentence) => sentence.value)).toEqual([
      "s0",
      "s1",
      "s2",
    ]);
  });

  it("is deterministic", () => {
    const sentences = [
      { index: 0, subwordLength: 3 },
      { index: 1, subwordLength: 7 },
      { index: 2, subwordLength: 7 },
      { index: 3, subwordLength: 2 },
    ];

    expect(buildBatches(sentences, 12)).toEqual(buildBatches(sentences, 12));
  });
});
