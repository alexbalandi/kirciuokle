import { describe, expect, it } from "vitest";
import {
  bestMiForSlots,
  buildLabelBridgeCache,
  decodePosRows,
} from "../src/client/local/bridge";
import type { LabelBridge } from "../src/client/local/types";

describe("local label bridge", () => {
  it("uses the fewest-spurious-slot tie-break", () => {
    const miVocab = [
      { label: "dkt., vyr. g., vns.", slots: { pos: "NOUN", gender: "Masc", number: "Sing" } },
      { label: "dkt., vyr. g.", slots: { pos: "NOUN", gender: "Masc" } },
    ];

    expect(bestMiForSlots({ pos: "NOUN", gender: "Masc" }, miVocab)).toBe(
      "dkt., vyr. g.",
    );
  });

  it("caches one best mi label per model label id", () => {
    const bridge: LabelBridge = {
      mi_vocab: [{ label: "dkt.", slots: { pos: "NOUN" } }],
      model_labels: {
        "NOUN|Case=Nom": { pos: "NOUN", case: "Nom" },
      },
    };

    const cache = buildLabelBridgeCache(bridge, ["NOUN|Case=Nom"]);

    expect(cache.size).toBe(1);
    expect(cache.get("NOUN|Case=Nom")).toBe("dkt.");
  });

  it("merges duplicate mi strings by summed probability", () => {
    const labels = ["A", "B", "C"];
    const cache = new Map([
      ["A", "dkt."],
      ["B", "dkt."],
      ["C", "bdv."],
    ]);
    const logits = [2, 1, 0];
    const denom = Math.exp(2) + Math.exp(1) + Math.exp(0);

    const rows = decodePosRows(logits, 0, labels, cache, {
      probabilityCut: 0,
      maxRows: 5,
    });

    expect(rows[0]?.label).toBe("dkt.");
    expect(rows[0]?.probability).toBeCloseTo((Math.exp(2) + Math.exp(1)) / denom);
    expect(rows[1]?.label).toBe("bdv.");
    expect(rows[1]?.probability).toBeCloseTo(Math.exp(0) / denom);
  });
});
