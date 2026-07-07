import { describe, expect, it } from "vitest";
import { createSpellcheckEngine, foldAscii } from "../src/client/spellcheck";

const engine = createSpellcheckEngine([
  "ačiū",
  "namas",
  "žmogus",
  "žmonės",
]);

describe("foldAscii", () => {
  it("folds Lithuanian diacritics and lowercases", () => {
    expect(foldAscii("ĄČĘĖĮŠŲŪŽ")).toBe("aceeisuuz");
    expect(foldAscii("Žmogus")).toBe("zmogus");
  });
});

describe("SpellcheckEngine.suggest", () => {
  it("restores dropped Lithuanian diacritics", () => {
    const suggestion = engine.suggest("zmogus");

    expect(suggestion.status).toBe("restore");
    expect(suggestion.candidates).toContain("žmogus");
  });

  it("accepts valid words case-insensitively", () => {
    expect(engine.suggest("Žmogus")).toEqual({ status: "ok", candidates: [] });
  });

  it("suggests edit-distance-1 typo corrections", () => {
    const suggestion = engine.suggest("žmogud");

    expect(suggestion.status).toBe("typo");
    expect(suggestion.candidates).toContain("žmogus");
  });

  it("leaves gibberish unknown", () => {
    expect(engine.suggest("qzxqzx")).toEqual({
      status: "unknown",
      candidates: [],
    });
  });

  it("reapplies the query case to candidates", () => {
    expect(engine.suggest("Zmogus").candidates).toContain("Žmogus");
    expect(engine.suggest("ZMOGUS").candidates).toContain("ŽMOGUS");
  });

  it("ranks restorations by frequency within the same edit band", () => {
    const frequencyEngine = createSpellcheckEngine(["aš\t100", "ąs\t1"]);

    expect(frequencyEngine.suggest("as")).toMatchObject({
      status: "restore",
      candidates: ["aš", "ąs"],
    });
  });

  it("suggests edit-distance-2 typo corrections and caps longer misses", () => {
    const typoEngine = createSpellcheckEngine(["žmogus"]);

    expect(typoEngine.suggest("žmogusxx")).toMatchObject({
      status: "typo",
      candidates: ["žmogus"],
    });
    expect(typoEngine.suggest("žmogusxxx")).toEqual({
      status: "unknown",
      candidates: [],
    });
  });

  it("counts a neighbouring transposition as edit distance 1", () => {
    const typoEngine = createSpellcheckEngine(["diena"]);

    expect(typoEngine.suggest("deina")).toMatchObject({
      status: "typo",
      candidates: ["diena"],
    });
  });

  it("uses loaded bigrams as a context tie-break before frequency", () => {
    const contextEngine = createSpellcheckEngine(
      ["aš\t100", "ąs\t1"],
      ["ir\tąs\t10"],
    );

    expect(contextEngine.suggest("as", { prev: "ir" }).candidates).toEqual([
      "ąs",
      "aš",
    ]);
  });
});
