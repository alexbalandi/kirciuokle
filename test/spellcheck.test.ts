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

  it("ranks the common restoration first (aciu → ačiū, not ačiu)", () => {
    const restoreEngine = createSpellcheckEngine(["ačiū\t14413", "ačiu\t83"]);
    const suggestion = restoreEngine.suggest("aciu");

    expect(suggestion.status).toBe("restore");
    expect(suggestion.candidates[0]).toBe("ačiū");
    expect(suggestion.autofix).toBe("ačiū"); // dominant → auto-fixable
  });

  it("does not set autofix when restorations are close in frequency", () => {
    // both fold to "sasas"; neither dominates → needs a human choice
    const ambiguous = createSpellcheckEngine(["šašas\t50", "sašas\t48"]);
    const suggestion = ambiguous.suggest("sasas");

    expect(suggestion.status).toBe("restore");
    expect(suggestion.autofix).toBeUndefined();
  });
});

describe("SpellcheckEngine — real-text robustness", () => {
  it("accepts a word from the accept vocabulary (no false positive)", () => {
    const engine = createSpellcheckEngine(["skelbti\t24", "pramonės\t46"]);

    expect(engine.suggest("skelbti").status).toBe("ok");
    expect(engine.suggest("pramonės").status).toBe("ok");
  });

  it("restores an accepted ASCII word whose diacritic form is far more frequent", () => {
    // "as" is itself listed (freq 2297) but "aš" (116732) dominates → it's a drop.
    const engine = createSpellcheckEngine(["aš\t116732", "as\t2297"]);
    const suggestion = engine.suggest("as");

    expect(suggestion.status).toBe("restore");
    expect(suggestion.candidates).toContain("aš");
    expect(suggestion.autofix).toBe("aš");
  });

  it("keeps a valid ASCII word accepted when its diacritic sibling is rare", () => {
    // "padaryta" is a real word; "padarytą" (21) does not dominate → stay accepted.
    const engine = createSpellcheckEngine(["padaryta\t297", "padarytą\t21"]);

    expect(engine.suggest("padaryta").status).toBe("ok");
  });

  it("strips combining stress marks before lookup (accented paste)", () => {
    // ã / ù carry a stress mark; the un-stressed word is in the dictionary.
    expect(engine.suggest("nãmas").status).toBe("ok");
    expect(engine.suggest("žmogùs").status).toBe("ok");
  });

  it("does not flag single-letter words", () => {
    expect(engine.suggest("i")).toEqual({ status: "ok", candidates: [] });
    expect(engine.suggest("a")).toEqual({ status: "ok", candidates: [] });
  });

  it("suppresses typo suggestions for Capitalized and ALL-CAPS words", () => {
    const engine = createSpellcheckEngine(["namas\t50"]);

    expect(engine.suggest("namaz").status).toBe("typo"); // lowercase misspelling
    expect(engine.suggest("Namaz").status).not.toBe("typo"); // proper-noun-like
    expect(engine.suggest("NAMAZ").status).not.toBe("typo"); // acronym-like
  });

  it("still restores a Capitalized diacritic drop (sentence-initial Aciu)", () => {
    const engine = createSpellcheckEngine(["ačiū\t14413"]);
    const suggestion = engine.suggest("Aciu");

    expect(suggestion.status).toBe("restore");
    expect(suggestion.candidates).toContain("Ačiū");
  });

  it("uses the hunspell suggest provider for a correction the wordlist lacks", () => {
    // "kalbėti" is valid but too rare to be in the slimmed wordlist; hunspell both
    // accepts it and suggests it for the dropped-diacritic misspelling "kalbeti".
    const engine = createSpellcheckEngine(["namas\t50"]);
    engine.setAcceptPredicate((w) => w === "kalbėti");
    engine.setSuggestProvider((w) =>
      w === "kalbeti" ? ["kalbėti", "kalba", "labai negalima"] : [],
    );

    const suggestion = engine.suggest("kalbeti");
    expect(suggestion.status).toBe("typo");
    expect(suggestion.candidates[0]).toBe("kalbėti");
    // multi-token and far suggestions are filtered out
    expect(suggestion.candidates).not.toContain("labai negalima");
  });

  it("reaches a double error (wrong vowel + dropped diacritic) via the fold index", () => {
    // "pokalbeti" = "pakalbėti" with o→a AND the ė dropped. Its fold "pokalbeti" is
    // one substitution from "pakalbeti" = fold("pakalbėti"), so a fold-neighbour probe
    // finds it even though it's 2 edits away.
    const engine = createSpellcheckEngine(["pakalbėti\t633", "pokalbis\t251"]);
    engine.setAcceptPredicate((w) => w === "pakalbėti" || w === "pokalbis");

    const suggestion = engine.suggest("pokalbeti");
    expect(suggestion.status).toBe("typo");
    expect(suggestion.candidates).toContain("pakalbėti");
  });

  it("never offers a candidate that is itself invalid (drops wordlist noise)", () => {
    // "pakalbeti" (no ė) is diacritic-less corpus noise in the wordlist; the delete
    // index would surface it as a fix for "pakalbet", but it's not a real word.
    const engine = createSpellcheckEngine(["pakalbeti\t9"]);
    engine.setAcceptPredicate((w) => w !== "pakalbeti"); // hunspell rejects the junk

    expect(engine.suggest("pakalbet").candidates).not.toContain("pakalbeti");
  });

  it("does not call the suggest provider for an accepted word", () => {
    const engine = createSpellcheckEngine(["namas\t50"]);
    engine.setAcceptPredicate((w) => w === "namas");
    let calls = 0;
    engine.setSuggestProvider(() => {
      calls += 1;
      return [];
    });

    expect(engine.suggest("namas").status).toBe("ok");
    expect(calls).toBe(0);
  });

  it("uses an injected accept predicate (hunspell) over the wordlist", () => {
    // "sąjungininkių" isn't in the wordlist, but hunspell accepts it → not flagged.
    const engine = createSpellcheckEngine(["namas\t50"]);
    const accepted = new Set(["sąjungininkių", "namas"]);
    engine.setAcceptPredicate((w) => accepted.has(w));

    expect(engine.suggest("sąjungininkių").status).toBe("ok");
    // the predicate is case-aware: it receives the stress-stripped, cased word
    engine.setAcceptPredicate((w) => w === "Lietuvoje");
    expect(engine.suggest("Lietuvojè").status).toBe("ok"); // grave stripped, case kept
  });
});
