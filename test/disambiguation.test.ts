import { describe, expect, it } from "vitest";
import {
  alignTokens,
  parseConllu,
  parseMi,
  pickVariant,
  scoreTags,
  tokenTags,
  type AccentVariant,
  type Token,
} from "../src/worker/disambiguation";

describe("UDPipe CoNLL-U parsing", () => {
  it("parses token lines and skips comments, ranges, and empty nodes", () => {
    const tokens = parseConllu(`# sent_id = 1
1-2\tČia_yra\t_\t_\t_\t_
1\tČia\tčia\tADV\t_\tDegree=Pos
2\tyra\tbūti\tAUX\t_\tMood=Ind|Person=3|Tense=Pres|VerbForm=Fin
2.1\tpraleisti\tpraleisti\tVERB\t_\t_
3\t.\t.\tPUNCT\t_\t_`);

    expect(tokens).toEqual([
      {
        form: "Čia",
        lemma: "čia",
        upos: "ADV",
        xpos: "_",
        feats: { Degree: "Pos" },
      },
      {
        form: "yra",
        lemma: "būti",
        upos: "AUX",
        xpos: "_",
        feats: {
          Mood: "Ind",
          Person: "3",
          Tense: "Pres",
          VerbForm: "Fin",
        },
      },
      {
        form: ".",
        lemma: ".",
        upos: "PUNCT",
        xpos: "_",
        feats: {},
      },
    ]);
  });
});

describe("UDPipe alignment", () => {
  it("walks word-like VDU parts and scans ahead for tokenization mismatches", () => {
    const tokens = [
      token("Čia", "čia", "ADV"),
      token(",", ",", "PUNCT"),
      token("yra", "būti", "AUX"),
    ];

    expect(
      alignTokens(
        [
          { string: "Čia", type: "WORD" },
          { string: " ", type: "SEPARATOR" },
          { string: "yra", type: "WORD" },
          { string: "namas", type: "WORD" },
        ],
        tokens,
      ),
    ).toEqual([tokens[0], tokens[2], null]);
  });
});

describe("MI parsing and scoring", () => {
  it("parses VDU morphology labels longest-abbreviation first", () => {
    expect(parseMi("vksm., būt. d. l., 3 asm.")).toEqual({
      pos: "VERB",
      tense: "PastIter",
      person: "3",
    });
  });

  it("scores a masc-nom adjective over a fem-acc adjective in masc-nom context", () => {
    const context = tokenTags(
      token("geras", "geras", "ADJ", {
        Case: "Nom",
        Degree: "Pos",
        Gender: "Masc",
        Number: "Sing",
      }),
    );

    expect(scoreTags(parseMi("bdv., vyr. g., vns. vard."), context)).toBe(10);
    expect(scoreTags(parseMi("bdv., mot. g., vns. gal."), context)).toBe(2);
  });
});

describe("variant picking", () => {
  it("uses lemma exceptions before scoring same-morphology homographs", () => {
    const variants: AccentVariant[] = [
      { form: "ỹra", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
      { form: "yrà", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
    ];

    expect(
      pickVariant("yra", variants, token("yra", "būti", "AUX"), "ỹra"),
    ).toEqual({ index: 1, resolvedBy: "lemma" });
  });

  it("keeps the VDU default when scores tie", () => {
    const variants: AccentVariant[] = [
      { form: "ỹra", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
      { form: "yrà", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
    ];

    expect(
      pickVariant("yra", variants, token("yra", "nebūti", "AUX"), "yrà"),
    ).toEqual({ index: 1 });
  });
});

function token(
  form: string,
  lemma: string,
  upos: string,
  feats: Record<string, string> = {},
): Token {
  return { form, lemma, upos, xpos: "_", feats };
}
