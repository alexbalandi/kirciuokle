import { afterEach, describe, expect, it, vi } from "vitest";
import type { AccentVariant } from "../src/worker/disambiguation";
import { toPublicVariants } from "../src/worker/disambiguation";
import { getWords } from "../src/worker/dictionary";
import worker from "../src/worker/index";
import { accentTextLocalFirst, tokenizeLikeVdu } from "../src/worker/localAccent";
import { accentText } from "../src/worker/vdu";
import { clearNonceCache } from "../src/worker/vdu";
import { captureWaitUntil, envFor, MemoryD1, stubVduFetch } from "./helpers";

const CIA_VARIANTS: AccentVariant[] = [
  { form: "čià", info: "prv.", mi: ["prv."] },
];

const YRA_VARIANTS: AccentVariant[] = [
  { form: "ỹra", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
  { form: "yrà", info: "vksm., es. l., 3 asm.", mi: ["vksm., es. l., 3 asm."] },
];

const WARM_TEXT = "C\u030Cia Velvet yra Москва.";
const WARM_TEXT_NFC = "Čia Velvet yra Москва.";
const WARM_CONLLU = `1\tČia\tčia\tADV\t_\t_
2\tVelvet\tvelvet\tPROPN\t_\t_
3\tyra\tbūti\tAUX\t_\tMood=Ind|Person=3|Tense=Pres|VerbForm=Fin
4\tМосква\tМосква\tPROPN\t_\t_`;

afterEach(() => {
  clearNonceCache();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("local-first accentuation", () => {
  it("matches legacy-normalized parts on a fully warm dictionary", async () => {
    const d1 = new MemoryD1();
    d1.setWord("čia", CIA_VARIANTS, { defaultForm: "čià", accentType: "ONE" });
    d1.setWord("velvet", [], { accentType: "NONE" });
    d1.setWord("yra", YRA_VARIANTS, {
      defaultForm: "ỹra",
      accentType: "MULTIPLE_MEANING",
    });
    stubVduFetch({
      conllu: WARM_CONLLU,
      textParts: [
        { string: "Čia", accented: "Čià", accentType: "ONE", type: "WORD" },
        { string: " ", type: "SEPARATOR" },
        { string: "Velvet", accentType: "NONE", type: "WORD" },
        { string: " ", type: "SEPARATOR" },
        {
          string: "yra",
          accented: "ỹra",
          accentType: "MULTIPLE_MEANING",
          type: "WORD",
        },
        { string: " ", type: "SEPARATOR" },
        { string: "Москва", type: "NON_LT" },
        { string: ".", type: "SEPARATOR" },
      ],
    });
    const { ctx, promises } = captureWaitUntil();

    const legacy = await accentText(WARM_TEXT_NFC, {
      lookupVariants: async (word) => (word === "yra" ? YRA_VARIANTS : []),
    });
    const local = await accentTextLocalFirst(WARM_TEXT, envFor(d1), ctx);

    expect(local.source).toBe("local");
    expect(local.tagger).toBe(legacy.tagger);
    // the local path additionally attaches reading info to plain words —
    // accentuation output itself must stay identical to the legacy path
    const stripInfo = (parts: typeof local.parts) =>
      parts.map(({ variants: _v, chosenMi: _m, ...rest }) => rest);
    expect(stripInfo(local.parts)).toEqual(stripInfo(legacy.parts));
    expect(local.parts).toEqual([
      {
        text: "Čia",
        accented: "Čià",
        type: "word",
        variants: [{ form: "čià", info: "prv." }],
        chosenMi: "prv.",
      },
      { text: " ", type: "sep" },
      { text: "Velvet", type: "word", unknown: true },
      { text: " ", type: "sep" },
      {
        text: "yra",
        accented: "yrà",
        type: "word",
        ambiguous: true,
        resolvedBy: "lemma",
        variants: [
          { form: "ỹra", info: "vksm., es. l., 3 asm." },
          { form: "yrà", info: "vksm., es. l., 3 asm." },
        ],
        chosen: 1,
        chosenMi: "vksm., es. l., 3 asm.",
      },
      { text: " ", type: "sep" },
      { text: "Москва", type: "word", unknown: true },
      { text: ".", type: "sep" },
    ]);
    expect(promises).toEqual([]);
  });

  it("keeps MULTIPLE_VARIANT hits plain and uses the stored default form", async () => {
    const d1 = new MemoryD1();
    d1.setWord(
      "pasiekia",
      [
        { form: "pasíekia", info: "vksm.", mi: ["vksm."] },
        { form: "pasiẽkia", info: "vksm.", mi: ["vksm."] },
      ],
      { defaultForm: "pasíekia", accentType: "MULTIPLE_VARIANT" },
    );
    const { ctx } = captureWaitUntil();

    const response = await accentTextLocalFirst("Pasiekia", envFor(d1), ctx, {
      useTagger: false,
    });

    expect(response.parts).toEqual([
      {
        text: "Pasiekia",
        accented: "Pasíekia",
        type: "word",
        variants: [
          { form: "pasíekia", info: "vksm." },
          { form: "pasiẽkia", info: "vksm." },
        ],
      },
    ]);
  });

  it("keeps ONE hits plain even when word_accent has suppressed extra readings", async () => {
    const d1 = new MemoryD1();
    d1.setWord(
      "kas",
      [
        { form: "kàs", info: "įv.", mi: ["įv."] },
        { form: "kas", info: "dll.", mi: ["dll."] },
      ],
      { defaultForm: "kàs", accentType: "ONE" },
    );
    const { ctx } = captureWaitUntil();

    const response = await accentTextLocalFirst("Kas kas", envFor(d1), ctx, {
      useTagger: false,
    });

    const kasVariants = [
      { form: "kàs", info: "įv." },
      { form: "kas", info: "dll." },
    ];
    expect(response.parts).toEqual([
      { text: "Kas", accented: "Kàs", type: "word", variants: kasVariants },
      { text: " ", type: "sep" },
      { text: "kas", accented: "kàs", type: "word", variants: kasVariants },
    ]);
  });

  it("uses case-sensitive canonical sides for alyta and Alyta", async () => {
    const d1 = new MemoryD1();
    d1.setWord(
      "alyta",
      [{ form: "Alytà", info: "dkt.", mi: ["dkt."] }],
      {
        defaultForm: null,
        accentType: "NONE",
        defaultFormTitle: "Alytà",
        accentTypeTitle: "MULTIPLE_MEANING",
      },
    );
    const { ctx } = captureWaitUntil();

    const response = await accentTextLocalFirst("alyta Alyta", envFor(d1), ctx, {
      useTagger: false,
    });

    expect(response.parts).toEqual([
      { text: "alyta", type: "word", unknown: true },
      { text: " ", type: "sep" },
      {
        // a single distinct accented form is not a choice — no ambiguity flag
        text: "Alyta",
        accented: "Alytà",
        type: "word",
        variants: [{ form: "Alytà", info: "dkt." }],
      },
    ]);
  });

  it("dedupes repeated readings in public variant info", () => {
    expect(
      toPublicVariants([{ form: "põ", info: "prl.; prl.; prl.; prl.", mi: ["prl."] }]),
    ).toEqual([{ form: "põ", info: "prl." }]);
  });

  it("keeps lowercase vilnius variant plain but capitalized Vilnius ambiguous", async () => {
    const d1 = new MemoryD1();
    d1.setWord(
      "vilnius",
      [
        { form: "vìlnius", info: "dkt.", mi: ["dkt."] },
        { form: "vil̃nius", info: "dkt.", mi: ["dkt."] },
      ],
      {
        defaultForm: "vìlnius",
        accentType: "MULTIPLE_VARIANT",
        defaultFormTitle: "Vìlnius",
        accentTypeTitle: "MULTIPLE_MEANING",
      },
    );
    const { ctx } = captureWaitUntil();

    const response = await accentTextLocalFirst("vilnius Vilnius", envFor(d1), ctx, {
      useTagger: false,
    });

    expect(response.parts).toEqual([
      {
        text: "vilnius",
        accented: "vìlnius",
        type: "word",
        variants: [
          { form: "vìlnius", info: "dkt." },
          { form: "vil̃nius", info: "dkt." },
        ],
      },
      { text: " ", type: "sep" },
      {
        text: "Vilnius",
        accented: "Vìlnius",
        type: "word",
        ambiguous: true,
        variants: [
          { form: "vìlnius", info: "dkt." },
          { form: "vil̃nius", info: "dkt." },
        ],
        chosen: 0,
      },
    ]);
  });

  it("emits uppercase Roman numerals as plain words without lookup flags", async () => {
    const d1 = new MemoryD1();
    const { ctx, promises } = captureWaitUntil();

    const response = await accentTextLocalFirst("II", envFor(d1), ctx, {
      useTagger: false,
    });

    expect(response.parts).toEqual([{ text: "II", type: "word" }]);
    expect(d1.selectBinds).toEqual([]);
    expect(promises).toEqual([]);
  });

  it("leaves dot abbreviations unknown without lookup before Roman numerals", async () => {
    const d1 = new MemoryD1();
    d1.setWord("kalba", [{ form: "kalbà", info: "dkt.", mi: ["dkt."] }], {
      defaultForm: "kalbà",
      accentType: "ONE",
      defaultFormTitle: "Kalbà",
      accentTypeTitle: "ONE",
    });
    const { ctx, promises } = captureWaitUntil();

    const response = await accentTextLocalFirst(
      "m. rus. V. kalba XX a.",
      envFor(d1),
      ctx,
      { useTagger: false },
    );

    expect(response.parts).toEqual([
      { text: "m.", type: "word", unknown: true },
      { text: " ", type: "sep" },
      { text: "rus.", type: "word", unknown: true },
      { text: " ", type: "sep" },
      { text: "V.", type: "word", unknown: true },
      { text: " ", type: "sep" },
      {
        text: "kalba",
        accented: "kalbà",
        type: "word",
        variants: [{ form: "kalbà", info: "dkt." }],
      },
      { text: " ", type: "sep" },
      { text: "XX", type: "word" },
      { text: " ", type: "sep" },
      { text: "a.", type: "word", unknown: true },
    ]);
    expect(d1.selectBinds).toEqual([["kalba"]]);
    expect(promises).toEqual([]);
  });

  it("refetches a legacy 5b row with NULL title columns and rewrites it complete", async () => {
    const d1 = new MemoryD1();
    d1.setWord("kalba", [{ form: "kalbà", info: "dkt.", mi: ["dkt."] }], {
      defaultForm: "kalbà",
      accentType: "ONE",
      defaultFormTitle: null,
      accentTypeTitle: null,
    });
    const fetchMock = stubVduFetch({
      wordResponses: {
        kalba: {
          accentInfo: [
            {
              accented: ["kalba\u0300"],
              information: [{ mi: "dkt.", meaning: "kalba" }],
            },
          ],
        },
      },
      textResponses: {
        kalba: [
          {
            string: "kalba",
            accented: "kalba\u0300",
            accentType: "ONE",
            type: "WORD",
          },
        ],
        Kalba: [
          {
            string: "Kalba",
            accented: "Kalba\u0300",
            accentType: "ONE",
            type: "WORD",
          },
        ],
      },
    });
    const { ctx, promises } = captureWaitUntil();

    const response = await accentTextLocalFirst("kalba", envFor(d1), ctx, {
      useTagger: false,
    });
    await Promise.all(promises);

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(response.parts).toEqual([
      {
        text: "kalba",
        accented: "kalbà",
        type: "word",
        variants: [{ form: "kalbà", info: "dkt. - kalba" }],
      },
    ]);
    expect(d1.getEntry("kalba")).toEqual({
      variants: [{ form: "kalbà", info: "dkt. - kalba", mi: ["dkt."] }],
      defaultForm: "kalbà",
      accentType: "ONE",
      defaultFormTitle: "Kalbà",
      accentTypeTitle: "ONE",
    });
  });

  it("falls back to legacy text accents over the miss budget and still seeds misses", async () => {
    const d1 = new MemoryD1();
    const words = Array.from({ length: 11 }, (_, index) =>
      String.fromCharCode(97 + Math.floor(index / 26)) +
      String.fromCharCode(97 + (index % 26)),
    );
    stubVduFetch({
      textParts: [
        { string: "fallback", accented: "fallback", accentType: "ONE", type: "WORD" },
      ],
    });
    const { ctx, promises } = captureWaitUntil();

    const response = await accentTextLocalFirst(
      words.join(" "),
      envFor(d1),
      ctx,
      { useTagger: false },
    );
    await Promise.all(promises);

    expect(response.source).toBe("vdu");
    expect(response.parts).toEqual([
      { text: "fallback", accented: "fallback", type: "word" },
    ]);
    expect(d1.store.size).toBe(10);
    expect((await getWords(envFor(d1), words.slice(0, 10))).size).toBe(10);
  });

  it("lets the accent source be overridden per request", async () => {
    const d1 = new MemoryD1();
    d1.setWord("čia", CIA_VARIANTS, { defaultForm: "čià", accentType: "ONE" });
    stubVduFetch({
      textParts: [
        { string: "Čia", accented: "Čià", accentType: "ONE", type: "WORD" },
      ],
    });
    const localCtx = captureWaitUntil();
    const localResponse = await worker.fetch(
      new Request("http://example.test/api/accent?source=local", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: "Čia" }),
      }),
      envFor(d1, "vdu"),
      localCtx.ctx as ExecutionContext,
    );

    await expect(localResponse.json()).resolves.toMatchObject({
      source: "local",
      parts: [{ text: "Čia", accented: "Čià", type: "word" }],
    });

    const vduCtx = captureWaitUntil();
    const vduResponse = await worker.fetch(
      new Request("http://example.test/api/accent?source=vdu", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: "Čia" }),
      }),
      envFor(d1, "local"),
      vduCtx.ctx as ExecutionContext,
    );

    await expect(vduResponse.json()).resolves.toMatchObject({
      source: "vdu",
      parts: [{ text: "Čia", accented: "Čià", type: "word" }],
    });
  });
});

describe("tokenizeLikeVdu edge cases", () => {
  it("keeps pre-accented words whole and untouched (NON_LT)", () => {
    // "mė́nuo" carries a combining acute (no precomposed form exists for ė́)
    const { textParts, lookupWords } = tokenizeLikeVdu("Tas mė́nuo baigėsi.");

    expect(textParts.map((p) => [p.string, p.type])).toEqual([
      ["Tas", "WORD"],
      [" ", "SEPARATOR"],
      ["mė́nuo".normalize("NFC"), "NON_LT"],
      [" ", "SEPARATOR"],
      ["baigėsi", "WORD"],
      [".", "SEPARATOR"],
    ]);
    expect(lookupWords.map((w) => w.key)).toEqual(["tas", "baigėsi"]);
  });

  it("treats pers. as an abbreviation", () => {
    const { textParts, lookupWords } = tokenizeLikeVdu("pers. Litva");

    expect(textParts[0]).toMatchObject({ string: "pers.", accentType: "NONE" });
    expect(lookupWords.map((w) => w.key)).toEqual(["litva"]);
  });
});
