import { afterEach, describe, expect, it, vi } from "vitest";
import type { AccentVariant } from "../src/worker/disambiguation";
import { getWords } from "../src/worker/dictionary";
import worker from "../src/worker/index";
import { accentTextLocalFirst } from "../src/worker/localAccent";
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
    expect(local.parts).toEqual(legacy.parts);
    expect(local.parts).toEqual([
      { text: "Čia", accented: "Čià", type: "word" },
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
      { text: "Pasiekia", accented: "Pasíekia", type: "word" },
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

    expect(response.parts).toEqual([
      { text: "Kas", accented: "Kàs", type: "word" },
      { text: " ", type: "sep" },
      { text: "kas", accented: "kàs", type: "word" },
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

  it("falls back to legacy text accents over the miss budget and still seeds misses", async () => {
    const d1 = new MemoryD1();
    const words = Array.from({ length: 16 }, (_, index) =>
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
    expect(d1.store.size).toBe(15);
    expect((await getWords(envFor(d1), words.slice(0, 15))).size).toBe(15);
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
