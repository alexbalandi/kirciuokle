import { afterEach, describe, expect, it, vi } from "vitest";
import type { AccentVariant } from "../src/worker/disambiguation";
import {
  getWords,
  lookupWordVariantsD1,
  NEGATIVE_WORD_TTL_MS,
  putWords,
} from "../src/worker/dictionary";
import { clearNonceCache } from "../src/worker/vdu";
import { captureWaitUntil, envFor, MemoryD1, stubVduFetch } from "./helpers";

const YRA_VARIANTS: AccentVariant[] = [
  { form: "ỹra", info: "vksm.", mi: ["vksm."] },
];

afterEach(() => {
  clearNonceCache();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("D1 word dictionary", () => {
  it("batch-reads more than 90 words and returns null for absent rows", async () => {
    const d1 = new MemoryD1();
    d1.setWord("w90", YRA_VARIANTS);

    const words = Array.from({ length: 91 }, (_, index) => `w${index}`);
    const result = await getWords(envFor(d1), words);

    expect(d1.selectBinds.map((binds) => binds.length)).toEqual([90, 1]);
    expect(result.get("w90")).toEqual({
      variants: YRA_VARIANTS,
      defaultForm: "ỹra",
      accentType: "ONE",
    });
    expect(result.get("w0")).toBeNull();
  });

  it("batch-writes within the 90-parameter limit and stores negatives for 30 days", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-02T00:00:00.000Z"));
    const d1 = new MemoryD1();
    const entries = Array.from({ length: 23 }, (_, index) => ({
      word: `w${index}`,
      variants: index === 22 ? [] : YRA_VARIANTS,
      defaultForm: index === 22 ? null : "ỹra",
      accentType: index === 22 ? null : "ONE",
    }));

    await putWords(envFor(d1), entries);

    expect(d1.insertBinds.map((binds) => binds.length)).toEqual([90, 48]);
    expect(d1.getVariants("w22")).toEqual([]);
    expect(Date.parse(d1.getNegativeUntil("w22") ?? "")).toBe(
      Date.parse("2026-07-02T00:00:00.000Z") + NEGATIVE_WORD_TTL_MS,
    );
    expect(d1.getNegativeUntil("w0")).toBeNull();
  });

  it("treats expired negatives as absent and valid negatives as hits", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-02T00:00:00.000Z"));
    const d1 = new MemoryD1();
    d1.setWord("old", [], { negativeUntil: "2026-07-01T00:00:00.000Z" });
    d1.setWord("fresh", [], { negativeUntil: "2026-07-03T00:00:00.000Z" });

    const result = await getWords(envFor(d1), ["old", "fresh"]);

    expect(result.get("old")).toBeNull();
    expect(result.get("fresh")).toEqual({
      variants: [],
      defaultForm: null,
      accentType: null,
    });
  });

  it("treats positive rows with NULL accent_type as incomplete misses", async () => {
    const d1 = new MemoryD1();
    d1.setWord("yra", YRA_VARIANTS, { defaultForm: null, accentType: null });
    const fetchMock = stubVduFetch({
      wordResponses: {
        yra: {
          accentInfo: [
            {
              accented: ["yra\u0300"],
              information: [{ mi: "vksm.", meaning: "būti" }],
            },
          ],
        },
      },
      textParts: [
        {
          string: "yra",
          accented: "yra\u0300",
          accentType: "ONE",
          type: "WORD",
        },
      ],
    });
    const { ctx, promises } = captureWaitUntil();

    const variants = await lookupWordVariantsD1("yra", envFor(d1), ctx);
    await Promise.all(promises);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(variants).toEqual([
      { form: "yrà", info: "vksm. - būti", mi: ["vksm."] },
    ]);
    expect(d1.getEntry("yra")).toEqual({
      variants,
      defaultForm: "yrà",
      accentType: "ONE",
    });
  });

  it("returns D1 hits without fetching VDU", async () => {
    const d1 = new MemoryD1();
    d1.setWord("yra", YRA_VARIANTS);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ctx, promises } = captureWaitUntil();

    await expect(lookupWordVariantsD1("YRA", envFor(d1), ctx)).resolves.toEqual(
      YRA_VARIANTS,
    );

    expect(fetchMock).not.toHaveBeenCalled();
    expect(promises).toEqual([]);
  });

  it("fetches VDU misses and writes positive results to D1", async () => {
    const d1 = new MemoryD1();
    const fetchMock = stubVduFetch({
      wordResponses: {
        yra: {
          accentInfo: [
            {
              accented: ["y\u0303ra"],
              information: [{ mi: "vksm.", meaning: "būti" }],
            },
          ],
        },
      },
      textParts: [
        {
          string: "yra",
          accented: "y\u0303ra",
          accentType: "MULTIPLE_MEANING",
          type: "WORD",
        },
      ],
    });
    const { ctx, promises, waitUntil } = captureWaitUntil();

    const variants = await lookupWordVariantsD1("YRA", envFor(d1), ctx);
    await Promise.all(promises);

    expect(variants).toEqual([
      { form: "ỹra", info: "vksm. - būti", mi: ["vksm."] },
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const postInit = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(String(postInit.body)).toContain("word=yra");
    expect(waitUntil).toHaveBeenCalledTimes(1);
    expect(d1.getVariants("yra")).toEqual(variants);
    expect(d1.getEntry("yra")).toEqual({
      variants,
      defaultForm: "ỹra",
      accentType: "MULTIPLE_MEANING",
    });
  });

  it("writes empty VDU results to D1 as valid negatives", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-02T00:00:00.000Z"));
    const d1 = new MemoryD1();
    stubVduFetch({ wordResponses: { velvet: {} } });
    const { ctx, promises } = captureWaitUntil();

    await expect(lookupWordVariantsD1("velvet", envFor(d1), ctx)).resolves.toEqual(
      [],
    );
    await Promise.all(promises);

    expect(d1.getVariants("velvet")).toEqual([]);
    expect(Date.parse(d1.getNegativeUntil("velvet") ?? "")).toBe(
      Date.parse("2026-07-02T00:00:00.000Z") + NEGATIVE_WORD_TTL_MS,
    );
  });
});
