import { afterEach, describe, expect, it, vi } from "vitest";
import type { AccentVariant } from "../src/worker/disambiguation";
import {
  accentText,
  clearNonceCache,
  extractNonce,
  flattenVariants,
  lookupWordVariantsKV,
  NEGATIVE_WORD_TTL_SECONDS,
  normalizeTextParts,
  splitTextIntoChunks,
} from "../src/worker/vdu";

type TestVduWordResponse = {
  accentInfo?: Array<{
    accented?: string[];
    information?: Array<{
      mi?: string;
      meaning?: string;
    }>;
  }>;
};

type CapturedPut = {
  key: string;
  value: string;
  options?: KVNamespacePutOptions;
};

class MemoryKV {
  readonly puts: CapturedPut[] = [];
  readonly store = new Map<string, string>();

  async get(key: string, type?: "json" | "text"): Promise<unknown> {
    const value = this.store.get(key);
    if (value === undefined) {
      return null;
    }

    if (type === "json") {
      return JSON.parse(value);
    }

    return value;
  }

  async put(
    key: string,
    value: string | ArrayBuffer | ArrayBufferView | ReadableStream,
    options?: KVNamespacePutOptions,
  ): Promise<void> {
    if (typeof value !== "string") {
      throw new Error("MemoryKV only supports string values.");
    }

    this.puts.push({ key, value, options });
    this.store.set(key, value);
  }
}

function envFor(kv: MemoryKV): { WORDS: KVNamespace } {
  return { WORDS: kv as unknown as KVNamespace };
}

function captureWaitUntil(): {
  ctx: Pick<ExecutionContext, "waitUntil">;
  promises: Promise<unknown>[];
  waitUntil: ReturnType<typeof vi.fn>;
} {
  const promises: Promise<unknown>[] = [];
  const waitUntil = vi.fn((promise: Promise<unknown>) => {
    promises.push(promise);
  });

  return { ctx: { waitUntil }, promises, waitUntil };
}

function stubWordAccentFetch(response: TestVduWordResponse) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = String(init?.body ?? "");

    if (url.includes("kirciuoklis")) {
      return new Response('<script>{"NONCE":"abcdef123456"}</script>');
    }

    if (body.includes("action=word_accent")) {
      return Response.json({
        code: 200,
        message: JSON.stringify(response),
      });
    }

    throw new Error(`Unexpected fetch: ${url} ${body}`);
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  clearNonceCache();
  vi.unstubAllGlobals();
});

describe("splitTextIntoChunks", () => {
  it("prefers sentence boundaries", () => {
    const chunks = splitTextIntoChunks("Pirmas sakinys. Antras sakinys!", 20);

    expect(chunks).toEqual(["Pirmas sakinys.", " Antras sakinys!"]);
    expect(chunks.every((chunk) => chunk.length <= 20)).toBe(true);
  });

  it("falls back to spaces", () => {
    const chunks = splitTextIntoChunks("vienas du trys keturi", 12);

    expect(chunks).toEqual(["vienas du ", "trys keturi"]);
  });

  it("hard-cuts long words", () => {
    const chunks = splitTextIntoChunks("abcdefghijkl", 5);

    expect(chunks).toEqual(["abcde", "fghij", "kl"]);
  });
});

describe("VDU response helpers", () => {
  it("extracts the WordPress nonce", () => {
    expect(extractNonce('<script>{"NONCE":"012345abcdef"}</script>')).toBe(
      "012345abcdef",
    );
  });

  it("normalizes text parts into the worker API shape", () => {
    expect(
      normalizeTextParts([
        {
          string: "Čia",
          accented: "Čia\u0300",
          accentType: "ONE",
          type: "WORD",
        },
        { string: " ", type: "SEPARATOR" },
        {
          string: "yra",
          accented: "y\u0303ra",
          accentType: "MULTIPLE_MEANING",
          type: "WORD",
        },
        { string: "Velvet", accentType: "NONE", type: "WORD" },
        { string: "Waver", type: "NON_LT" },
      ]),
    ).toEqual([
      { text: "Čia", accented: "Čià", type: "word" },
      { text: " ", type: "sep" },
      { text: "yra", accented: "ỹra", type: "word", ambiguous: true },
      { text: "Velvet", type: "word", unknown: true },
      { text: "Waver", type: "word", unknown: true },
    ]);
  });

  it("flattens variant forms with morphology and meaning", () => {
    expect(
      flattenVariants({
        accentInfo: [
          {
            accented: ["y\u0303ra"],
            information: [{ mi: "vksm." }, { mi: "3 asm.", meaning: "būti" }],
          },
          {
            accented: ["yra\u0300", "yr\u0300a"],
            information: [{ mi: "dktv.", meaning: "skylė" }],
          },
        ],
      }),
    ).toEqual([
      { form: "ỹra", info: "vksm.; 3 asm. - būti", mi: ["vksm.", "3 asm."] },
      { form: "yrà", info: "dktv. - skylė", mi: ["dktv."] },
      { form: "yr̀a", info: "dktv. - skylė", mi: ["dktv."] },
    ]);
  });

  it("uses mocked fetch for upstream accent calls", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response('<script>{"NONCE":"abcdef123456"}</script>'),
      )
      .mockResolvedValueOnce(
        Response.json({
          code: 200,
          message: JSON.stringify({
            textParts: [
              {
                string: "Čia",
                accented: "Čia\u0300",
                accentType: "ONE",
                type: "WORD",
              },
            ],
          }),
        }),
      );

    vi.stubGlobal("fetch", fetchMock);

    await expect(accentText("Čia", { useTagger: false })).resolves.toEqual({
      tagger: "unavailable",
      parts: [{ text: "Čia", accented: "Čià", type: "word" }],
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const postInit = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(postInit.method).toBe("POST");
    expect(String(postInit.body)).toContain("action=text_accents");
    expect(String(postInit.body)).toContain("nonce=abcdef123456");
    expect(String(postInit.body)).toContain("body=%C4%8Cia");
  });

  it("degrades to VDU defaults when the tagger is unavailable", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const body = String(init?.body ?? "");

      if (url.includes("lindat.mff.cuni.cz")) {
        throw new Error("tagger down");
      }

      if (url.includes("kirciuoklis")) {
        return new Response('<script>{"NONCE":"abcdef123456"}</script>');
      }

      if (body.includes("action=text_accents")) {
        return Response.json({
          code: 200,
          message: JSON.stringify({
            textParts: [
              {
                string: "Čia",
                accented: "Čia\u0300",
                accentType: "ONE",
                type: "WORD",
              },
              { string: " ", type: "SEPARATOR" },
              {
                string: "yra",
                accented: "y\u0303ra",
                accentType: "MULTIPLE_MEANING",
                type: "WORD",
              },
            ],
          }),
        });
      }

      if (body.includes("action=word_accent")) {
        return Response.json({
          code: 200,
          message: JSON.stringify({
            accentInfo: [
              {
                accented: ["y\u0303ra"],
                information: [{ mi: "vksm., es. l., 3 asm." }],
              },
              {
                accented: ["yra\u0300"],
                information: [{ mi: "vksm., es. l., 3 asm." }],
              },
            ],
          }),
        });
      }

      throw new Error(`Unexpected fetch: ${url} ${body}`);
    });

    vi.stubGlobal("fetch", fetchMock);

    await expect(accentText("Čia yra")).resolves.toEqual({
      tagger: "unavailable",
      parts: [
        { text: "Čia", accented: "Čià", type: "word" },
        { text: " ", type: "sep" },
        {
          text: "yra",
          accented: "ỹra",
          type: "word",
          ambiguous: true,
          variants: [
            { form: "ỹra", info: "vksm., es. l., 3 asm." },
            { form: "yrà", info: "vksm., es. l., 3 asm." },
          ],
          chosen: 0,
        },
      ],
    });
  });
});

describe("KV word dictionary", () => {
  it("returns KV hits without fetching VDU", async () => {
    const kv = new MemoryKV();
    const variants: AccentVariant[] = [
      { form: "ỹra", info: "vksm.", mi: ["vksm."] },
    ];
    kv.store.set(
      "yra",
      JSON.stringify({ variants, fetchedAt: "2026-07-02T00:00:00.000Z" }),
    );
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ctx, promises } = captureWaitUntil();

    await expect(lookupWordVariantsKV("YRA", envFor(kv), ctx)).resolves.toEqual(
      variants,
    );

    expect(fetchMock).not.toHaveBeenCalled();
    expect(kv.puts).toEqual([]);
    expect(promises).toEqual([]);
  });

  it("fetches VDU misses and writes positive results to KV", async () => {
    const kv = new MemoryKV();
    const fetchMock = stubWordAccentFetch({
      accentInfo: [
        {
          accented: ["y\u0303ra"],
          information: [{ mi: "vksm.", meaning: "būti" }],
        },
      ],
    });
    const { ctx, promises, waitUntil } = captureWaitUntil();

    const variants = await lookupWordVariantsKV("YRA", envFor(kv), ctx);
    await Promise.all(promises);

    expect(variants).toEqual([
      { form: "ỹra", info: "vksm. - būti", mi: ["vksm."] },
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const postInit = fetchMock.mock.calls[1]?.[1] as RequestInit;
    expect(String(postInit.body)).toContain("word=yra");
    expect(waitUntil).toHaveBeenCalledTimes(1);
    expect(kv.puts).toHaveLength(1);
    expect(kv.puts[0]?.key).toBe("yra");
    expect(kv.puts[0]?.options).toBeUndefined();
    const stored = JSON.parse(kv.store.get("yra") ?? "") as {
      variants: AccentVariant[];
      fetchedAt: string;
    };
    expect(stored.variants).toEqual(variants);
    expect(Number.isNaN(Date.parse(stored.fetchedAt))).toBe(false);
  });

  it("writes empty VDU results to KV with a negative TTL", async () => {
    const kv = new MemoryKV();
    stubWordAccentFetch({});
    const { ctx, promises } = captureWaitUntil();

    await expect(lookupWordVariantsKV("velvet", envFor(kv), ctx)).resolves.toEqual(
      [],
    );
    await Promise.all(promises);

    expect(kv.puts).toHaveLength(1);
    expect(kv.puts[0]?.key).toBe("velvet");
    expect(kv.puts[0]?.options).toEqual({
      expirationTtl: NEGATIVE_WORD_TTL_SECONDS,
    });
    const stored = JSON.parse(kv.store.get("velvet") ?? "") as {
      variants: AccentVariant[];
    };
    expect(stored.variants).toEqual([]);
  });

  it("treats malformed KV JSON as a miss and overwrites it", async () => {
    const kv = new MemoryKV();
    kv.store.set("yra", "{");
    stubWordAccentFetch({
      accentInfo: [
        {
          accented: ["yra\u0300"],
          information: [{ mi: "dktv.", meaning: "skylė" }],
        },
      ],
    });
    const { ctx, promises } = captureWaitUntil();

    const variants = await lookupWordVariantsKV("yra", envFor(kv), ctx);
    await Promise.all(promises);

    expect(variants).toEqual([
      { form: "yrà", info: "dktv. - skylė", mi: ["dktv."] },
    ]);
    const stored = JSON.parse(kv.store.get("yra") ?? "") as {
      variants: AccentVariant[];
    };
    expect(stored.variants).toEqual(variants);
  });
});
