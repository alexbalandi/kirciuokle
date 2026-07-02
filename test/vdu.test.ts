import { afterEach, describe, expect, it, vi } from "vitest";
import {
  accentText,
  clearNonceCache,
  extractNonce,
  flattenVariants,
  normalizeTextParts,
  splitTextIntoChunks,
} from "../src/worker/vdu";

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
