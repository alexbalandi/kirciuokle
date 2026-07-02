import { afterEach, describe, expect, it, vi } from "vitest";
import {
  detectLang,
  morphologySegments,
  translateMorphology,
} from "../src/client/i18n";

declare global {
  var localStorage: {
    getItem(key: string): string | null;
  };
  var navigator: {
    language?: string;
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("translateMorphology", () => {
  it("translates two readings into English", () => {
    expect(
      translateMorphology(
        "bdv., vyr. g., vns. šauksm.; bdv., vyr. g., vns. vard.",
        "en",
      ),
    ).toBe(
      "adjective, masculine, singular vocative; adjective, masculine, singular nominative",
    );
  });

  it("translates two readings into Russian", () => {
    expect(
      translateMorphology(
        "bdv., vyr. g., vns. šauksm.; bdv., vyr. g., vns. vard.",
        "ru",
      ),
    ).toBe(
      "прилагательное, мужской род, ед. число звательный; прилагательное, мужской род, ед. число именительный",
    );
  });

  it("keeps unknown fragments verbatim", () => {
    expect(translateMorphology("bdv., nežinoma, vns. vard.", "en")).toBe(
      "adjective, nežinoma, singular nominative",
    );
  });

  it("keeps dictionary meanings after mi verbatim", () => {
    expect(translateMorphology("vksm., 3 asm. - būti", "en")).toBe(
      "verb, 3rd person - būti",
    );
  });

  it("matches the longest abbreviation first", () => {
    expect(translateMorphology("vksm., būt. k. l., 3 asm.", "en")).toBe(
      "verb, simple past, 3rd person",
    );
  });
});

describe("morphologySegments", () => {
  const twoReadings = "bdv., vyr. g., vns. šauksm.; bdv., vyr. g., vns. vard.";
  const twoReadingsLt = [
    "būdvardis",
    "vyriškoji giminė",
    "vienaskaita",
    "šauksmininkas",
    "būdvardis",
    "vyriškoji giminė",
    "vienaskaita",
    "vardininkas",
  ];

  it("segments two readings for English with full Lithuanian terms", () => {
    const segments = morphologySegments(twoReadings, "en");

    expect(segments.map((segment) => segment.text).join("")).toBe(
      translateMorphology(twoReadings, "en"),
    );
    expect(segments.flatMap((segment) => (segment.lt ? [segment.lt] : []))).toEqual(
      twoReadingsLt,
    );
  });

  it("segments two readings for Russian with full Lithuanian terms", () => {
    const segments = morphologySegments(twoReadings, "ru");

    expect(segments.map((segment) => segment.text).join("")).toBe(
      translateMorphology(twoReadings, "ru"),
    );
    expect(segments.flatMap((segment) => (segment.lt ? [segment.lt] : []))).toEqual(
      twoReadingsLt,
    );
  });

  it("keeps unknown fragments plain", () => {
    const segments = morphologySegments("bdv., nežinoma, vns. vard.", "en");
    const unknown = segments.find((segment) => segment.text === "nežinoma");

    expect(segments.map((segment) => segment.text).join("")).toBe(
      translateMorphology("bdv., nežinoma, vns. vard.", "en"),
    );
    expect(unknown).toEqual({ text: "nežinoma" });
  });

  it("keeps dictionary meanings plain", () => {
    const segments = morphologySegments("vksm., 3 asm. - būti", "en");
    const meaning = segments.find((segment) => segment.text.includes("būti"));

    expect(segments.map((segment) => segment.text).join("")).toBe(
      translateMorphology("vksm., 3 asm. - būti", "en"),
    );
    expect(meaning).toEqual({ text: " - būti" });
  });

  it("does not annotate Lithuanian segments", () => {
    const segments = morphologySegments(twoReadings, "lt");

    expect(segments.map((segment) => segment.text).join("")).toBe(
      translateMorphology(twoReadings, "lt"),
    );
    expect(segments.some((segment) => "lt" in segment)).toBe(false);
  });
});

describe("detectLang", () => {
  it("prefers localStorage over navigator.language", () => {
    vi.stubGlobal("localStorage", { getItem: vi.fn(() => "ru") });
    vi.stubGlobal("navigator", { language: "lt-LT" });

    expect(detectLang()).toBe("ru");
  });

  it("falls back to navigator.language when localStorage has no supported lang", () => {
    vi.stubGlobal("localStorage", { getItem: vi.fn(() => "fr") });
    vi.stubGlobal("navigator", { language: "lt-LT" });

    expect(detectLang()).toBe("lt");
  });

  it("defaults to English for unsupported navigator languages", () => {
    vi.stubGlobal("localStorage", { getItem: vi.fn(() => null) });
    vi.stubGlobal("navigator", { language: "pl-PL" });

    expect(detectLang()).toBe("en");
  });
});
