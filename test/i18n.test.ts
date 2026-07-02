import { afterEach, describe, expect, it, vi } from "vitest";
import { detectLang, translateMorphology } from "../src/client/i18n";

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
