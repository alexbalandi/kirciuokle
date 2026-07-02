import { vi } from "vitest";
import type { AccentVariant } from "../src/worker/disambiguation";
import type { WordDictionaryEntry } from "../src/worker/dictionary";

type StoredWord = {
  word: string;
  variants: string;
  fetched_at: string;
  negative_until: string | null;
  default_form: string | null;
  accent_type: string | null;
  default_form_title: string | null;
  accent_type_title: string | null;
};

type TestVduWordResponse = {
  accentInfo?: Array<{
    accented?: string[];
    information?: Array<{
      mi?: string;
      meaning?: string;
    }>;
  }>;
};

type TestVduTextPart = {
  string?: string;
  accented?: string;
  accentType?: string;
  type?: string;
};

export class MemoryD1 {
  readonly store = new Map<string, StoredWord>();
  readonly selectBinds: unknown[][] = [];
  readonly insertBinds: unknown[][] = [];

  prepare(sql: string): D1PreparedStatement {
    return {
      bind: (...values: unknown[]) => ({
        all: async <T>() => {
          if (!/^\s*SELECT\b/i.test(sql)) {
            throw new Error(`Unexpected all() SQL: ${sql}`);
          }

          this.selectBinds.push(values);
          const results = values
            .map((value) => this.store.get(String(value)))
            .filter((row): row is StoredWord => Boolean(row))
            .map((row) => ({ ...row })) as T[];

          return { results, success: true } as D1Result<T>;
        },
        run: async () => {
          if (!/^\s*INSERT\b/i.test(sql)) {
            throw new Error(`Unexpected run() SQL: ${sql}`);
          }

          this.insertBinds.push(values);
          for (let index = 0; index < values.length; index += 8) {
            const word = String(values[index]);
            this.store.set(word, {
              word,
              variants: String(values[index + 1]),
              fetched_at: String(values[index + 2]),
              negative_until:
                values[index + 3] === null ? null : String(values[index + 3]),
              default_form:
                values[index + 4] === null ? null : String(values[index + 4]),
              accent_type:
                values[index + 5] === null ? null : String(values[index + 5]),
              default_form_title:
                values[index + 6] === null ? null : String(values[index + 6]),
              accent_type_title:
                values[index + 7] === null ? null : String(values[index + 7]),
            } as StoredWord);
          }

          return { success: true } as D1Result;
        },
      }),
    } as unknown as D1PreparedStatement;
  }

  setWord(
    word: string,
    variants: AccentVariant[],
    options: {
      fetchedAt?: string;
      negativeUntil?: string | null;
      defaultForm?: string | null;
      accentType?: string | null;
      defaultFormTitle?: string | null;
      accentTypeTitle?: string | null;
    } = {},
  ): void {
    const defaultForm =
      "defaultForm" in options ? options.defaultForm ?? null : variants[0]?.form ?? null;
    const accentType =
      "accentType" in options
        ? options.accentType ?? null
        : variants.length > 0
          ? "ONE"
          : null;
    const defaultFormTitle =
      "defaultFormTitle" in options
        ? options.defaultFormTitle ?? null
        : titleCaseForm(defaultForm);
    const accentTypeTitle =
      "accentTypeTitle" in options ? options.accentTypeTitle ?? null : accentType;

    this.store.set(word, {
      word,
      variants: JSON.stringify(variants),
      fetched_at: options.fetchedAt ?? "2026-07-02T00:00:00.000Z",
      negative_until: options.negativeUntil ?? null,
      default_form: defaultForm,
      accent_type: accentType,
      default_form_title: defaultFormTitle,
      accent_type_title: accentTypeTitle,
    });
  }

  getVariants(word: string): AccentVariant[] {
    const row = this.store.get(word);
    return row ? (JSON.parse(row.variants) as AccentVariant[]) : [];
  }

  getNegativeUntil(word: string): string | null {
    return this.store.get(word)?.negative_until ?? null;
  }

  getEntry(word: string): WordDictionaryEntry | null {
    const row = this.store.get(word);
    if (!row) {
      return null;
    }

    return {
      variants: JSON.parse(row.variants) as AccentVariant[],
      defaultForm: row.default_form,
      accentType: row.accent_type,
      defaultFormTitle: row.default_form_title,
      accentTypeTitle: row.accent_type_title,
    };
  }
}

export function envFor(d1: MemoryD1, accentSource?: "local" | "vdu") {
  return {
    ASSETS: { fetch: vi.fn() } as unknown as Fetcher,
    DICT: d1 as unknown as D1Database,
    ACCENT_SOURCE: accentSource,
  };
}

export function captureWaitUntil(): {
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

export function stubVduFetch(options: {
  wordResponses?: Record<string, TestVduWordResponse>;
  textParts?: TestVduTextPart[];
  textResponses?: Record<string, TestVduTextPart[]>;
  conllu?: string;
}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const body = String(init?.body ?? "");

    if (url.includes("lindat.mff.cuni.cz")) {
      if (options.conllu === undefined) {
        throw new Error("tagger down");
      }

      return Response.json({ result: options.conllu });
    }

    if (url.includes("kirciuoklis")) {
      return new Response('<script>{"NONCE":"abcdef123456"}</script>');
    }

    const params = new URLSearchParams(body);
    const action = params.get("action");

    if (action === "word_accent") {
      return Response.json({
        code: 200,
        message: JSON.stringify(
          options.wordResponses?.[params.get("word") ?? ""] ?? {},
        ),
      });
    }

    if (action === "text_accents") {
      const bodyText = params.get("body") ?? "";
      return Response.json({
        code: 200,
        message: JSON.stringify({
          textParts: options.textResponses?.[bodyText] ?? options.textParts ?? [],
        }),
      });
    }

    throw new Error(`Unexpected fetch: ${url} ${body}`);
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function titleCaseForm(form: string | null): string | null {
  if (!form) {
    return null;
  }

  const letters = Array.from(form.normalize("NFC"));
  const first = letters[0];

  if (!first) {
    return form;
  }

  return `${first.toUpperCase()}${letters.slice(1).join("")}`;
}
