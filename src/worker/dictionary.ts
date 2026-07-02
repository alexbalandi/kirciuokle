import type { AccentVariant } from "./disambiguation";
import { fetchWordEntry } from "./vdu";

const MAX_D1_BOUND_PARAMETERS = 90;
const PUT_PARAMETERS_PER_ENTRY = 8;
export const NEGATIVE_WORD_TTL_MS = 30 * 24 * 60 * 60 * 1000;

export type WordDictionaryEnv = {
  DICT: D1Database;
};

export type WordDictionaryEntry = {
  variants: AccentVariant[];
  defaultForm: string | null;
  accentType: string | null;
  defaultFormTitle: string | null;
  accentTypeTitle: string | null;
};

export type WordDictionaryPutEntry = WordDictionaryEntry & {
  word: string;
};

type WordRow = {
  word: string;
  variants: string;
  negative_until: string | null;
  default_form: string | null;
  accent_type: string | null;
  default_form_title: string | null;
  accent_type_title: string | null;
};

export async function getWords(
  env: WordDictionaryEnv,
  words: string[],
): Promise<Map<string, WordDictionaryEntry | null>> {
  const keys = distinctNormalizedWords(words);
  const result = new Map<string, WordDictionaryEntry | null>(
    keys.map((word) => [word, null]),
  );
  const now = Date.now();

  for (const chunk of chunks(keys, MAX_D1_BOUND_PARAMETERS)) {
    if (chunk.length === 0) {
      continue;
    }

    const placeholders = chunk.map(() => "?").join(", ");
    const rows = await env.DICT
      .prepare(
        `SELECT word, variants, negative_until, default_form, accent_type, default_form_title, accent_type_title FROM words WHERE word IN (${placeholders})`,
      )
      .bind(...chunk)
      .all<WordRow>();

    for (const row of rows.results ?? []) {
      const word = normalizeWordKey(row.word);
      const variants = parseVariants(row.variants);
      if (variants === null) {
        result.set(word, null);
        continue;
      }

      if (row.accent_type_title === null) {
        result.set(word, null);
        continue;
      }

      if (isExpiredNegative(row, now)) {
        result.set(word, null);
        continue;
      }

      result.set(word, {
        variants,
        defaultForm: row.default_form?.normalize("NFC") ?? null,
        accentType: row.accent_type,
        defaultFormTitle: row.default_form_title?.normalize("NFC") ?? null,
        accentTypeTitle: row.accent_type_title,
      });
    }
  }

  return result;
}

export async function putWords(
  env: WordDictionaryEnv,
  entries: WordDictionaryPutEntry[],
): Promise<void> {
  const now = new Date();
  const fetchedAt = now.toISOString();
  const negativeUntil = new Date(now.getTime() + NEGATIVE_WORD_TTL_MS).toISOString();
  const normalizedEntries = normalizePutEntries(entries);

  for (const chunk of chunks(
    normalizedEntries,
    Math.floor(MAX_D1_BOUND_PARAMETERS / PUT_PARAMETERS_PER_ENTRY),
  )) {
    if (chunk.length === 0) {
      continue;
    }

    const valuesSql = chunk.map(() => "(?, ?, ?, ?, ?, ?, ?, ?)").join(", ");
    const values = chunk.flatMap((entry) => [
      entry.word,
      JSON.stringify(entry.variants),
      fetchedAt,
      isNegativeEntry(entry) ? negativeUntil : null,
      entry.defaultForm,
      entry.accentType,
      entry.defaultFormTitle,
      entry.accentTypeTitle,
    ]);

    await env.DICT
      .prepare(
        `INSERT OR REPLACE INTO words (word, variants, fetched_at, negative_until, default_form, accent_type, default_form_title, accent_type_title) VALUES ${valuesSql}`,
      )
      .bind(...values)
      .run();
  }
}

export async function lookupWordVariantsD1(
  word: string,
  env: WordDictionaryEnv,
  ctx?: Pick<ExecutionContext, "waitUntil">,
): Promise<AccentVariant[]> {
  const key = normalizeWordKey(word);
  const cached = (await getWords(env, [key])).get(key);
  if (cached !== null && cached !== undefined) {
    return cached.variants;
  }

  const entry = await fetchWordEntry(key);
  const put = putWords(env, [{ word: key, ...entry }]);

  if (ctx) {
    ctx.waitUntil(put);
  } else {
    await put;
  }

  return entry.variants;
}

export function normalizeWordKey(word: string): string {
  return word.normalize("NFC").toLowerCase();
}

function normalizePutEntries(
  entries: WordDictionaryPutEntry[],
): WordDictionaryPutEntry[] {
  const byWord = new Map<string, WordDictionaryPutEntry>();

  for (const entry of entries) {
    const word = normalizeWordKey(entry.word);
    byWord.set(word, {
      word,
      variants: entry.variants.map((variant) => ({
        form: variant.form.normalize("NFC"),
        info: variant.info,
        mi: [...variant.mi],
      })),
      defaultForm: entry.defaultForm?.normalize("NFC") ?? null,
      accentType: entry.accentType,
      defaultFormTitle: entry.defaultFormTitle?.normalize("NFC") ?? null,
      accentTypeTitle: entry.accentTypeTitle,
    });
  }

  return [...byWord.values()];
}

function distinctNormalizedWords(words: string[]): string[] {
  return [...new Set(words.map(normalizeWordKey))];
}

function chunks<T>(items: T[], size: number): T[][] {
  const result: T[][] = [];

  for (let index = 0; index < items.length; index += size) {
    result.push(items.slice(index, index + size));
  }

  return result;
}

function isNegativeEntry(entry: WordDictionaryPutEntry): boolean {
  return (
    entry.variants.length === 0 &&
    entry.defaultForm === null &&
    entry.defaultFormTitle === null
  );
}

function isExpiredNegative(row: WordRow, now: number): boolean {
  return row.negative_until !== null && Date.parse(row.negative_until) < now;
}

function parseVariants(raw: string): AccentVariant[] | null {
  try {
    const value = JSON.parse(raw) as unknown;
    if (!Array.isArray(value) || !value.every(isAccentVariant)) {
      return null;
    }

    return value.map((variant) => ({
      form: variant.form.normalize("NFC"),
      info: variant.info,
      mi: [...variant.mi],
    }));
  } catch {
    return null;
  }
}

function isAccentVariant(value: unknown): value is AccentVariant {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as Partial<AccentVariant>;
  return (
    typeof candidate.form === "string" &&
    typeof candidate.info === "string" &&
    Array.isArray(candidate.mi) &&
    candidate.mi.every((label) => typeof label === "string")
  );
}
