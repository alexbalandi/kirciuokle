import type { AccentResponse, Part } from "../shared/types";
import {
  alignTokens,
  matchCase,
  pickReadingMi,
  pickVariant,
  tokenTags,
  toPublicVariants,
  type AccentVariant,
} from "./disambiguation";
import { tagText } from "./udpipe";

const NONCE_URL = "https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/";
const AJAX_URL = "https://kalbu.vdu.lt/ajax-call";
const NONCE_TTL_MS = 6 * 60 * 60 * 1000;
const DEFAULT_CHUNK_SIZE = 4500;
export const WORD_CACHE_SECONDS = 7 * 24 * 60 * 60;

type NonceCache = {
  value: string;
  expiresAt: number;
};

type VduAjaxEnvelope = {
  code?: number;
  message?: unknown;
};

export type VduTextPart = {
  string?: string;
  accented?: string;
  accentType?: string;
  type?: string;
};

type VduTextResponse = {
  textParts?: VduTextPart[];
};

type VduAccentInfo = {
  accented?: string[];
  information?: Array<{
    mi?: string;
    meaning?: string;
  }>;
};

type VduWordResponse = {
  accentInfo?: VduAccentInfo[];
};

export type WordAccentEntry = {
  variants: AccentVariant[];
  defaultForm: string | null;
  accentType: string | null;
  defaultFormTitle: string | null;
  accentTypeTitle: string | null;
};

let nonceCache: NonceCache | null = null;

export class UpstreamError extends Error {
  constructor(message = "VDU kirčiuoklė laikinai nepasiekiama.") {
    super(message);
    this.name = "UpstreamError";
  }
}

class RetryableVduError extends Error {
  constructor(message = "VDU API response was not usable.") {
    super(message);
    this.name = "RetryableVduError";
  }
}

export function clearNonceCache(): void {
  nonceCache = null;
}

export function extractNonce(html: string): string | null {
  return html.match(/"NONCE":"([0-9a-f]+)"/)?.[1] ?? null;
}

async function fetchNonce(): Promise<string> {
  const response = await fetch(NONCE_URL, {
    headers: { accept: "text/html,application/xhtml+xml" },
  });

  if (!response.ok) {
    throw new UpstreamError();
  }

  const nonce = extractNonce(await response.text());
  if (!nonce) {
    throw new UpstreamError();
  }

  nonceCache = {
    value: nonce,
    expiresAt: Date.now() + NONCE_TTL_MS,
  };

  return nonce;
}

async function getNonce(forceRefresh = false): Promise<string> {
  if (!forceRefresh && nonceCache && nonceCache.expiresAt > Date.now()) {
    return nonceCache.value;
  }

  return fetchNonce();
}

async function postVdu<T>(
  action: "text_accents" | "word_accent",
  fields: Record<string, string>,
): Promise<T> {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const nonce = await getNonce(attempt > 0);
      const body = new URLSearchParams({ action, nonce });

      for (const [key, value] of Object.entries(fields)) {
        body.set(key, value);
      }

      const response = await fetch(AJAX_URL, {
        method: "POST",
        headers: {
          "content-type": "application/x-www-form-urlencoded",
          accept: "application/json",
        },
        body,
      });

      if (!response.ok) {
        throw new RetryableVduError();
      }

      const envelope = (await response.json()) as VduAjaxEnvelope;
      if (envelope.code !== 200) {
        throw new RetryableVduError();
      }

      // word_accent answers `message: false` for words entirely outside the
      // dictionary (e.g. non-Lithuanian spellings) — a genuine negative,
      // not a transport error.
      if (envelope.message === false) {
        return {} as T;
      }

      if (typeof envelope.message !== "string") {
        throw new RetryableVduError();
      }

      return JSON.parse(envelope.message) as T;
    } catch (error) {
      if (attempt === 0) {
        nonceCache = null;
        continue;
      }

      if (error instanceof UpstreamError) {
        throw error;
      }

      throw new UpstreamError();
    }
  }

  throw new UpstreamError();
}

export function splitTextIntoChunks(
  text: string,
  maxLength = DEFAULT_CHUNK_SIZE,
): string[] {
  if (text.length <= maxLength) {
    return text ? [text] : [];
  }

  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const remaining = text.length - start;
    if (remaining <= maxLength) {
      chunks.push(text.slice(start));
      break;
    }

    const segment = text.slice(start, start + maxLength);
    let cut = findSentenceBoundary(segment);

    if (cut <= 0) {
      cut = segment.lastIndexOf(" ") + 1;
    }

    if (cut <= 0) {
      cut = maxLength;
    }

    chunks.push(text.slice(start, start + cut));
    start += cut;
  }

  return chunks;
}

function findSentenceBoundary(segment: string): number {
  for (let index = segment.length - 1; index >= 0; index -= 1) {
    if (segment[index] === "." || segment[index] === "!" || segment[index] === "?" || segment[index] === "\n") {
      return index + 1;
    }
  }

  return -1;
}

export function normalizeTextParts(textParts: VduTextPart[]): Part[] {
  return textParts.map((part) => {
    const original = (part.string ?? "").normalize("NFC");

    if (part.type === "SEPARATOR") {
      return {
        text: original,
        type: "sep",
      };
    }

    const normalized: Part = {
      text: original,
      type: "word",
    };

    if (part.accented) {
      normalized.accented = part.accented.normalize("NFC");
    }

    if (part.accentType === "MULTIPLE_MEANING") {
      normalized.ambiguous = true;
    }

    if (part.type === "NON_LT" || part.accentType === "NONE") {
      normalized.unknown = true;
      delete normalized.accented;
    }

    return normalized;
  });
}

export function flattenVariants(response: VduWordResponse): AccentVariant[] {
  return (response.accentInfo ?? []).flatMap((entry) => {
    const info = formatInformation(entry.information ?? []);
    const mi = (entry.information ?? [])
      .map((item) => item.mi)
      .filter((label): label is string => Boolean(label));

    return (entry.accented ?? []).map((form) => ({
      form: form.normalize("NFC"),
      info,
      mi,
    }));
  });
}

function formatInformation(
  information: NonNullable<VduAccentInfo["information"]>,
): string {
  return information
    .map((item) => [item.mi, item.meaning].filter(Boolean).join(" - "))
    .filter(Boolean)
    .join("; ");
}

export type AccentTextOptions = {
  lookupVariants?: (word: string) => Promise<AccentVariant[]>;
  useTagger?: boolean;
  /** Attach reading info to every covered word, not just ambiguous ones. */
  attachInfoForAll?: boolean;
  /** Cheap (no-network) lookup used for the non-ambiguous words when
      attachInfoForAll is set; defaults to lookupVariants. */
  lookupInfoVariants?: (word: string) => Promise<AccentVariant[]>;
};

type TaggerResult =
  | { tagger: "ok"; tokens: Awaited<ReturnType<typeof tagText>> }
  | { tagger: "unavailable"; tokens: [] };

export async function accentText(
  text: string,
  options: AccentTextOptions = {},
): Promise<AccentResponse> {
  return {
    ...(await accentTextParts(text, await fetchTextAccentParts(text), options)),
    source: "vdu",
  };
}

export async function accentTextParts(
  text: string,
  textParts: VduTextPart[],
  options: AccentTextOptions = {},
): Promise<Omit<AccentResponse, "source">> {
  const taggerPromise = getTaggerResult(text, options.useTagger !== false);
  const taggerResult = await taggerPromise;
  const parts = normalizeTextParts(textParts);
  const wordParts = textParts.filter(isWordPart);
  const aligned =
    taggerResult.tagger === "ok"
      ? alignTokens(textParts, taggerResult.tokens)
      : Array<TokenOrNull>(wordParts.length).fill(null);
  const variantsByWord = await fetchAmbiguousVariants(
    distinctAmbiguousWords(wordParts),
    options.lookupVariants ?? lookupWordVariants,
  );

  if (options.attachInfoForAll) {
    const rest = distinctWordKeys(wordParts).filter(
      (key) => !variantsByWord.has(key),
    );
    const infoLookup =
      options.lookupInfoVariants ?? options.lookupVariants ?? lookupWordVariants;
    const infoByWord = await fetchAmbiguousVariants(rest, infoLookup);
    for (const [key, value] of infoByWord) {
      variantsByWord.set(key, value);
    }
  }

  let wordIndex = 0;
  const disambiguatedParts = parts.map((part, index) => {
    const original = textParts[index];
    if (!original || !isWordPart(original)) {
      return part;
    }

    const token = aligned[wordIndex] ?? null;
    wordIndex += 1;
    const context = token ? tokenTags(token) : undefined;
    const contextTags = context && Object.keys(context).length > 0 ? context : undefined;

    if (original.accentType !== "MULTIPLE_MEANING") {
      if (part.unknown || !part.accented) {
        return part;
      }

      // Plain word: the token tags always travel along so lazily-fetched
      // readings can be scored client-side; with attachInfoForAll the
      // dictionary readings and the matched reading ship right away.
      const enriched: Part = { ...part };
      if (contextTags) {
        enriched.tokenTags = contextTags;
      }

      if (!options.attachInfoForAll) {
        return enriched;
      }

      const raw = variantsByWord.get(normalizeWordKey(original.string ?? "")) ?? [];
      if (raw.length === 0) {
        return enriched;
      }

      enriched.variants = toPublicVariants(raw);
      const readingMi = contextTags ? pickReadingMi(raw, contextTags) : undefined;
      if (readingMi) {
        enriched.chosenMi = readingMi;
      }

      return enriched;
    }

    const defaultForm = original.accented?.normalize("NFC") ?? part.accented;
    const variants = variantsByWord.get(normalizeWordKey(original.string ?? ""));
    const safeVariants = variants ?? [];
    const choice = pickVariant(part.text, safeVariants, token, defaultForm);
    const selectedVariant =
      choice.index === null ? undefined : safeVariants[choice.index];
    const accented = (
      selectedVariant?.form ??
      defaultForm ??
      part.accented ??
      part.text
    ).normalize("NFC");
    const publicVariants = toPublicVariants(safeVariants);
    const distinctForms = new Set(
      publicVariants.map((variant) => variant.form.normalize("NFC")),
    );

    const readingMi =
      contextTags && safeVariants.length > 0
        ? pickReadingMi(
            selectedVariant ? [selectedVariant] : safeVariants,
            contextTags,
          )
        : undefined;

    if (distinctForms.size <= 1) {
      // All readings share one accented form (e.g. põ as four prepositional
      // readings) — there is nothing to choose, so don't flag it.
      const resolved: Part = {
        ...part,
        accented: matchCase(accented, part.text).normalize("NFC"),
        variants: publicVariants,
      };
      delete resolved.ambiguous;
      if (readingMi) {
        resolved.chosenMi = readingMi;
      }
      if (contextTags) {
        resolved.tokenTags = contextTags;
      }
      return resolved;
    }

    const chosenPart: Part = {
      ...part,
      ambiguous: true,
      accented: matchCase(accented, part.text).normalize("NFC"),
      variants: publicVariants,
    };

    if (contextTags) {
      chosenPart.tokenTags = contextTags;
    }

    if (choice.index !== null) {
      chosenPart.chosen = choice.index;
    }

    if (choice.resolvedBy) {
      chosenPart.resolvedBy = choice.resolvedBy;
    }

    if (readingMi) {
      chosenPart.chosenMi = readingMi;
    }

    return chosenPart;
  });

  return { parts: disambiguatedParts, tagger: taggerResult.tagger };
}

type TokenOrNull = Awaited<ReturnType<typeof tagText>>[number] | null;

async function getTaggerResult(
  text: string,
  useTagger: boolean,
): Promise<TaggerResult> {
  if (!useTagger) {
    return { tagger: "unavailable", tokens: [] };
  }

  try {
    return { tagger: "ok", tokens: await tagText(text) };
  } catch {
    return { tagger: "unavailable", tokens: [] };
  }
}

async function fetchTextAccentParts(text: string): Promise<VduTextPart[]> {
  const textParts: VduTextPart[] = [];

  for (const chunk of splitTextIntoChunks(text)) {
    const response = await postVdu<VduTextResponse>("text_accents", {
      body: chunk,
    });

    textParts.push(...(response.textParts ?? []));
  }

  return textParts;
}

function isWordPart(part: VduTextPart): boolean {
  return part.type === "WORD" || part.type === "NON_LT";
}

function distinctWordKeys(wordParts: VduTextPart[]): string[] {
  const words = new Set<string>();

  for (const part of wordParts) {
    if (part.string) {
      words.add(normalizeWordKey(part.string));
    }
  }

  return [...words].sort();
}

function distinctAmbiguousWords(wordParts: VduTextPart[]): string[] {
  const words = new Set<string>();

  for (const part of wordParts) {
    if (part.accentType !== "MULTIPLE_MEANING" || !part.string) {
      continue;
    }

    words.add(normalizeWordKey(part.string));
  }

  return [...words].sort();
}

function normalizeWordKey(word: string): string {
  return word.normalize("NFC").toLowerCase();
}

async function fetchAmbiguousVariants(
  words: string[],
  lookupVariants: (word: string) => Promise<AccentVariant[]>,
): Promise<Map<string, AccentVariant[]>> {
  return lookupVariantsConcurrently(words, lookupVariants, { swallowErrors: true });
}

export async function lookupVariantsConcurrently(
  words: string[],
  lookupVariants: (word: string) => Promise<AccentVariant[]> = lookupWordVariants,
  options: { swallowErrors?: boolean } = {},
): Promise<Map<string, AccentVariant[]>> {
  return lookupConcurrently(words, lookupVariants, {
    ...options,
    fallback: () => [],
  });
}

export async function lookupWordEntriesConcurrently(
  words: string[],
  lookupEntry: (word: string) => Promise<WordAccentEntry> = fetchWordEntry,
  options: { swallowErrors?: boolean } = {},
): Promise<Map<string, WordAccentEntry>> {
  return lookupConcurrently(words, lookupEntry, {
    ...options,
    fallback: () => ({
      variants: [],
      defaultForm: null,
      accentType: "NONE",
      defaultFormTitle: null,
      accentTypeTitle: "NONE",
    }),
  });
}

async function lookupConcurrently<T>(
  words: string[],
  lookup: (word: string) => Promise<T>,
  options: { swallowErrors?: boolean; fallback: () => T },
): Promise<Map<string, T>> {
  const resultsByWord = new Map<string, T>();
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < words.length) {
      const word = words[nextIndex]!;
      nextIndex += 1;

      try {
        resultsByWord.set(word, await lookup(word));
      } catch {
        if (!options.swallowErrors) {
          throw new UpstreamError();
        }

        resultsByWord.set(word, options.fallback());
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(6, words.length) }, () => worker()),
  );

  return resultsByWord;
}

export async function lookupWordVariants(word: string): Promise<AccentVariant[]> {
  const response = await postVdu<VduWordResponse>("word_accent", { word });
  return flattenVariants(response);
}

export async function fetchWordEntry(word: string): Promise<WordAccentEntry> {
  const variants = await lookupWordVariants(word);
  const lower = await fetchCanonicalWordSide(word);
  const title = await fetchCanonicalWordSide(toTitleCase(word));

  return {
    variants,
    defaultForm: lower.form,
    accentType: lower.type,
    defaultFormTitle: title.form,
    accentTypeTitle: title.type,
  };
}

async function fetchCanonicalWordSide(
  word: string,
): Promise<{ form: string | null; type: string }> {
  const textParts = await fetchTextAccentParts(word);
  const wordPart = textParts.find((part) => part.type === "WORD");
  const form = wordPart?.accented?.normalize("NFC") ?? null;

  if (!wordPart || !form) {
    return { form: null, type: "NONE" };
  }

  return {
    form,
    type: wordPart.accentType ?? "ONE",
  };
}

function toTitleCase(word: string): string {
  const letters = Array.from(word.normalize("NFC"));
  const first = letters[0];

  if (!first) {
    return word;
  }

  return `${first.toUpperCase()}${letters.slice(1).join("")}`;
}
