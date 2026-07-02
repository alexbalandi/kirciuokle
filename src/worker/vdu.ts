import type { AccentResponse, Part } from "../shared/types";
import {
  alignTokens,
  matchCase,
  pickVariant,
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

type VduTextPart = {
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
      if (envelope.code !== 200 || typeof envelope.message !== "string") {
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

type AccentTextOptions = {
  lookupVariants?: (word: string) => Promise<AccentVariant[]>;
  useTagger?: boolean;
};

type TaggerResult =
  | { tagger: "ok"; tokens: Awaited<ReturnType<typeof tagText>> }
  | { tagger: "unavailable"; tokens: [] };

export async function accentText(
  text: string,
  options: AccentTextOptions = {},
): Promise<AccentResponse> {
  const taggerPromise = getTaggerResult(text, options.useTagger !== false);
  const textPartsPromise = fetchTextAccentParts(text);
  const [textParts, taggerResult] = await Promise.all([
    textPartsPromise,
    taggerPromise,
  ]);
  const parts = normalizeTextParts(textParts);
  const wordParts = textParts.filter(isWordPart);
  const aligned =
    taggerResult.tagger === "ok"
      ? alignTokens(textParts, taggerResult.tokens)
      : Array<TokenOrNull>(wordParts.length).fill(null);
  const ambiguousWords = distinctAmbiguousWords(wordParts);
  const variantsByWord = await fetchAmbiguousVariants(
    ambiguousWords,
    options.lookupVariants ?? lookupWordVariants,
  );

  let wordIndex = 0;
  const disambiguatedParts = parts.map((part, index) => {
    const original = textParts[index];
    if (!original || !isWordPart(original)) {
      return part;
    }

    const token = aligned[wordIndex] ?? null;
    wordIndex += 1;

    if (original.accentType !== "MULTIPLE_MEANING") {
      return part;
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
    const chosenPart: Part = {
      ...part,
      ambiguous: true,
      accented: matchCase(accented, part.text).normalize("NFC"),
      variants: toPublicVariants(safeVariants),
    };

    if (choice.index !== null) {
      chosenPart.chosen = choice.index;
    }

    if (choice.resolvedBy) {
      chosenPart.resolvedBy = choice.resolvedBy;
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
  const variantsByWord = new Map<string, AccentVariant[]>();
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < words.length) {
      const word = words[nextIndex]!;
      nextIndex += 1;

      try {
        variantsByWord.set(word, await lookupVariants(word));
      } catch {
        variantsByWord.set(word, []);
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(6, words.length) }, () => worker()),
  );

  return variantsByWord;
}

export async function lookupWordVariants(word: string): Promise<AccentVariant[]> {
  const response = await postVdu<VduWordResponse>("word_accent", { word });
  return flattenVariants(response);
}

export async function lookupWordVariantsCached(
  word: string,
  ctx?: Pick<ExecutionContext, "waitUntil">,
): Promise<AccentVariant[]> {
  const cache = getDefaultCache();
  const cacheKey = new Request(
    `https://kirciuokle.local/cache/word?w=${encodeURIComponent(normalizeWordKey(word))}`,
  );

  if (cache) {
    const cached = await readCachedVariants(cache, cacheKey);
    if (cached) {
      return cached;
    }
  }

  const variants = await lookupWordVariants(normalizeWordKey(word));

  if (cache) {
    const response = Response.json(
      { variants },
      {
        headers: {
          "cache-control": `public, max-age=${WORD_CACHE_SECONDS}`,
        },
      },
    );
    const put = cache.put(cacheKey, response);

    if (ctx) {
      ctx.waitUntil(put.catch(() => undefined));
    } else {
      await put.catch(() => undefined);
    }
  }

  return variants;
}

function getDefaultCache(): Cache | null {
  return typeof caches === "undefined" ? null : caches.default;
}

async function readCachedVariants(
  cache: Cache,
  cacheKey: Request,
): Promise<AccentVariant[] | null> {
  try {
    const response = await cache.match(cacheKey);
    if (!response) {
      return null;
    }

    const payload = (await response.json()) as { variants?: unknown };
    if (!Array.isArray(payload.variants)) {
      return null;
    }

    return payload.variants.flatMap((variant) => {
      if (!isCachedVariant(variant)) {
        return [];
      }

      return {
        form: variant.form.normalize("NFC"),
        info: variant.info,
        mi: variant.mi,
      };
    });
  } catch {
    return null;
  }
}

function isCachedVariant(value: unknown): value is AccentVariant {
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
