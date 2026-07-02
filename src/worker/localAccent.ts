import type { AccentResponse } from "../shared/types";
import {
  getWords,
  lookupWordVariantsD1,
  normalizeWordKey,
  putWords,
  type WordDictionaryEntry,
  type WordDictionaryEnv,
} from "./dictionary";
import {
  accentText,
  accentTextParts,
  lookupWordEntriesConcurrently,
  type AccentTextOptions,
  type VduTextPart,
} from "./vdu";
import { matchCase } from "./disambiguation";

// Combining marks stay inside the token so pre-accented input (e.g. a
// pasted "mė́nuo") is treated as one NON_LT word and left untouched instead
// of being split at the mark and re-accented around it.
const WORD_RE = /[\p{L}\p{M}]+/gu;
const LT_WORD_RE = /^[A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž]+$/u;
const ROMAN_NUMERAL_RE = /^[IVXLCDM]+$/u;
export const MISS_BUDGET = 10;
export const ABBREVIATIONS = new Set([
  "a",
  "akad",
  "adr",
  "angl",
  "aps",
  "apskr",
  "dab",
  "dir",
  "doc",
  "dr",
  "egz",
  "est",
  "etc",
  "gen",
  "gr",
  "gyv",
  "habil",
  "insp",
  "isp",
  "it",
  "jaun",
  "kan",
  "kpt",
  "kt",
  "latv",
  "lenk",
  "liet",
  "lot",
  "m",
  "min",
  "mjr",
  "mln",
  "mlrd",
  "mstl",
  "nr",
  "pan",
  "pav",
  "pgl",
  "pirm",
  "plg",
  "plk",
  "pers",
  "pr",
  "pranc",
  "proc",
  "prof",
  "psn",
  "pvz",
  "rus",
  "sav",
  "sek",
  "sen",
  "sk",
  "str",
  "šnek",
  "šv",
  "tarm",
  "tel",
  "tūkst",
  "ukr",
  "val",
  "vok",
  "vyr",
  "žr",
]);

type LocalAccentOptions = Pick<AccentTextOptions, "useTagger">;

type LocalWord = {
  partIndex: number;
  text: string;
  key: string;
};

type TokenizedText = {
  textParts: VduTextPart[];
  lookupWords: LocalWord[];
};

export async function accentTextLocalFirst(
  text: string,
  env: WordDictionaryEnv,
  ctx: Pick<ExecutionContext, "waitUntil">,
  options: LocalAccentOptions = {},
): Promise<AccentResponse> {
  const normalizedText = text.normalize("NFC");
  const tokenized = tokenizeLikeVdu(normalizedText);
  const distinctWords = [...new Set(tokenized.lookupWords.map((word) => word.key))];
  const entriesByWord = await getWords(env, distinctWords);
  const misses = distinctWords.filter((word) => entriesByWord.get(word) === null);

  if (misses.length > MISS_BUDGET) {
    scheduleSeedMisses(misses.slice(0, MISS_BUDGET), env, ctx);
    return accentText(normalizedText, {
      lookupVariants: (word) => lookupWordVariantsD1(word, env, ctx),
      useTagger: options.useTagger,
    });
  }

  if (misses.length > 0) {
    const fetched = await lookupWordEntriesConcurrently(misses);
    for (const [word, entry] of fetched) {
      entriesByWord.set(word, entry);
    }

    ctx.waitUntil(
      putWords(
        env,
        [...fetched].map(([word, entry]) => ({ word, ...entry })),
      ),
    );
  }

  applyDictionaryResults(tokenized, entriesByWord);
  const parts = await accentTextParts(normalizedText, tokenized.textParts, {
    lookupVariants: async (word) =>
      entriesByWord.get(normalizeWordKey(word))?.variants ?? [],
    useTagger: options.useTagger,
  });

  return { ...parts, source: "local" };
}

export function tokenizeLikeVdu(text: string): TokenizedText {
  const textParts: VduTextPart[] = [];
  const lookupWords: LocalWord[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(WORD_RE)) {
    const word = match[0].normalize("NFC");
    const index = match.index ?? 0;

    if (index > lastIndex) {
      textParts.push({
        string: text.slice(lastIndex, index).normalize("NFC"),
        type: "SEPARATOR",
      });
    }

    const partIndex = textParts.length;
    const wordEnd = index + match[0].length;
    if (isAbbreviation(word, text, wordEnd)) {
      textParts.push({
        string: text.slice(index, wordEnd + 1).normalize("NFC"),
        type: "WORD",
        accentType: "NONE",
      });
      lastIndex = wordEnd + 1;
      continue;
    }

    if (ROMAN_NUMERAL_RE.test(word)) {
      textParts.push({ string: word, type: "WITH_NUMBER" });
    } else if (LT_WORD_RE.test(word)) {
      const key = normalizeWordKey(word);
      textParts.push({ string: word, type: "WORD" });
      lookupWords.push({ partIndex, text: word, key });
    } else {
      textParts.push({ string: word, type: "NON_LT" });
    }

    lastIndex = wordEnd;
  }

  if (lastIndex < text.length) {
    textParts.push({
      string: text.slice(lastIndex).normalize("NFC"),
      type: "SEPARATOR",
    });
  }

  return { textParts, lookupWords };
}

function applyDictionaryResults(
  tokenized: TokenizedText,
  entriesByWord: Map<string, WordDictionaryEntry | null>,
): void {
  for (const word of tokenized.lookupWords) {
    const part = tokenized.textParts[word.partIndex];
    if (!part) {
      continue;
    }

    const entry = entriesByWord.get(word.key);
    const side = entry ? selectCanonicalSide(entry, word.text) : null;

    if (!side || side.type === "NONE" || !side.form) {
      part.accentType = "NONE";
      delete part.accented;
      continue;
    }

    part.accented = matchCase(side.form, word.text).normalize("NFC");
    part.accentType = side.type ?? "ONE";
  }
}

function isAbbreviation(word: string, text: string, wordEnd: number): boolean {
  if (text[wordEnd] !== ".") {
    return false;
  }

  return Array.from(word).length === 1 || ABBREVIATIONS.has(normalizeWordKey(word));
}

function selectCanonicalSide(
  entry: WordDictionaryEntry,
  word: string,
): { form: string | null; type: string | null } {
  if (startsWithUppercase(word)) {
    return {
      form: entry.defaultFormTitle,
      type: entry.accentTypeTitle,
    };
  }

  return {
    form: entry.defaultForm,
    type: entry.accentType,
  };
}

function startsWithUppercase(word: string): boolean {
  const first = Array.from(word)[0];
  return Boolean(first && first.toUpperCase() === first && first.toLowerCase() !== first);
}

function scheduleSeedMisses(
  words: string[],
  env: WordDictionaryEnv,
  ctx: Pick<ExecutionContext, "waitUntil">,
): void {
  if (words.length === 0) {
    return;
  }

  ctx.waitUntil(
    lookupWordEntriesConcurrently(words)
      .then((entriesByWord) =>
        putWords(
          env,
          [...entriesByWord].map(([word, entry]) => ({ word, ...entry })),
        ),
      )
      .catch((error: unknown) => {
        console.error("Failed to seed local accent dictionary", error);
      }),
  );
}
