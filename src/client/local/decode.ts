const STRESS_MARKS = new Set(["\u0300", "\u0301", "\u0303"]);
const COMBINING_DOT_ABOVE = "\u0307";
const I_DOT_BASES = new Set(["i", "I", "j", "J"]);
const GRAVE = "\u0300";
const ACUTE = "\u0301";
const TILDE = "\u0303";
const VOWELS = new Set(Array.from("aeiouyąęėįūų"));
const LONG_VOWELS = new Set(Array.from("yąęėįūų"));
const SONORANTS = new Set(Array.from("lmnr"));
const PURE_DIPHTHONGS = new Set(["au", "ai", "ei", "ui", "uo", "ie"]);
const VOWEL_BASES = new Set(Array.from("aeiouy"));

export const DEFAULT_STRESS_MARKS = [GRAVE, ACUTE, TILDE] as const;

export type StressDecode =
  | { noStress: true }
  | { pos: number; mark: string; noStress: false };

export function decodeStress(
  logits: ArrayLike<number>,
  noStressLogit: number,
  offset: number,
  word: string,
  marks: readonly string[],
  maxChars: number,
): StressDecode {
  const key = wordKey(word);
  const chars = Array.from(key);
  let bestLogit = noStressLogit;
  let best: StressDecode | null = null;

  for (let pos = 0; pos < Math.min(chars.length, maxChars); pos += 1) {
    for (let markIndex = 0; markIndex < marks.length; markIndex += 1) {
      const mark = marks[markIndex]!;
      if (!validTarget(chars, pos, mark)) {
        continue;
      }
      const value = logits[offset + pos * marks.length + markIndex] ?? -Infinity;
      if (value > bestLogit) {
        bestLogit = value;
        best = { pos, mark, noStress: false };
      }
    }
  }

  return best ?? { noStress: true };
}

export function isValidStressTarget(word: string, pos: number, mark: string): boolean {
  return validTarget(Array.from(wordKey(word)), pos, mark);
}

function validTarget(chars: readonly string[], pos: number, mark: string): boolean {
  if (!(pos >= 0 && pos < chars.length)) {
    return false;
  }

  const ch = chars[pos]!;
  const prev = pos > 0 ? chars[pos - 1]! : "";
  const next = pos + 1 < chars.length ? chars[pos + 1]! : "";

  if (SONORANTS.has(ch)) {
    return mark === TILDE && VOWELS.has(prev);
  }

  if (!VOWELS.has(ch)) {
    return false;
  }

  if (LONG_VOWELS.has(ch)) {
    return mark !== GRAVE;
  }

  if ((ch === "i" || ch === "u") && !VOWELS.has(prev) && !VOWELS.has(next)) {
    if (SONORANTS.has(next)) {
      return mark === GRAVE;
    }
    if (ch === "i") {
      return mark === GRAVE;
    }
    return mark !== ACUTE;
  }

  return true;
}

export function applyStress(word: string, pos: number, mark: string): string {
  const plain = stripAccents(normalizeLt(word));
  const chars = Array.from(plain);

  if (pos < 0 || pos >= chars.length) {
    return word;
  }

  chars.splice(pos + 1, 0, mark);
  return normalizeNotation(chars.join("").normalize("NFC"));
}

export function wordKey(text: string): string {
  return stripAccents(normalizeLt(text)).toLowerCase();
}

function normalizeLt(text: string): string {
  if (!text) {
    return "";
  }

  const out: string[] = [];
  let lastBase = "";

  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch)) {
      if (ch === COMBINING_DOT_ABOVE && I_DOT_BASES.has(lastBase)) {
        continue;
      }
      out.push(ch);
    } else {
      lastBase = ch;
      out.push(ch);
    }
  }

  return out.join("").normalize("NFC");
}

function stripAccents(text: string): string {
  if (!text) {
    return "";
  }

  const out: string[] = [];
  let lastBase = "";

  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch)) {
      if (STRESS_MARKS.has(ch)) {
        continue;
      }
      if (ch === COMBINING_DOT_ABOVE && I_DOT_BASES.has(lastBase)) {
        continue;
      }
      out.push(ch);
    } else {
      lastBase = ch;
      out.push(ch);
    }
  }

  return out.join("").normalize("NFC");
}

function normalizeNotation(text: string): string {
  if (!text || !hasStress(text)) {
    return text;
  }

  const clusters = graphemeClusters(text);
  const moves: Array<[number, number, string, string]> = [];

  for (let i = 0; i < clusters.length; i += 1) {
    const cluster = clusters[i]!;
    const base = plainBase(cluster);
    if (!base) {
      continue;
    }

    const next = i + 1 < clusters.length ? plainBase(clusters[i + 1]!) : null;
    const prev = i > 0 ? plainBase(clusters[i - 1]!) : null;
    const after = i + 2 < clusters.length ? clusters[i + 2]![0]!.toLowerCase() : "";

    if (cluster.includes(TILDE)) {
      if (
        next &&
        PURE_DIPHTHONGS.has(base + next) &&
        !PURE_DIPHTHONGS.has(next + after)
      ) {
        moves.push([i, i + 1, TILDE, TILDE]);
      } else if (
        "aeiu".includes(base) &&
        next &&
        SONORANTS.has(next) &&
        !VOWEL_BASES.has(after) &&
        !(prev && PURE_DIPHTHONGS.has(prev + base))
      ) {
        moves.push([i, i + 1, TILDE, TILDE]);
      }
    } else if (
      cluster.includes(ACUTE) &&
      SONORANTS.has(base) &&
      prev &&
      VOWEL_BASES.has(prev)
    ) {
      moves.push([i, i, ACUTE, TILDE]);
    }
  }

  for (const [src, dst, drop, add] of moves) {
    if (src !== dst && clusters[dst]!.some((mark) => STRESS_MARKS.has(mark))) {
      continue;
    }

    const dropIndex = clusters[src]!.indexOf(drop);
    if (dropIndex >= 0) {
      clusters[src]!.splice(dropIndex, 1);
      clusters[dst]!.push(add);
    }
  }

  return clusters.flat().join("").normalize("NFC");
}

function graphemeClusters(text: string): string[][] {
  const clusters: string[][] = [];

  for (const ch of text.normalize("NFD")) {
    if (isCombining(ch) && clusters.length) {
      clusters[clusters.length - 1]!.push(ch);
    } else {
      clusters.push([ch]);
    }
  }

  return clusters;
}

function plainBase(cluster: readonly string[]): string | null {
  if (cluster.slice(1).every((mark) => STRESS_MARKS.has(mark))) {
    return cluster[0]?.toLowerCase() ?? null;
  }

  return null;
}

function hasStress(text: string): boolean {
  return Array.from(text.normalize("NFD")).some((ch) => STRESS_MARKS.has(ch));
}

function isCombining(ch: string): boolean {
  return /\p{M}/u.test(ch);
}
