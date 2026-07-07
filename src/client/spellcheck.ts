export type SpellcheckStatus = "ok" | "restore" | "typo" | "unknown";

export type SpellcheckSuggestion = {
  status: SpellcheckStatus;
  candidates: string[];
};

const WORDLIST_URL = "/spellcheck-lt.txt";
const MAX_CANDIDATES = 8;
const LT_DIACRITIC_RE = /[ąčęėįšųūž]/iu;
const ASCII_RE = /^[\x00-\x7F]+$/;
const FOLD_MAP: Record<string, string> = {
  ą: "a",
  č: "c",
  ę: "e",
  ė: "e",
  į: "i",
  š: "s",
  ų: "u",
  ū: "u",
  ž: "z",
  Ą: "a",
  Č: "c",
  Ę: "e",
  Ė: "e",
  Į: "i",
  Š: "s",
  Ų: "u",
  Ū: "u",
  Ž: "z",
};

let sharedEnginePromise: Promise<SpellcheckEngine> | null = null;

export function foldAscii(word: string): string {
  return Array.from(word.normalize("NFC"), (char) => FOLD_MAP[char] ?? char)
    .join("")
    .toLowerCase();
}

export function createSpellcheckEngine(forms: Iterable<string>): SpellcheckEngine {
  return new SpellcheckEngine(forms);
}

export async function suggest(word: string): Promise<SpellcheckSuggestion> {
  const engine = await loadSpellcheckEngine();
  return engine.suggest(word);
}

export async function loadSpellcheckEngine(): Promise<SpellcheckEngine> {
  sharedEnginePromise ??= fetchWordlist()
    .then((text) =>
      text
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
    )
    .then((forms) => new SpellcheckEngine(forms));

  return sharedEnginePromise;
}

export function resetSpellcheckForTests(): void {
  sharedEnginePromise = null;
}

export class SpellcheckEngine {
  readonly valid = new Set<string>();
  readonly foldIndex = new Map<string, string[]>();
  readonly deleteIndex = new Map<string, string[]>();

  private readonly forms: string[] = [];

  constructor(forms: Iterable<string>) {
    const seenForms = new Set<string>();

    for (const form of forms) {
      const normalized = normalizeForm(form);
      if (!normalized || seenForms.has(normalized)) {
        continue;
      }

      seenForms.add(normalized);
      this.forms.push(normalized);
      this.valid.add(normalized);

      if (LT_DIACRITIC_RE.test(normalized)) {
        pushUnique(this.foldIndex, foldAscii(normalized), normalized);
      }

      for (const deletion of deletes1(normalized)) {
        pushUnique(this.deleteIndex, deletion, normalized);
      }
    }

    for (const candidates of this.foldIndex.values()) {
      candidates.sort(compareForms);
    }
    for (const candidates of this.deleteIndex.values()) {
      candidates.sort(compareForms);
    }
  }

  suggest(word: string): SpellcheckSuggestion {
    const normalized = normalizeForm(word);
    if (!normalized) {
      return { status: "unknown", candidates: [] };
    }

    if (this.valid.has(normalized)) {
      return { status: "ok", candidates: [] };
    }

    const folded = foldAscii(normalized);
    const restoreCandidates = this.foldIndex.get(folded) ?? [];
    if (restoreCandidates.length > 0 && isPureAscii(word)) {
      return {
        status: "restore",
        candidates: rankCandidates(normalized, restoreCandidates).map((candidate) =>
          reapplyCase(candidate, word),
        ),
      };
    }

    const typoCandidates = this.typoCandidates(normalized);
    if (typoCandidates.length > 0) {
      return {
        status: "typo",
        candidates: rankCandidates(normalized, typoCandidates).map((candidate) =>
          reapplyCase(candidate, word),
        ),
      };
    }

    return { status: "unknown", candidates: [] };
  }

  private typoCandidates(query: string): string[] {
    const candidates = new Set<string>();
    collectEditCandidates(this.deleteIndex.get(query), query, candidates);

    for (const deletion of deletes1(query)) {
      collectEditCandidates(this.deleteIndex.get(deletion), query, candidates);
      if (this.valid.has(deletion) && editDistanceAtMostOne(query, deletion)) {
        candidates.add(deletion);
      }
    }

    candidates.delete(query);
    return [...candidates];
  }
}

async function fetchWordlist(): Promise<string> {
  const response = await fetch(WORDLIST_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${WORDLIST_URL}: ${response.status}`);
  }

  return response.text();
}

function normalizeForm(word: string): string {
  return word.normalize("NFC").toLowerCase().trim();
}

function isPureAscii(word: string): boolean {
  return ASCII_RE.test(word.normalize("NFC"));
}

function deletes1(word: string): string[] {
  const chars = Array.from(word);
  const deletions = new Set<string>();

  for (let index = 0; index < chars.length; index += 1) {
    deletions.add(`${chars.slice(0, index).join("")}${chars.slice(index + 1).join("")}`);
  }

  return [...deletions];
}

function collectEditCandidates(
  candidates: readonly string[] | undefined,
  query: string,
  out: Set<string>,
): void {
  if (!candidates) {
    return;
  }

  for (const candidate of candidates) {
    if (editDistanceAtMostOne(query, candidate)) {
      out.add(candidate);
    }
  }
}

function rankCandidates(query: string, candidates: readonly string[]): string[] {
  return [...new Set(candidates)]
    .sort((left, right) => {
      const edit = editDistance(query, left) - editDistance(query, right);
      if (edit !== 0) {
        return edit;
      }

      const diacritics = countLtDiacritics(left) - countLtDiacritics(right);
      if (diacritics !== 0) {
        return diacritics;
      }

      const length = Array.from(left).length - Array.from(right).length;
      return length || compareForms(left, right);
    })
    .slice(0, MAX_CANDIDATES);
}

function countLtDiacritics(word: string): number {
  return Array.from(word).filter((char) => LT_DIACRITIC_RE.test(char)).length;
}

function editDistanceAtMostOne(left: string, right: string): boolean {
  return editDistance(left, right) <= 1;
}

function editDistance(left: string, right: string): number {
  const a = Array.from(left);
  const b = Array.from(right);
  if (Math.abs(a.length - b.length) > 1) {
    return 2;
  }

  if (a.length === b.length) {
    let edits = 0;
    for (let index = 0; index < a.length; index += 1) {
      if (a[index] !== b[index]) {
        edits += 1;
        if (edits > 1) {
          return edits;
        }
      }
    }
    return edits;
  }

  const shorter = a.length < b.length ? a : b;
  const longer = a.length < b.length ? b : a;
  let edits = 0;
  let shortIndex = 0;
  let longIndex = 0;

  while (shortIndex < shorter.length && longIndex < longer.length) {
    if (shorter[shortIndex] === longer[longIndex]) {
      shortIndex += 1;
      longIndex += 1;
      continue;
    }

    edits += 1;
    if (edits > 1) {
      return edits;
    }
    longIndex += 1;
  }

  return edits + (longer.length - longIndex);
}

function reapplyCase(candidate: string, query: string): string {
  const normalizedQuery = query.normalize("NFC");
  if (hasCasedLetter(normalizedQuery) && normalizedQuery === normalizedQuery.toUpperCase()) {
    return candidate.toUpperCase();
  }

  const queryChars = Array.from(normalizedQuery);
  const first = queryChars[0] ?? "";
  const rest = queryChars.slice(1).join("");
  if (
    first &&
    first === first.toUpperCase() &&
    first !== first.toLowerCase() &&
    rest === rest.toLowerCase()
  ) {
    const candidateChars = Array.from(candidate);
    const candidateFirst = candidateChars[0] ?? "";
    return `${candidateFirst.toUpperCase()}${candidateChars.slice(1).join("")}`;
  }

  return candidate;
}

function hasCasedLetter(value: string): boolean {
  return value.toLowerCase() !== value.toUpperCase();
}

function pushUnique(map: Map<string, string[]>, key: string, value: string): void {
  const existing = map.get(key);
  if (existing) {
    if (!existing.includes(value)) {
      existing.push(value);
    }
    return;
  }

  map.set(key, [value]);
}

function compareForms(left: string, right: string): number {
  return left.localeCompare(right, "lt");
}
