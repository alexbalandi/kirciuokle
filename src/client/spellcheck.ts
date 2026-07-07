export type SpellcheckStatus = "ok" | "restore" | "typo" | "unknown";

export type SpellcheckSuggestion = {
  status: SpellcheckStatus;
  candidates: string[];
  /** For `restore`: the candidate "fix all" should apply automatically — set only
      when unambiguous (one candidate, or one clearly dominant by frequency). */
  autofix?: string;
};

// Words this short aren't worth spellchecking: a single letter has no fold
// candidates (the wordlist starts at length 2) and matches dozens of two-letter
// words by one insertion, so it produces noise, not corrections.
const MIN_CHECK_LENGTH = 2;

// The accept vocabulary (`valid`) is huge; suggestions come from a common subset.
// On a large production list, only forms this frequent get restore/typo indexes
// (bounds memory + noise). Small hand-built lists (tests) index everything.
const CORRECTION_MIN_FREQ = 5;
const CORRECTION_GATE_LIST_SIZE = 5_000;

// A pure-ASCII word that is itself accepted is still treated as a diacritic-DROP
// (offer restore) when its top diacritic form is both common and dramatically more
// frequent than the word itself — e.g. "as"→"aš" (50×), "už" (104×). A genuine
// ASCII word's diacritic sibling isn't (padaryta→padarytą is 0.1×). Big margin.
const RESTORE_DOMINANCE = 8;
const RESTORE_MIN_FREQ = 100;

// Combining stress marks (grave/acute/tilde) — stripped before lookup so already
// accented text (our own output, or a pasted accented article) still matches the
// un-stressed wordlist. LT diacritics (ogonek/caron/macron/dot) are NOT stripped.
const STRESS_MARKS_RE = /[̀́̃]/g;

export type SpellcheckContext = { prev?: string; next?: string };

const WORDLIST_URL = "/spellcheck-lt.txt";
const BIGRAMS_URL = "/spellcheck-bigrams.txt";
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
let sharedBigramsPromise: Promise<void> | null = null;

export function foldAscii(word: string): string {
  return Array.from(word.normalize("NFC"), (char) => FOLD_MAP[char] ?? char)
    .join("")
    .toLowerCase();
}

export function createSpellcheckEngine(
  forms: Iterable<string>,
  bigrams?: Iterable<string> | ReadonlyMap<string, number>,
): SpellcheckEngine {
  return new SpellcheckEngine(forms, bigrams);
}

export async function suggest(
  word: string,
  context?: SpellcheckContext,
): Promise<SpellcheckSuggestion> {
  const engine = await loadSpellcheckEngine();
  if (hasContext(context)) {
    await ensureBigrams();
  }
  return engine.suggest(word, context);
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
  sharedBigramsPromise = null;
}

export class SpellcheckEngine {
  readonly valid = new Set<string>();
  readonly freq = new Map<string, number>();
  readonly bigrams = new Map<string, number>();
  readonly foldIndex = new Map<string, string[]>();
  readonly deleteIndex = new Map<string, string[]>();

  private readonly forms: string[] = [];

  constructor(
    forms: Iterable<string>,
    bigrams?: Iterable<string> | ReadonlyMap<string, number>,
  ) {
    const seenForms = new Set<string>();

    for (const line of forms) {
      const { form, freq } = parseWordlistLine(line);
      const normalized = normalizeForm(form);
      if (!normalized || seenForms.has(normalized)) {
        continue;
      }

      seenForms.add(normalized);
      this.forms.push(normalized);
      this.valid.add(normalized);
      this.freq.set(normalized, freq);
    }

    // Two-tier vocabulary (SPEC56 §1a-bis): every form is "accepted" (added to
    // `valid`), but the correction indexes — restore (foldIndex) and typo
    // (deleteIndex) — are built only over the common, frequent subset. The accept
    // set now also carries hundreds of thousands of freq-list words so real text
    // isn't false-flagged; those must NOT all get delete keys (memory + noise), so
    // the correction subset is gated by frequency. Small hand-built lists (tests)
    // are below the gate size and index every form.
    const gateByFrequency = this.forms.length > CORRECTION_GATE_LIST_SIZE;
    for (const normalized of this.forms) {
      if (gateByFrequency && (this.freq.get(normalized) ?? 0) < CORRECTION_MIN_FREQ) {
        continue;
      }

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

    if (bigrams) {
      this.setBigrams(bigrams);
    }
  }

  suggest(word: string, context?: SpellcheckContext): SpellcheckSuggestion {
    const normalized = normalizeForm(word);
    if (!normalized || Array.from(normalized).length < MIN_CHECK_LENGTH) {
      return { status: "ok", candidates: [] };
    }

    const inValid = this.valid.has(normalized);

    // Restore (ASCII → diacritics), ranked frequency-first so the most common real
    // word wins (aciu → ačiū 14413, not ačiu 83). Fires when the word isn't
    // accepted at all, OR when it IS accepted but is a dominant diacritic-drop
    // (see RESTORE_DOMINANCE) — that's how "as"/"aciu" restore even though the
    // frequency list also lists them, while "padaryta"/"kur" stay accepted.
    if (isPureAscii(word)) {
      const restoreCandidates = this.foldIndex.get(foldAscii(normalized)) ?? [];
      if (restoreCandidates.length > 0) {
        const ranked = rankCandidates(
          normalized,
          restoreCandidates,
          this,
          context,
          restoreDistanceBand,
          true,
        );
        const best = ranked[0];
        if (best && (!inValid || this.isDominantDrop(normalized, best))) {
          const autofix = dominantRestore(ranked, this.freq);
          return {
            status: "restore",
            candidates: ranked.map((candidate) => reapplyCase(candidate, word)),
            autofix: autofix ? reapplyCase(autofix, word) : undefined,
          };
        }
      }
    }

    if (inValid) {
      return { status: "ok", candidates: [] };
    }

    // Typo suggestions for a Capitalized / ALL-CAPS word are almost always wrong —
    // it's a proper noun or acronym (Ankaroje, Erdogano, NATO), not a misspelling.
    // Restore (handled above) still applies, e.g. sentence-initial "Aciu" → "Ačiū".
    const typoCandidates = startsWithUppercase(word) ? [] : this.typoCandidates(normalized);
    if (typoCandidates.length > 0) {
      return {
        status: "typo",
        candidates: rankCandidates(
          normalized,
          typoCandidates,
          this,
          context,
          boundedDamerauLevenshteinCap2,
        ).map((candidate) => reapplyCase(candidate, word)),
      };
    }

    return { status: "unknown", candidates: [] };
  }

  setBigrams(lines: Iterable<string> | ReadonlyMap<string, number>): void {
    this.bigrams.clear();

    if (isReadonlyBigramMap(lines)) {
      for (const [key, count] of lines) {
        const parsed = parseBigramKey(key, count);
        if (parsed) {
          this.bigrams.set(parsed.key, parsed.count);
        }
      }
      return;
    }

    for (const line of lines) {
      const parsed = parseBigramLine(line);
      if (parsed) {
        this.bigrams.set(parsed.key, parsed.count);
      }
    }
  }

  // An accepted pure-ASCII word is nonetheless a diacritic-drop worth restoring
  // when its top diacritic form is common (>= RESTORE_MIN_FREQ) AND far more
  // frequent than the word itself (>= RESTORE_DOMINANCE×).
  private isDominantDrop(word: string, candidate: string): boolean {
    const candidateFreq = this.freq.get(candidate) ?? 0;
    const wordFreq = this.freq.get(word) ?? 0;
    return (
      candidateFreq >= RESTORE_MIN_FREQ &&
      candidateFreq >= RESTORE_DOMINANCE * Math.max(wordFreq, 1)
    );
  }

  private typoCandidates(query: string): string[] {
    const candidates = new Set<string>();
    collectEditCandidates(this.deleteIndex.get(query), query, candidates);

    const queryDeletes = new Set([...deletes1(query), ...deletes2(query)]);
    for (const deletion of queryDeletes) {
      collectEditCandidates(this.deleteIndex.get(deletion), query, candidates);

      if (
        this.valid.has(deletion) &&
        boundedDamerauLevenshteinCap2(query, deletion) <= 2
      ) {
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

async function fetchBigrams(): Promise<string> {
  const response = await fetch(BIGRAMS_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${BIGRAMS_URL}: ${response.status}`);
  }

  return response.text();
}

async function ensureBigrams(): Promise<void> {
  sharedBigramsPromise ??= Promise.all([loadSpellcheckEngine(), fetchBigrams()])
    .then(([engine, text]) => {
      engine.setBigrams(
        text
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
      );
    })
    .catch(() => {
      // Context ranking is opportunistic; spelling suggestions still work
      // without the small bigram table.
    });

  return sharedBigramsPromise;
}

function parseWordlistLine(line: string): { form: string; freq: number } {
  const [form = "", freqRaw = ""] = line.split("\t", 2);
  const freq = Number.parseInt(freqRaw, 10);
  return {
    form,
    freq: Number.isFinite(freq) && freq > 0 ? freq : 0,
  };
}

function parseBigramLine(line: string): { key: string; count: number } | null {
  const [prev = "", next = "", countRaw = ""] = line.split("\t", 3);
  return parseBigramParts(prev, next, countRaw);
}

function parseBigramKey(
  key: string,
  count: number,
): { key: string; count: number } | null {
  const [prev = "", next = ""] = key.split("\t", 2);
  return parseBigramParts(prev, next, String(count));
}

function parseBigramParts(
  prevRaw: string,
  nextRaw: string,
  countRaw: string,
): { key: string; count: number } | null {
  const prev = normalizeForm(prevRaw);
  const next = normalizeForm(nextRaw);
  const count = Number.parseInt(countRaw, 10);
  if (!prev || !next || !Number.isFinite(count) || count <= 0) {
    return null;
  }

  return { key: `${prev}\t${next}`, count };
}

function isReadonlyBigramMap(
  lines: Iterable<string> | ReadonlyMap<string, number>,
): lines is ReadonlyMap<string, number> {
  return typeof (lines as ReadonlyMap<string, number>).get === "function";
}

function normalizeForm(word: string): string {
  return word
    .normalize("NFD")
    .replace(STRESS_MARKS_RE, "")
    .normalize("NFC")
    .toLowerCase()
    .trim();
}

function startsWithUppercase(word: string): boolean {
  const first = Array.from(word.normalize("NFC"))[0] ?? "";
  return first !== "" && first === first.toUpperCase() && first !== first.toLowerCase();
}

function normalizeContextWord(word: string | undefined): string | undefined {
  const normalized = normalizeForm(word ?? "");
  return normalized || undefined;
}

function hasContext(context: SpellcheckContext | undefined): boolean {
  return Boolean(normalizeContextWord(context?.prev) || normalizeContextWord(context?.next));
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

function deletes2(word: string): string[] {
  const deletions = new Set<string>();
  for (const deletion of deletes1(word)) {
    for (const nested of deletes1(deletion)) {
      deletions.add(nested);
    }
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
    if (boundedDamerauLevenshteinCap2(query, candidate) <= 2) {
      out.add(candidate);
    }
  }
}

function rankCandidates(
  query: string,
  candidates: readonly string[],
  engine: SpellcheckEngine,
  context: SpellcheckContext | undefined,
  distance: (query: string, candidate: string) => number,
  // Restore: rank frequency before the distance band (a common word is the
  // intended one even if it restores more diacritics). Typo: keep edit distance
  // primary (a 1-edit fix beats a 2-edit one).
  preferFrequency = false,
): string[] {
  const normalizedContext = {
    prev: normalizeContextWord(context?.prev),
    next: normalizeContextWord(context?.next),
  };

  const editScore = (left: string, right: string): number =>
    distance(query, left) - distance(query, right);
  const contextScore = (left: string, right: string): number =>
    contextScoreFor(engine, normalizedContext, right) -
    contextScoreFor(engine, normalizedContext, left);
  const freqScore = (left: string, right: string): number =>
    (engine.freq.get(right) ?? 0) - (engine.freq.get(left) ?? 0);

  return [...new Set(candidates)]
    .sort((left, right) => {
      const primary = preferFrequency
        ? contextScore(left, right) ||
          freqScore(left, right) ||
          editScore(left, right)
        : editScore(left, right) ||
          contextScore(left, right) ||
          freqScore(left, right);
      if (primary !== 0) {
        return primary;
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

// The candidate "fix all" may apply for a restore without asking: either the only
// option, or one whose frequency dominates the runner-up by a wide margin (so
// there's effectively one common word for that ASCII spelling).
function dominantRestore(
  ranked: readonly string[],
  freq: ReadonlyMap<string, number>,
): string | undefined {
  if (ranked.length === 0) {
    return undefined;
  }
  if (ranked.length === 1) {
    return ranked[0];
  }
  const top = freq.get(ranked[0]!) ?? 0;
  const second = freq.get(ranked[1]!) ?? 0;
  return top > 0 && top >= Math.max(second * 8, second + 1) ? ranked[0] : undefined;
}

function restoreDistanceBand(query: string, candidate: string): number {
  const queryChars = Array.from(query);
  const candidateChars = Array.from(candidate);
  if (queryChars.length !== candidateChars.length) {
    return boundedDamerauLevenshteinCap2(query, candidate);
  }

  let substitutions = 0;
  for (let index = 0; index < queryChars.length; index += 1) {
    if (queryChars[index] !== candidateChars[index]) {
      substitutions += 1;
    }
  }

  return Math.min(substitutions, 3);
}

function contextScoreFor(
  engine: SpellcheckEngine,
  context: { prev?: string; next?: string },
  candidate: string,
): number {
  if (engine.bigrams.size === 0) {
    return 0;
  }

  const prevScore = context.prev
    ? (engine.bigrams.get(`${context.prev}\t${candidate}`) ?? 0)
    : 0;
  const nextScore = context.next
    ? (engine.bigrams.get(`${candidate}\t${context.next}`) ?? 0)
    : 0;
  return prevScore + nextScore;
}

function countLtDiacritics(word: string): number {
  return Array.from(word).filter((char) => LT_DIACRITIC_RE.test(char)).length;
}

function boundedDamerauLevenshteinCap2(left: string, right: string): number {
  const a = Array.from(left);
  const b = Array.from(right);
  if (Math.abs(a.length - b.length) > 2) {
    return 3;
  }

  const rows = a.length + 1;
  const columns = b.length + 1;
  const dp: number[][] = Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => 3),
  );

  for (let row = 0; row < rows; row += 1) {
    dp[row]![0] = Math.min(row, 3);
  }
  for (let column = 0; column < columns; column += 1) {
    dp[0]![column] = Math.min(column, 3);
  }

  for (let row = 1; row < rows; row += 1) {
    for (let column = 1; column < columns; column += 1) {
      const substitutionCost = a[row - 1] === b[column - 1] ? 0 : 1;
      let best = Math.min(
        dp[row - 1]![column]! + 1,
        dp[row]![column - 1]! + 1,
        dp[row - 1]![column - 1]! + substitutionCost,
      );

      if (
        row > 1 &&
        column > 1 &&
        a[row - 1] === b[column - 2] &&
        a[row - 2] === b[column - 1]
      ) {
        best = Math.min(best, dp[row - 2]![column - 2]! + 1);
      }

      dp[row]![column] = Math.min(best, 3);
    }
  }

  return Math.min(dp[a.length]![b.length]!, 3);
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
