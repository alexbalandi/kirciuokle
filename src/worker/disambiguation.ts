import type { Variant } from "../shared/types";

export type Token = {
  form: string;
  lemma: string;
  upos: string;
  xpos: string;
  feats: Record<string, string>;
};

export type AccentVariant = Variant & {
  mi: string[];
};

type Slot =
  | "pos"
  | "gender"
  | "number"
  | "case"
  | "tense"
  | "person"
  | "voice"
  | "degree";

export type TagSlots = Partial<Record<Slot, string>>;

type AlignablePart = {
  string?: string;
  type?: string;
};

export const MI_TAGS: Record<string, [Slot, string]> = {
  "dkt.": ["pos", "NOUN"],
  "bdv.": ["pos", "ADJ"],
  "vksm.": ["pos", "VERB"],
  "dlv.": ["pos", "PART_VERB"],
  "psdlv.": ["pos", "PART_VERB"],
  "padlv.": ["pos", "PART_VERB"],
  "prv.": ["pos", "ADV"],
  "įv.": ["pos", "PRON"],
  "sktv.": ["pos", "NUM"],
  "jng.": ["pos", "CCONJ"],
  "prl.": ["pos", "ADP"],
  "dll.": ["pos", "PART"],
  "jst.": ["pos", "INTJ"],
  "vyr. g.": ["gender", "Masc"],
  "mot. g.": ["gender", "Fem"],
  "bev. g.": ["gender", "Neut"],
  "vns.": ["number", "Sing"],
  "dgs.": ["number", "Plur"],
  "vard.": ["case", "Nom"],
  "kilm.": ["case", "Gen"],
  "naud.": ["case", "Dat"],
  "gal.": ["case", "Acc"],
  "įnag.": ["case", "Ins"],
  "viet.": ["case", "Loc"],
  "šauksm.": ["case", "Voc"],
  "es. l.": ["tense", "Pres"],
  "būt. l.": ["tense", "Past"],
  "būt. k. l.": ["tense", "Past"],
  "būt. d. l.": ["tense", "PastIter"],
  "būs. l.": ["tense", "Fut"],
  "1 asm.": ["person", "1"],
  "2 asm.": ["person", "2"],
  "3 asm.": ["person", "3"],
  "veik. r.": ["voice", "Act"],
  "neveik. r.": ["voice", "Pass"],
  "aukšt. l.": ["degree", "Cmp"],
  "aukšč. l.": ["degree", "Sup"],
};

export const LEMMA_EXCEPTIONS: Record<string, string> = {
  "yra\u0000būti": "yrà",
  "yra\u0000irti": "ỹra",
};

const MI_TAG_KEYS = Object.keys(MI_TAGS).sort((a, b) => b.length - a.length);
const SCORING_SLOTS: Slot[] = [
  "case",
  "gender",
  "number",
  "tense",
  "person",
  "voice",
  "degree",
];

export function parseConllu(conllu: string): Token[] {
  const tokens: Token[] = [];

  for (const line of conllu.split(/\r?\n/)) {
    if (!line || line.startsWith("#")) {
      continue;
    }

    const columns = line.split("\t");
    if (columns.length < 6 || !/^\d+$/.test(columns[0] ?? "")) {
      continue;
    }

    tokens.push({
      form: columns[1] ?? "",
      lemma: columns[2] ?? "",
      upos: columns[3] ?? "",
      xpos: columns[4] ?? "",
      feats: parseFeats(columns[5] ?? "_"),
    });
  }

  return tokens;
}

function parseFeats(raw: string): Record<string, string> {
  if (raw === "_") {
    return {};
  }

  const feats: Record<string, string> = {};
  for (const feature of raw.split("|")) {
    const separator = feature.indexOf("=");
    if (separator <= 0) {
      continue;
    }

    feats[feature.slice(0, separator)] = feature.slice(separator + 1);
  }

  return feats;
}

export function alignTokens(
  parts: AlignablePart[],
  tokens: Token[],
): Array<Token | null> {
  // VDU only emits letter tokens; drop UDPipe's number/punctuation tokens,
  // otherwise digit-heavy text (dates, scores) desyncs the scan window and
  // disambiguation silently degrades to defaults.
  tokens = tokens.filter((token) => /\p{L}/u.test(token.form));
  const aligned: Array<Token | null> = [];
  let tokenIndex = 0;

  for (const part of parts) {
    if (part.type !== "WORD" && part.type !== "NON_LT") {
      continue;
    }

    let found: Token | null = null;
    const original = part.string ?? "";
    const scanEnd = Math.min(tokenIndex + 8, tokens.length);

    for (let index = tokenIndex; index < scanEnd; index += 1) {
      if (tokens[index]?.form.toLowerCase() === original.toLowerCase()) {
        found = tokens[index] ?? null;
        tokenIndex = index + 1;
        break;
      }
    }

    aligned.push(found);
  }

  return aligned;
}

export function parseMi(mi: string): TagSlots {
  const tags: TagSlots = {};
  let remaining = mi.trim();

  for (const abbreviation of MI_TAG_KEYS) {
    if (!remaining.includes(abbreviation)) {
      continue;
    }

    const [slot, value] = MI_TAGS[abbreviation]!;
    tags[slot] ??= value;
    remaining = remaining.split(abbreviation).join(" ");
  }

  return tags;
}

export function tokenTags(token: Token): TagSlots {
  const tags: TagSlots = {};

  if (token.upos === "VERB" || token.upos === "AUX") {
    tags.pos = token.feats.VerbForm === "Part" ? "PART_VERB" : "VERB";
  } else if (token.upos === "NOUN" || token.upos === "PROPN") {
    tags.pos = "NOUN";
  } else if (token.upos === "DET") {
    // POS family follows VDU conventions: no DET in Lithuanian traditional grammar; see docs/SPEC13.md.
    tags.pos = "PRON";
  } else if (token.upos === "CCONJ" || token.upos === "SCONJ") {
    tags.pos = "CCONJ";
  } else {
    tags.pos = token.upos;
  }

  copyFeature(tags, token.feats, "gender", "Gender");
  copyFeature(tags, token.feats, "number", "Number");
  copyFeature(tags, token.feats, "case", "Case");
  copyFeature(tags, token.feats, "tense", "Tense");
  copyFeature(tags, token.feats, "person", "Person");
  copyFeature(tags, token.feats, "voice", "Voice");

  const degree = token.feats.Degree;
  if (degree && degree !== "Pos") {
    tags.degree = degree;
  }

  return tags;
}

function copyFeature(
  tags: TagSlots,
  feats: Record<string, string>,
  slot: Slot,
  feature: string,
): void {
  const value = feats[feature];
  if (value) {
    tags[slot] = value;
  }
}

export function scoreTags(variantTags: TagSlots, contextTags: TagSlots): number {
  let score = 0;

  if (variantTags.pos && contextTags.pos) {
    score += variantTags.pos === contextTags.pos ? 4 : -3;
  }

  for (const slot of SCORING_SLOTS) {
    const variantValue = variantTags[slot];
    const contextValue = contextTags[slot];

    if (!variantValue || !contextValue) {
      continue;
    }

    score += variantValue === contextValue ? 2 : -2;
  }

  return score;
}

export function scoreVariant(variant: AccentVariant, contextTags: TagSlots): number {
  if (variant.mi.length === 0) {
    return 0;
  }

  return Math.max(
    ...variant.mi.map((label) => scoreTags(parseMi(label), contextTags)),
  );
}

export function pickReadingMi(
  variants: AccentVariant[],
  contextTags: TagSlots,
): string | undefined {
  let best: { label: string; score: number } | null = null;

  for (const variant of variants) {
    for (const label of variant.mi) {
      const score = scoreTags(parseMi(label), contextTags);
      if (!best || score > best.score) {
        best = { label, score };
      }
    }
  }

  return best && best.score > 0 ? best.label : undefined;
}

export function pickVariant(
  word: string,
  variants: AccentVariant[],
  token: Token | null,
  defaultForm?: string,
): { index: number | null; resolvedBy?: "lemma" | "context" } {
  if (variants.length === 0) {
    return { index: null };
  }

  const defaultIndex = findVariantIndex(variants, defaultForm);
  const fallbackIndex = defaultIndex >= 0 ? defaultIndex : 0;

  if (!token) {
    return { index: fallbackIndex };
  }

  const exceptionForm = LEMMA_EXCEPTIONS[`${word.toLowerCase()}\u0000${token.lemma}`];
  if (exceptionForm) {
    const exceptionIndex = findVariantIndex(variants, exceptionForm);
    if (exceptionIndex >= 0) {
      return { index: exceptionIndex, resolvedBy: "lemma" };
    }
  }

  const contextTags = tokenTags(token);
  const scored = variants
    .map((variant, index) => ({
      index,
      score: scoreVariant(variant, contextTags),
    }))
    .sort((a, b) => b.score - a.score || a.index - b.index);

  if (scored.length > 1 && scored[0]!.score > scored[1]!.score) {
    return { index: scored[0]!.index, resolvedBy: "context" };
  }

  return { index: fallbackIndex };
}

function findVariantIndex(
  variants: AccentVariant[],
  form: string | undefined,
): number {
  if (!form) {
    return -1;
  }

  const normalized = form.normalize("NFC");
  return variants.findIndex((variant) => variant.form.normalize("NFC") === normalized);
}

export function matchCase(accented: string, original: string): string {
  if (original.length > 1 && original.toUpperCase() === original) {
    return accented.toUpperCase();
  }

  if (original[0] && original[0].toUpperCase() === original[0]) {
    return accented[0] ? accented[0].toUpperCase() + accented.slice(1) : accented;
  }

  return accented;
}

export function toPublicVariants(variants: AccentVariant[]): Variant[] {
  return variants.map(({ form, info }) => ({ form, info: dedupeInfo(info) }));
}

// Cached VDU entries can carry repeated readings ("prl.; prl.; prl.").
function dedupeInfo(info: string): string {
  return [
    ...new Set(
      info
        .split("; ")
        .map((reading) => reading.trim())
        .filter(Boolean),
    ),
  ].join("; ");
}
