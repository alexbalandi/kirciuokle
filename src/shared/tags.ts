// Morphology tag scoring shared by the worker (context disambiguation) and
// the client (marking the matched reading for lazily-loaded words).

export type Slot =
  | "pos"
  | "gender"
  | "number"
  | "case"
  | "tense"
  | "person"
  | "voice"
  | "degree";

export type TagSlots = Partial<Record<Slot, string>>;

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

export const MI_TAG_KEYS = Object.keys(MI_TAGS).sort((a, b) => b.length - a.length);

export const SCORING_SLOTS: Slot[] = [
  "case",
  "gender",
  "number",
  "tense",
  "person",
  "voice",
  "degree",
];

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
