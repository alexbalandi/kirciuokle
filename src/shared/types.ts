import type { TagSlots } from "./tags";

export type Part = {
  text: string;
  accented?: string;
  type: "word" | "sep";
  ambiguous?: true;
  resolvedBy?: "lemma" | "context";
  variants?: Variant[];
  chosen?: number;
  /** The exact reading (mi label) the context tagger matched, if any. */
  chosenMi?: string;
  /** The context tagger's tags for this word — lets the client score
      readings that were fetched after the accent response. */
  tokenTags?: TagSlots;
  unknown?: true;
};

export type AccentRequest = {
  text: string;
};

export type AccentResponse = {
  source: "local" | "vdu";
  parts: Part[];
  tagger: "ok" | "unavailable";
};

export type Variant = {
  form: string;
  info: string;
};

export type WordResponse = {
  variants: Variant[];
};

export type ErrorResponse = {
  error: string;
};
