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
  /** Local-mode only: a numeral+suffix fragment (e.g. "81-erių") whose
      suffix is stressed but which has no meaningful POS reading. */
  numeralFragment?: true;
};

export type AccentRequest = {
  text: string;
  /** Raw CoNLL-U tags the client fetched from UDPipe (browser-side, from the
      user's own IP). Optional — the worker falls back to a server-side UDPipe
      call when absent. */
  tags?: string;
};

export type AccentResponse = {
  source: "local" | "vdu";
  parts: Part[];
  tagger: "ok" | "unavailable";
};

export type Variant = {
  form: string;
  info: string;
  /** Local-mode only: softmax probability of this reading (0..1). */
  p?: number;
};

export type WordResponse = {
  variants: Variant[];
};

export type ErrorResponse = {
  error: string;
};
