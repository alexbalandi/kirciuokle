export type Part = {
  text: string;
  accented?: string;
  type: "word" | "sep";
  ambiguous?: true;
  resolvedBy?: "lemma" | "context";
  variants?: Variant[];
  chosen?: number;
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
