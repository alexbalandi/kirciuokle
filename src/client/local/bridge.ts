import { SCORING_SLOTS, type TagSlots } from "../../shared/tags";
import type { BridgeSlots, LabelBridge, MiVocabEntry, PosRow } from "./types";

const POS_PROB_CUT = 0.1;
const MAX_POPOVER_ROWS = 5;

export function buildLabelBridgeCache(
  bridge: LabelBridge,
  labels: readonly string[],
): Map<string, string> {
  const miVocab = Array.isArray(bridge.mi_vocab) ? bridge.mi_vocab : [];
  const modelLabels = bridge.model_labels ?? {};
  const cache = new Map<string, string>();

  for (const label of labels) {
    cache.set(label, bestMiForSlots(modelLabels[label] ?? {}, miVocab));
  }

  return cache;
}

export function decodePosRows(
  logits: ArrayLike<number>,
  offset: number,
  labels: readonly string[],
  labelBridgeCache: ReadonlyMap<string, string>,
  options: { probabilityCut?: number; maxRows?: number } = {},
): PosRow[] {
  const probabilityCut = options.probabilityCut ?? POS_PROB_CUT;
  const maxRows = options.maxRows ?? MAX_POPOVER_ROWS;
  let max = -Infinity;

  for (let i = 0; i < labels.length; i += 1) {
    max = Math.max(max, logits[offset + i] ?? -Infinity);
  }

  let denom = 0;
  for (let i = 0; i < labels.length; i += 1) {
    denom += Math.exp((logits[offset + i] ?? -Infinity) - max);
  }

  const merged = new Map<string, number>();
  for (let i = 0; i < labels.length; i += 1) {
    const probability = Math.exp((logits[offset + i] ?? -Infinity) - max) / denom;
    if (probability <= probabilityCut) {
      continue;
    }

    const label = labels[i]!;
    const mi = labelBridgeCache.get(label) || label;
    merged.set(mi, (merged.get(mi) ?? 0) + probability);
  }

  return Array.from(merged, ([label, probability]) => ({ label, probability }))
    .sort(
      (left, right) =>
        right.probability - left.probability || left.label.localeCompare(right.label),
    )
    .slice(0, maxRows);
}

export function bestMiForSlots(
  contextSlots: BridgeSlots,
  miVocab: readonly MiVocabEntry[],
): string {
  let best: { label: string; score: number; spurious: number } | null = null;

  for (const candidate of miVocab) {
    const candidateSlots = candidate.slots ?? {};
    const score = scoreTags(candidateSlots, contextSlots);
    const spurious = spuriousSlots(candidateSlots, contextSlots);

    if (
      !best ||
      score > best.score ||
      (score === best.score && spurious < best.spurious) ||
      (score === best.score &&
        spurious === best.spurious &&
        candidate.label.length < best.label.length)
    ) {
      best = { label: candidate.label, score, spurious };
    }
  }

  return best?.label ?? "";
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

export function spuriousSlots(variantTags: TagSlots, contextTags: TagSlots): number {
  let count = 0;

  for (const slot of Object.keys(variantTags) as Array<keyof TagSlots>) {
    if (!(slot in contextTags)) {
      count += 1;
    }
  }

  return count;
}
