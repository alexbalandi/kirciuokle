export type BatchableSentence = {
  index: number;
  subwordLength: number;
};

export function buildBatches<T extends BatchableSentence>(
  sentences: readonly T[],
  tokenBudget: number,
): T[][] {
  const sorted = [...sentences].sort((left, right) => {
    const byLength = right.subwordLength - left.subwordLength;
    return byLength || left.index - right.index;
  });
  const batches: T[][] = [];
  let current: T[] = [];
  let currentMax = 0;

  for (const sentence of sorted) {
    const nextMax = Math.max(currentMax, sentence.subwordLength);
    const nextCost = nextMax * (current.length + 1);

    if (current.length && nextCost > tokenBudget) {
      batches.push(current);
      current = [];
      currentMax = 0;
    }

    current.push(sentence);
    currentMax = Math.max(currentMax, sentence.subwordLength);
  }

  if (current.length) {
    batches.push(current);
  }

  return batches;
}

export function restoreSentenceOrder<T extends { index: number }>(items: readonly T[]): T[] {
  return [...items].sort((left, right) => left.index - right.index);
}
