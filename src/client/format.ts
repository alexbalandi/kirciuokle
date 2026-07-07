export function formatProbability(probability: number): string {
  const percent = probability * 100;
  return percent >= 10 ? `${Math.round(percent)}%` : `${percent.toFixed(1)}%`;
}
