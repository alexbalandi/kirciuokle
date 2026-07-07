import { describe, expect, it } from "vitest";
import { formatProbability } from "../src/client/format";
import { formatBytes } from "../src/client/local/assets";

describe("client formatting", () => {
  it("formats local model sizes as rounded decimal MB", () => {
    expect(formatBytes(537_586_710)).toBe("538 MB");
    expect(formatBytes(156_384_275)).toBe("156 MB");
  });

  it("formats probabilities with integer percents at 10% and one decimal below", () => {
    expect(formatProbability(0.674)).toBe("67%");
    expect(formatProbability(0.1)).toBe("10%");
    expect(formatProbability(0.099)).toBe("9.9%");
    expect(formatProbability(0.084)).toBe("8.4%");
  });
});
