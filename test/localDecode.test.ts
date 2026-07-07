import { describe, expect, it } from "vitest";
import {
  DEFAULT_STRESS_MARKS,
  applyStress,
  decodeStress,
  isValidStressTarget,
} from "../src/client/local/decode";

const [GRAVE, ACUTE, TILDE] = DEFAULT_STRESS_MARKS;

describe("local stress decode validity mask", () => {
  it("matches the train_guesser validity fixtures", () => {
    expect(isValidStressTarget("abatija", 4, GRAVE)).toBe(true);
    expect(isValidStressTarget("abatija", 4, ACUTE)).toBe(false);

    expect(isValidStressTarget("slėnio", 2, GRAVE)).toBe(false);

    expect(isValidStressTarget("vyras", 1, GRAVE)).toBe(false);
    expect(isValidStressTarget("vyras", 1, ACUTE)).toBe(true);

    expect(isValidStressTarget("pirko", 1, GRAVE)).toBe(true);
    expect(isValidStressTarget("pirko", 1, TILDE)).toBe(false);
    expect(isValidStressTarget("pirko", 2, TILDE)).toBe(true);
    expect(isValidStressTarget("pirko", 2, GRAVE)).toBe(false);

    expect(isValidStressTarget("vienas", 1, ACUTE)).toBe(true);

    expect(isValidStressTarget("namas", 1, GRAVE)).toBe(true);
    expect(isValidStressTarget("namas", 1, ACUTE)).toBe(true);
    expect(isValidStressTarget("namas", 1, TILDE)).toBe(true);
  });

  it("lets the no-stress cell win", () => {
    const logits = new Array(6 * DEFAULT_STRESS_MARKS.length).fill(-10);
    logits[1 * DEFAULT_STRESS_MARKS.length] = 0;

    expect(
      decodeStress(logits, 1, 0, "namas", DEFAULT_STRESS_MARKS, 6),
    ).toEqual({ noStress: true });
  });

  it("inserts marks with NFC-correct Lithuanian clusters", () => {
    expect(applyStress("mėnuo", 1, TILDE)).toBe("mė̃nuo".normalize("NFC"));
  });
});
