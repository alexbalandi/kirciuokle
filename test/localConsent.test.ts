import { describe, expect, it, vi } from "vitest";
import { createLocalDownloadGate } from "../src/client/local/consent";

describe("local download consent gate", () => {
  it("does not start the engine before consent when the model is not cached", async () => {
    const states: string[] = [];
    const ensureEngine = vi.fn(async () => {});
    const gate = createLocalDownloadGate({
      hasCachedModel: vi.fn(async () => false),
      ensureEngine,
      onState: (state) => states.push(state),
    });

    await expect(gate.enterLocalMode()).resolves.toBe("needs-consent");

    expect(ensureEngine).not.toHaveBeenCalled();
    expect(gate.state).toBe("needs-consent");
    expect(states).toEqual(["checking-cache", "needs-consent"]);

    await expect(gate.consentToDownload()).resolves.toBe("ready");

    expect(ensureEngine).toHaveBeenCalledTimes(1);
    expect(states).toEqual(["checking-cache", "needs-consent", "loading", "ready"]);
  });

  it("starts the engine immediately when the model is already cached", async () => {
    const ensureEngine = vi.fn(async () => {});
    const gate = createLocalDownloadGate({
      hasCachedModel: vi.fn(async () => true),
      ensureEngine,
    });

    await expect(gate.enterLocalMode()).resolves.toBe("ready");

    expect(ensureEngine).toHaveBeenCalledTimes(1);
    expect(gate.state).toBe("ready");
  });
});
