import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveActiveLoad, resolveModelTierInfo } from "../src/client/local/assets";
import type { ModelManifest } from "../src/client/local/types";

class MemoryCache {
  readonly responses = new Map<string, Response>();

  async match(input: RequestInfo | URL): Promise<Response | undefined> {
    const url = input instanceof Request ? input.url : String(input);
    return this.responses.get(url)?.clone();
  }

  async keys(): Promise<Request[]> {
    return Array.from(this.responses.keys(), (url) => new Request(url));
  }

  async delete(input: RequestInfo | URL): Promise<boolean> {
    const url = input instanceof Request ? input.url : String(input);
    return this.responses.delete(url);
  }

  async put(input: RequestInfo | URL, response: Response): Promise<void> {
    const url = input instanceof Request ? input.url : String(input);
    this.responses.set(url, response.clone());
  }
}

function makeStorage(): Storage {
  const values = new Map<string, string>();
  return {
    get length() {
      return values.size;
    },
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    key: (index: number) => Array.from(values.keys())[index] ?? null,
    removeItem: (key: string) => {
      values.delete(key);
    },
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  };
}

function cachedResponse(bytes: number): Response {
  return new Response(null, {
    headers: { "content-length": String(bytes) },
  });
}

function activeKey(tier: string): string {
  return `accent-local-active-v1-${tier}`;
}

describe("local model asset manifest", () => {
  let cache: MemoryCache;
  let storage: Storage;

  beforeEach(() => {
    cache = new MemoryCache();
    storage = makeStorage();
    vi.stubGlobal("window", {
      location: { href: "https://example.test/app" },
      caches: {},
    });
    vi.stubGlobal("localStorage", storage);
    vi.stubGlobal("caches", {
      open: vi.fn(async () => cache),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves light and heavy tiers to distinct files and byte sizes", () => {
    const manifest: ModelManifest = {
      default_model: "joint.int8.partial.onnx",
      version: "release-1",
      tiers: {
        light: "joint.int8.full.onnx",
        heavy: "joint.int8.partial.onnx",
      },
      models: {
        "joint.int8.partial.onnx": {
          bytes: 470_223_894,
          tier: "heavy",
        },
        "joint.int8.full.onnx": {
          bytes: 139_543_571,
          tier: "light",
        },
      },
    };

    expect(resolveModelTierInfo(manifest, "light")).toEqual({
      tier: "light",
      modelFile: "joint.int8.full.onnx",
      bytes: 139_543_571,
      version: "release-1",
    });
    expect(resolveModelTierInfo(manifest, "heavy")).toEqual({
      tier: "heavy",
      modelFile: "joint.int8.partial.onnx",
      bytes: 470_223_894,
      version: "release-1",
    });
  });

  it("resolves a cached active file and reports an update when current differs", async () => {
    const manifest: ModelManifest = {
      version: "release-2",
      tiers: { heavy: "new-heavy.onnx" },
      models: {
        "new-heavy.onnx": { bytes: 456, tier: "heavy" },
      },
    };
    storage.setItem(
      activeKey("heavy"),
      JSON.stringify({ file: "old-heavy.onnx", bytes: 123, version: "release-1" }),
    );
    cache.responses.set(
      "https://example.test/local-model/old-heavy.onnx",
      cachedResponse(123),
    );

    await expect(resolveActiveLoad(manifest, "heavy")).resolves.toEqual({
      loadFile: "old-heavy.onnx",
      loadBytes: 123,
      loadVersion: "release-1",
      updateAvailable: true,
      updateFile: "new-heavy.onnx",
      updateBytes: 456,
      updateVersion: "release-2",
    });
  });

  it("loads the current file when an active record is not cached", async () => {
    const manifest: ModelManifest = {
      version: "release-2",
      tiers: { heavy: "new-heavy.onnx" },
      models: {
        "new-heavy.onnx": { bytes: 456, tier: "heavy" },
      },
    };
    storage.setItem(
      activeKey("heavy"),
      JSON.stringify({ file: "old-heavy.onnx", bytes: 123, version: "release-1" }),
    );

    await expect(resolveActiveLoad(manifest, "heavy")).resolves.toEqual({
      loadFile: "new-heavy.onnx",
      loadBytes: 456,
      loadVersion: "release-2",
      updateAvailable: false,
      updateFile: "new-heavy.onnx",
      updateBytes: 456,
      updateVersion: "release-2",
    });
  });

  it("loads the current file when no active record exists", async () => {
    const manifest: ModelManifest = {
      version: "release-2",
      tiers: { light: "new-light.onnx" },
      models: {
        "new-light.onnx": { bytes: 234, tier: "light" },
      },
    };

    await expect(resolveActiveLoad(manifest, "light")).resolves.toEqual({
      loadFile: "new-light.onnx",
      loadBytes: 234,
      loadVersion: "release-2",
      updateAvailable: false,
      updateFile: "new-light.onnx",
      updateBytes: 234,
      updateVersion: "release-2",
    });
  });
});
