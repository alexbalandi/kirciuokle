import { describe, expect, it, vi } from "vitest";
import worker, { type Env } from "../src/worker/index";

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    ASSETS: {
      fetch: vi.fn(async () => new Response("<!doctype html>", {
        headers: { "content-type": "text/html" },
      })),
    } as unknown as Fetcher,
    DICT: {} as D1Database,
    ...overrides,
  };
}

function r2Object(body: string, key: string): R2ObjectBody {
  return {
    body: new Response(body).body,
    bodyUsed: false,
    checksums: {},
    customMetadata: {},
    etag: "abc",
    httpEtag: '"abc"',
    httpMetadata: {},
    key,
    range: undefined,
    size: body.length,
    uploaded: new Date("2026-07-07T00:00:00Z"),
    version: "test",
    arrayBuffer: async () => new TextEncoder().encode(body).buffer,
    blob: async () => new Blob([body]),
    json: async () => JSON.parse(body) as unknown,
    text: async () => body,
    writeHttpMetadata: () => {},
  } as unknown as R2ObjectBody;
}

describe("worker local model serving", () => {
  it("falls through to assets when MODEL_BUCKET is absent", async () => {
    const env = makeEnv();
    const response = await worker.fetch(
      new Request("https://example.test/local-model/manifest.json"),
      env,
      { waitUntil: vi.fn() } as unknown as ExecutionContext,
    );

    expect(response.headers.get("Cross-Origin-Embedder-Policy")).toBe("require-corp");
    expect(env.ASSETS.fetch).toHaveBeenCalledOnce();
  });

  it("serves local model files from R2 with isolation and immutable cache headers", async () => {
    const get = vi.fn(async (key: string) => r2Object("onnx", key));
    const env = makeEnv({
      MODEL_BUCKET: { get } as unknown as R2Bucket,
    });

    const response = await worker.fetch(
      new Request("https://example.test/local-model/joint.int8.full.onnx"),
      env,
      { waitUntil: vi.fn() } as unknown as ExecutionContext,
    );

    expect(get).toHaveBeenCalledWith("joint.int8.full.onnx");
    expect(await response.text()).toBe("onnx");
    expect(response.headers.get("content-type")).toBe("application/octet-stream");
    expect(response.headers.get("cache-control")).toBe(
      "public, max-age=31536000, immutable",
    );
    expect(response.headers.get("Cross-Origin-Resource-Policy")).toBe("same-origin");
    expect(response.headers.get("Cross-Origin-Embedder-Policy")).toBe("require-corp");
    expect(response.headers.get("Cross-Origin-Opener-Policy")).toBe("same-origin");
  });
});
