import { describe, expect, it } from "vitest";
import { resolveModelTierInfo } from "../src/client/local/assets";
import type { ModelManifest } from "../src/client/local/types";

describe("local model asset manifest", () => {
  it("resolves light and heavy tiers to distinct files and byte sizes", () => {
    const manifest: ModelManifest = {
      default_model: "joint.int8.partial.onnx",
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
    });
    expect(resolveModelTierInfo(manifest, "heavy")).toEqual({
      tier: "heavy",
      modelFile: "joint.int8.partial.onnx",
      bytes: 470_223_894,
    });
  });
});
