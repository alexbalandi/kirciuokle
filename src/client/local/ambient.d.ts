declare module "/local-model/runtime/ort.min.mjs" {
  export type TensorData =
    | Float32Array
    | Float64Array
    | Int32Array
    | BigInt64Array
    | BigUint64Array
    | number[];

  export class Tensor<TData extends TensorData = TensorData> {
    constructor(type: "int64", data: BigInt64Array, dims: readonly number[]);
    readonly data: TData;
    readonly dims: readonly number[];
    dispose?: () => void | Promise<void>;
  }

  export type SessionFeeds = Record<string, Tensor>;
  export type SessionOutputs = Record<string, Tensor>;

  export type InferenceSession = {
    run(feeds: SessionFeeds): Promise<SessionOutputs>;
    release?: () => void;
    dispose?: () => void;
  };

  export const InferenceSession: {
    create(
      model: Uint8Array,
      options: {
        executionProviders: readonly ["wasm"];
        graphOptimizationLevel: "all";
      },
    ): Promise<InferenceSession>;
  };

  export const env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
      proxy: boolean;
    };
    versions?: {
      web?: string;
    };
  };
}

declare module "/local-model/runtime/transformers.min.js" {
  import type { Tokenizer } from "./types";

  export const env: {
    allowLocalModels: boolean;
    allowRemoteModels: boolean;
    localModelPath: string;
    backends?: {
      onnx?: {
        wasm?: {
          wasmPaths: string;
          proxy: boolean;
        };
      };
    };
  };

  export const AutoTokenizer: {
    from_pretrained(
      modelPath: string,
      options: { local_files_only: boolean },
    ): Promise<Tokenizer>;
  };
}
