import type { Part } from "../../shared/types";
import type { TagSlots } from "../../shared/tags";

export type ExecutionMode = "worker" | "main";

export type CacheStatus = "hit" | "miss" | "stored" | "failed" | "unavailable";
export type LocalModelTier = "light" | "heavy";

export type LocalModelUpdateInfo = {
  file: string;
  bytes: number | null;
  version: string | null;
};

export type LocalModelStatus =
  | { type: "idle" }
  | { type: "metadata" }
  | { type: "verify-runtime"; file: string; received: number; total: number }
  | {
      type: "modelInfo";
      tier: LocalModelTier;
      modelFile: string;
      expectedBytes: number | null;
      cacheState: boolean;
      threads: number;
    }
  | { type: "transfer"; cached: boolean; received: number; total: number | null }
  | { type: "session"; bytes: number; mode: ExecutionMode | "fallback" }
  | {
      type: "ready";
      tier: LocalModelTier;
      modelFile: string;
      modelVersion: string | null;
      bytes: number;
      cacheStatus: CacheStatus;
      threads: number;
      executionMode: ExecutionMode;
      updateAvailable: boolean;
      update: LocalModelUpdateInfo | null;
    }
  | { type: "failed"; message: string };

export type LocalRunStatus =
  | { type: "ready" }
  | { type: "running"; sentences: number; batches: number }
  | {
      type: "batch";
      renderedSentences: number;
      sentences: number;
      batch: number;
      batches: number;
      tokensPerSecond: number;
    }
  | {
      type: "done";
      sentences: number;
      inferredTokens: number;
      totalTokens: number;
      tokensPerSecond: number;
      elapsedMs: number;
      batches: number;
    }
  | { type: "memoryLimit" }
  | { type: "error"; message: string };

export type LocalProgress = LocalModelStatus | LocalRunStatus;

export type ManifestRuntimeFile = {
  bytes: number;
  sha256: string;
  package?: string;
  source?: string;
};

export type ModelManifest = {
  created_utc?: string;
  default_model?: string;
  model_bytes?: number;
  version?: string;
  tiers?: Partial<Record<LocalModelTier, string>>;
  models?: Record<
    string,
    {
      bytes?: number;
      default?: boolean;
      sha256?: string;
      tier?: LocalModelTier | string;
      version?: string;
    }
  >;
  runtime?: {
    path?: string;
    files?: Record<string, ManifestRuntimeFile>;
  };
};

export type JointMeta = {
  char_vocab: Record<string, number>;
  int8_onnx?: string;
  labels: string[];
  marks: string[];
  max_chars?: number;
};

export type BridgeSlots = TagSlots & Record<string, string | undefined>;

export type MiVocabEntry = {
  label: string;
  slots?: BridgeSlots;
};

export type LabelBridge = {
  mi_vocab?: MiVocabEntry[];
  model_labels?: Record<string, BridgeSlots>;
};

export type PosRow = {
  label: string;
  probability: number;
};

export type SurfaceToken = {
  text: string;
  start: number;
  end: number;
  isWord: boolean;
  modelText?: string;
  accentableStart?: number;
  accentableEnd?: number;
  numeralFragment?: true;
};

export type PreparedSentence = {
  index: number;
  text: string;
  start: number;
  end: number;
  tokens: SurfaceToken[];
  inputIds: number[] | null;
  firstSubword: number[] | null;
  lastSubword: number[] | null;
  subwordLength: number;
};

export type DecodedToken = {
  accented: string;
  predicted: boolean;
  pos: PosRow[];
  noStress: boolean;
};

export type DecodedSentence = {
  index: number;
  start: number;
  end: number;
  tokens: SurfaceToken[];
  decodedTokens: DecodedToken[];
};

export type LocalRunStats = {
  sentences: number;
  batches: number;
  tokens: number;
  totalTokens: number;
  elapsedMs: number;
  tokensPerSecond: number;
};

export type MemoryStatus = {
  wasmBytes: number;
  wasmMaxBytes: number;
  wasmMemoryCount: number;
  jsHeapBytes: number | null;
  jsHeapLimitBytes: number | null;
  wasmRatio: number | null;
};

export type LocalStats = {
  executionMode: ExecutionMode | null;
  memory: MemoryStatus;
  lastRun: LocalRunStats | null;
  modelFile: string | null;
  modelVersion: string | null;
  modelTier: LocalModelTier | null;
  updateAvailable: boolean;
  update: LocalModelUpdateInfo | null;
  cacheStatus: CacheStatus | null;
  threads: number | null;
};

export type LocalAccentResult = {
  parts: Part[];
  stats: LocalStats;
};

export type Tokenizer = {
  encode(
    text: string,
    options: { add_special_tokens: boolean },
  ): number[] | { data: ArrayLike<number> };
  pad_token_id?: number;
  bos_token_id?: number;
  eos_token_id?: number;
};

export type OrtTensorData =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | BigUint64Array
  | number[];

export type OrtTensor = {
  readonly data: OrtTensorData;
  readonly dims: readonly number[];
  dispose?: () => void | Promise<void>;
};

export type OrtFeeds = Record<string, OrtTensor>;
export type OrtOutputs = Record<string, OrtTensor>;

export type OrtSession = {
  run(feeds: OrtFeeds): Promise<OrtOutputs>;
  release?: () => void;
  dispose?: () => void;
};

export type OrtModule = {
  Tensor: new (
    type: "int64",
    data: BigInt64Array,
    dims: readonly number[],
  ) => OrtTensor;
  InferenceSession: {
    create(
      model: Uint8Array,
      options: {
        executionProviders: readonly ["wasm"];
        graphOptimizationLevel: "all";
      },
    ): Promise<OrtSession>;
  };
  env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
      proxy: boolean;
    };
    versions?: {
      web?: string;
    };
  };
};

export type TransformersModule = {
  env: {
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
  AutoTokenizer: {
    from_pretrained(
      modelPath: string,
      options: { local_files_only: boolean },
    ): Promise<Tokenizer>;
  };
};
