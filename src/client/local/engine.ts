import type { AccentResponse, Part, Variant } from "../../shared/types";
import { parseMi } from "../../shared/tags";
import { buildBatches } from "./batching";
import {
  LOCAL_MODEL_BASE,
  loadModelAssets,
  preferredWasmThreads,
  runtimeBaseUrl,
  type CacheWriteState,
  type ModelStatusSink,
} from "./assets";
import { buildLabelBridgeCache, decodePosRows } from "./bridge";
import { applyStress, decodeStress, wordKey } from "./decode";
import type {
  CacheStatus,
  DecodedSentence,
  DecodedToken,
  ExecutionMode,
  JointMeta,
  LocalAccentResult,
  LocalModelStatus,
  LocalRunStats,
  LocalRunStatus,
  LocalStats,
  MemoryStatus,
  OrtFeeds,
  OrtModule,
  OrtOutputs,
  OrtSession,
  OrtTensor,
  PosRow,
  PreparedSentence,
  SurfaceToken,
  Tokenizer,
  TransformersModule,
} from "./types";

const MAX_SUBWORDS = 128;
const MIN_TOKEN_BUDGET = 256;
const DEFAULT_TOKEN_BUDGET = 2048;
const POS_PROB_CUT = 0.1;
const RESOLVED_PROB = 0.9;
const SESSION_PROXY_WATCHDOG_MS = 45_000;
const WASM_PAGE_BYTES = 64 * 1024;
const WASM_32BIT_CEILING_BYTES = 4 * 1024 * 1024 * 1024;
const WASM_HIGH_WATER_RATIO = 0.75;
const TOKEN_RE =
  /\p{N}+(?:[.,]\p{N}+)?-[\p{L}\p{M}]+|[\p{L}\p{M}\p{N}_]+|[^\p{L}\p{M}\p{N}_\s]/gu;
const HYPHENATED_NUMERAL_RE =
  /^\p{N}+(?:[.,]\p{N}+)?-([\p{L}\p{M}]+)$/u;
const SENTENCE_END_RE = /[.!?…]+(?:["')\]]+)?\s+(?=[A-ZĄČĘĖĮŠŲŪŽ])/gu;

declare global {
  interface Window {
    __localAccentReady?: boolean;
    __localAccentStats?: LocalStats;
  }
}

const wasmMemoryTracker = installWasmMemoryTracker();

export class LocalAccentEngine {
  private lastRun: LocalRunStats | null = null;

  private constructor(
    private readonly ort: OrtModule,
    private readonly session: OrtSession,
    private readonly tokenizer: Tokenizer,
    private readonly meta: JointMeta,
    private readonly labelBridgeCache: ReadonlyMap<string, string>,
    private cacheStatus: CacheStatus,
    private readonly modelFile: string,
    private readonly modelVersion: string | null,
    private readonly executionMode: ExecutionMode,
    private readonly threads: number,
  ) {}

  static async create(onStatus: ModelStatusSink = () => {}): Promise<LocalAccentEngine> {
    const assets = await loadModelAssets(onStatus);
    const [ort, transformers] = await Promise.all([loadOrtRuntime(), loadTransformersRuntime()]);
    const threads = preferredWasmThreads();
    const runtimeUrl = runtimeBaseUrl();
    const modelByteLength = assets.modelBytes.byteLength;

    transformers.env.allowLocalModels = true;
    transformers.env.allowRemoteModels = false;
    transformers.env.localModelPath = "";
    configureTransformersOnnxRuntime(transformers, runtimeUrl);

    ort.env.wasm.wasmPaths = runtimeUrl;
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.proxy = true;

    const [tokenizer, sessionInfo] = await Promise.all([
      transformers.AutoTokenizer.from_pretrained(LOCAL_MODEL_BASE, {
        local_files_only: true,
      }),
      createSessionWithProgressiveProxy(
        ort,
        assets.modelBytes,
        new URL(assets.modelFile, new URL(LOCAL_MODEL_BASE, window.location.href)).href,
        assets.expectedBytes,
        modelByteLength,
        threads,
        onStatus,
      ),
    ]);

    const labelBridgeCache = buildLabelBridgeCache(assets.bridge, assets.meta.labels);
    const engine = new LocalAccentEngine(
      ort,
      sessionInfo.session,
      tokenizer,
      assets.meta,
      labelBridgeCache,
      assets.cacheWriteState?.status ?? assets.cacheStatus,
      assets.modelFile,
      assets.manifest.created_utc ?? null,
      sessionInfo.mode,
      sessionInfo.threads,
    );

    watchCacheStatus(engine, assets.cacheWriteState);

    onStatus({
      type: "ready",
      modelFile: assets.modelFile,
      bytes: modelByteLength,
      cacheStatus: engine.cacheStatus,
      threads: sessionInfo.threads,
      executionMode: sessionInfo.mode,
    });
    window.__localAccentReady = true;
    window.__localAccentStats = engine.getStats();

    return engine;
  }

  async accent(
    text: string,
    onRunStatus: (status: LocalRunStatus) => void = () => {},
  ): Promise<LocalAccentResult> {
    const normalized = text.normalize("NFC");
    const sentences = splitSentences(normalized).map((sentence) =>
      prepareSentence(sentence, this.tokenizer),
    );

    if (!sentences.length) {
      onRunStatus({ type: "ready" });
      return { parts: [], stats: this.getStats() };
    }

    const baseTokenBudget = DEFAULT_TOKEN_BUDGET;
    let effectiveTokenBudget = adaptiveTokenBudget(baseTokenBudget);
    let batches = buildBatches(sentences, effectiveTokenBudget);
    const totalTokens = sentences.reduce((sum, sentence) => sum + sentence.tokens.length, 0);
    const decodedBySentence = new Map<number, DecodedSentence>();
    let renderedSentences = 0;
    let inferredTokens = 0;
    const startMs = performance.now();

    onRunStatus({
      type: "running",
      sentences: sentences.length,
      batches: batches.length,
    });

    try {
      for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
        const batch = batches[batchIndex]!;
        let decoded: DecodedSentence[];
        try {
          decoded = await this.runBatch(batch);
        } catch (error) {
          if (isMemoryAllocationError(error)) {
            console.warn(error);
            this.lastRun = {
              sentences: sentences.length,
              batches: batches.length,
              tokens: inferredTokens,
              totalTokens,
              elapsedMs: performance.now() - startMs,
              tokensPerSecond: 0,
            };
            onRunStatus({ type: "memoryLimit" });
            window.__localAccentStats = this.getStats();
            return { parts: partsFromDecodedSentences(normalized, decodedBySentence), stats: this.getStats() };
          }
          throw error;
        }

        for (const sentenceResult of decoded) {
          decodedBySentence.set(sentenceResult.index, sentenceResult);
          releaseSentenceInferenceInputs(batch.find((sentence) => sentence.index === sentenceResult.index));
          renderedSentences += 1;
          inferredTokens += sentenceResult.decodedTokens.filter((token) => token.predicted).length;
        }

        const elapsed = performance.now() - startMs;
        const tokensPerSecond = inferredTokens / Math.max(elapsed / 1000, 0.001);
        const memory = readMemoryStatus();
        const nextTokenBudget = adaptiveTokenBudget(baseTokenBudget, memory);

        if (nextTokenBudget !== effectiveTokenBudget && batchIndex + 1 < batches.length) {
          const pending = batches.slice(batchIndex + 1).flat();
          const rebuilt = buildBatches(pending, nextTokenBudget);
          batches = batches.slice(0, batchIndex + 1).concat(rebuilt);
          effectiveTokenBudget = nextTokenBudget;
        }

        onRunStatus({
          type: "batch",
          renderedSentences,
          sentences: sentences.length,
          batch: batchIndex + 1,
          batches: batches.length,
          tokensPerSecond,
        });
        await nextFrame();
      }

      const elapsedMs = performance.now() - startMs;
      const tokensPerSecond = inferredTokens / Math.max(elapsedMs / 1000, 0.001);
      this.lastRun = {
        sentences: sentences.length,
        batches: batches.length,
        tokens: inferredTokens,
        totalTokens,
        elapsedMs,
        tokensPerSecond,
      };
      onRunStatus({
        type: "done",
        sentences: sentences.length,
        inferredTokens,
        totalTokens,
        tokensPerSecond,
        elapsedMs,
        batches: batches.length,
      });
      window.__localAccentStats = this.getStats();

      return {
        parts: partsFromDecodedSentences(normalized, decodedBySentence),
        stats: this.getStats(),
      };
    } catch (error) {
      onRunStatus({ type: "error", message: errorMessage(error) });
      throw error;
    }
  }

  getStats(): LocalStats {
    return {
      executionMode: this.executionMode,
      memory: readMemoryStatus(),
      lastRun: this.lastRun,
      modelFile: this.modelFile,
      modelVersion: this.modelVersion,
      cacheStatus: this.cacheStatus,
      threads: this.threads,
    };
  }

  private async runBatch(batch: PreparedSentence[]): Promise<DecodedSentence[]> {
    let feeds: OrtFeeds | null = null;
    let outputs: OrtOutputs | null = null;

    try {
      feeds = makeFeeds(this.ort, batch, this.meta, this.tokenizer);
      outputs = await this.session.run(feeds);
      return decodeBatch(batch, outputs, this.meta, this.labelBridgeCache);
    } finally {
      await disposeOrtValues(outputs);
      await disposeOrtValues(feeds);
    }
  }

  setCacheStatus(status: CacheStatus): void {
    this.cacheStatus = status;
  }
}

export function localAccentResponse(parts: Part[]): AccentResponse {
  return {
    source: "local",
    tagger: "ok",
    parts,
  };
}

export function partsFromDecodedSentences(
  text: string,
  decodedBySentence: ReadonlyMap<number, DecodedSentence>,
): Part[] {
  const sentences = Array.from(decodedBySentence.values()).sort(
    (left, right) => left.start - right.start || left.index - right.index,
  );
  const parts: Part[] = [];
  let cursor = 0;

  for (const sentence of sentences) {
    if (sentence.start > cursor) {
      appendSep(parts, text.slice(cursor, sentence.start));
    }

    for (let index = 0; index < sentence.tokens.length; index += 1) {
      const surface = sentence.tokens[index]!;
      const decoded = sentence.decodedTokens[index];

      if (surface.start > cursor) {
        appendSep(parts, text.slice(cursor, surface.start));
      }

      if (surface.isWord && decoded?.predicted) {
        parts.push(partFromDecodedToken(surface, decoded));
      } else {
        appendSep(parts, text.slice(surface.start, surface.end));
      }

      cursor = surface.end;
    }

    if (sentence.end > cursor) {
      appendSep(parts, text.slice(cursor, sentence.end));
      cursor = sentence.end;
    }
  }

  if (cursor < text.length) {
    appendSep(parts, text.slice(cursor));
  }

  return parts;
}

function partFromDecodedToken(surface: SurfaceToken, decoded: DecodedToken): Part {
  const part: Part = {
    text: surface.text.normalize("NFC"),
    type: "word",
    accented: decoded.accented.normalize("NFC"),
  };

  if (surface.numeralFragment) {
    part.numeralFragment = true;
    return part;
  }

  const variants = localVariants(decoded.accented, decoded.pos);
  const chosenMi = variants[0]?.info;
  const p = variants[0]?.p ?? 0;
  const rowsAboveCut = variants.filter((variant) => (variant.p ?? 0) > POS_PROB_CUT);

  if (variants.length) {
    part.variants = variants;
    part.chosen = 0;
    part.chosenMi = chosenMi;
    if (chosenMi) {
      part.tokenTags = parseMi(chosenMi);
    }
  }

  if (rowsAboveCut.length >= 2) {
    part.ambiguous = true;
  } else if (variants.length === 1 && p >= RESOLVED_PROB) {
    part.resolvedBy = "context";
  }

  if (decoded.noStress && variants.length === 0) {
    part.unknown = true;
  }

  return part;
}

function localVariants(accented: string, rows: readonly PosRow[]): Variant[] {
  return rows.map((row) => ({
    form: accented.normalize("NFC"),
    info: row.label,
    p: row.probability,
  }));
}

function appendSep(parts: Part[], text: string): void {
  if (!text) {
    return;
  }

  const previous = parts[parts.length - 1];
  if (previous?.type === "sep") {
    previous.text += text;
    return;
  }

  parts.push({ text, type: "sep" });
}

type SentenceSlice = {
  index: number;
  text: string;
  start: number;
  end: number;
};

function splitSentences(text: string): SentenceSlice[] {
  const sentences: SentenceSlice[] = [];
  let sentenceIndex = 0;

  for (const paragraphMatch of text.matchAll(/[^\n]+/gu)) {
    const rawParagraph = paragraphMatch[0];
    const paragraphStart = paragraphMatch.index ?? 0;
    const leading = rawParagraph.match(/^\s*/u)?.[0].length ?? 0;
    const trailing = rawParagraph.match(/\s*$/u)?.[0].length ?? 0;
    const paragraph = rawParagraph.slice(leading, rawParagraph.length - trailing);
    const paragraphOffset = paragraphStart + leading;

    if (!paragraph) {
      continue;
    }

    let start = 0;
    SENTENCE_END_RE.lastIndex = 0;
    for (const match of paragraph.matchAll(SENTENCE_END_RE)) {
      const end = (match.index ?? 0) + match[0].length;
      const piece = paragraph.slice(start, end).trim();
      if (piece) {
        const pieceLeading = paragraph.slice(start, end).match(/^\s*/u)?.[0].length ?? 0;
        const pieceTrailing = paragraph.slice(start, end).match(/\s*$/u)?.[0].length ?? 0;
        sentences.push({
          index: sentenceIndex,
          text: piece,
          start: paragraphOffset + start + pieceLeading,
          end: paragraphOffset + end - pieceTrailing,
        });
        sentenceIndex += 1;
      }
      start = end;
    }

    const tailRaw = paragraph.slice(start);
    const tail = tailRaw.trim();
    if (tail) {
      const tailLeading = tailRaw.match(/^\s*/u)?.[0].length ?? 0;
      const tailTrailing = tailRaw.match(/\s*$/u)?.[0].length ?? 0;
      sentences.push({
        index: sentenceIndex,
        text: tail,
        start: paragraphOffset + start + tailLeading,
        end: paragraphOffset + paragraph.length - tailTrailing,
      });
      sentenceIndex += 1;
    }
  }

  return sentences;
}

function prepareSentence(sentence: SentenceSlice, tokenizer: Tokenizer): PreparedSentence {
  const tokens = tokenizeSurface(sentence.text, sentence.start);
  const encoded = encodeSentence(tokenizer, tokens);

  return {
    ...sentence,
    tokens,
    ...encoded,
  };
}

export function tokenizeSurface(text: string, offset: number): SurfaceToken[] {
  return [...text.matchAll(TOKEN_RE)].map((match) => {
    const start = offset + (match.index ?? 0);
    const token = match[0].normalize("NFC");
    const numeralMatch = token.match(HYPHENATED_NUMERAL_RE);
    if (numeralMatch) {
      const suffix = numeralMatch[1]!.normalize("NFC");
      const suffixStart = token.length - suffix.length;
      return {
        text: token,
        start,
        end: start + match[0].length,
        isWord: true,
        modelText: suffix,
        accentableStart: suffixStart,
        accentableEnd: token.length,
        numeralFragment: true,
      };
    }

    return {
      text: token,
      start,
      end: start + match[0].length,
      isWord: /\p{L}/u.test(match[0]),
    };
  });
}

function encodeSentence(
  tokenizer: Tokenizer,
  tokens: readonly SurfaceToken[],
): Pick<PreparedSentence, "inputIds" | "firstSubword" | "lastSubword" | "subwordLength"> {
  const inputIds = [Number(tokenizer.bos_token_id ?? 0)];
  const firstSubword = Array<number>(tokens.length).fill(-1);
  const lastSubword = Array<number>(tokens.length).fill(-1);

  for (let wordIndex = 0; wordIndex < tokens.length; wordIndex += 1) {
    const token = tokens[wordIndex]!;
    const tokenIds = normalizeIds(
      tokenizer.encode(` ${modelTokenText(token)}`, { add_special_tokens: false }),
    );

    if (!tokenIds.length) {
      continue;
    }

    if (inputIds.length + tokenIds.length + 1 > MAX_SUBWORDS) {
      continue;
    }

    firstSubword[wordIndex] = inputIds.length;
    inputIds.push(...tokenIds);
    lastSubword[wordIndex] = inputIds.length - 1;
  }

  inputIds.push(Number(tokenizer.eos_token_id ?? 2));

  return {
    inputIds,
    firstSubword,
    lastSubword,
    subwordLength: inputIds.length,
  };
}

function normalizeIds(ids: number[] | { data: ArrayLike<number> }): number[] {
  if (Array.isArray(ids)) {
    return ids.map(Number);
  }

  return Array.from(ids.data, Number);
}

function makeFeeds(
  ort: OrtModule,
  batch: readonly PreparedSentence[],
  meta: JointMeta,
  tokenizer: Tokenizer,
): OrtFeeds {
  const batchSize = batch.length;
  const subwords = Math.max(...batch.map((sentence) => sentence.inputIds?.length ?? 0));
  const words = Math.max(...batch.map((sentence) => sentence.tokens.length), 1);
  const maxChars = Number(meta.max_chars || 30);
  const inputIds = Array<number>(batchSize * subwords).fill(Number(tokenizer.pad_token_id ?? 1));
  const attentionMask = Array<number>(batchSize * subwords).fill(0);
  const firstSubword = Array<number>(batchSize * words).fill(-1);
  const lastSubword = Array<number>(batchSize * words).fill(-1);
  const charIds = Array<number>(batchSize * words * maxChars).fill(0);

  for (let row = 0; row < batchSize; row += 1) {
    const sentence = batch[row]!;
    const sentenceInputIds = sentence.inputIds ?? [];
    const sentenceFirstSubword = sentence.firstSubword ?? [];
    const sentenceLastSubword = sentence.lastSubword ?? [];

    for (let col = 0; col < sentenceInputIds.length; col += 1) {
      const offset = row * subwords + col;
      inputIds[offset] = sentenceInputIds[col]!;
      attentionMask[offset] = 1;
    }

    for (let word = 0; word < sentence.tokens.length; word += 1) {
      firstSubword[row * words + word] = sentenceFirstSubword[word] ?? -1;
      lastSubword[row * words + word] = sentenceLastSubword[word] ?? -1;
      const keyChars = Array.from(wordKey(modelTokenText(sentence.tokens[word]!)));

      for (let char = 0; char < Math.min(keyChars.length, maxChars); char += 1) {
        charIds[(row * words + word) * maxChars + char] =
          meta.char_vocab[keyChars[char]!] ?? 1;
      }
    }
  }

  return {
    input_ids: int64Tensor(ort, inputIds, [batchSize, subwords]),
    attention_mask: int64Tensor(ort, attentionMask, [batchSize, subwords]),
    first_subword: int64Tensor(ort, firstSubword, [batchSize, words]),
    last_subword: int64Tensor(ort, lastSubword, [batchSize, words]),
    char_ids: int64Tensor(ort, charIds, [batchSize, words, maxChars]),
  };
}

function modelTokenText(token: SurfaceToken): string {
  return token.modelText ?? token.text;
}

function decodeBatch(
  batch: readonly PreparedSentence[],
  outputs: OrtOutputs,
  meta: JointMeta,
  labelBridgeCache: ReadonlyMap<string, string>,
): DecodedSentence[] {
  const posLogits = requireOutput(outputs, "pos_logits");
  const stressLogits = requireOutput(outputs, "stress_logits");
  const noStressLogits = requireOutput(outputs, "no_stress_logits");
  const posData = numberTensorData(posLogits);
  const stressData = numberTensorData(stressLogits);
  const noStressData = numberTensorData(noStressLogits);
  const labelCount = meta.labels.length;
  const markCount = meta.marks.length;
  const maxChars = Number(meta.max_chars || 30);
  const maxWords = Number(posLogits.dims[1] ?? 0);

  return batch.map((sentence, row) => ({
    index: sentence.index,
    start: sentence.start,
    end: sentence.end,
    tokens: sentence.tokens,
    decodedTokens: sentence.tokens.map((surface, word) => {
      const predicted = (sentence.firstSubword?.[word] ?? -1) >= 0 && word < maxWords;

      if (!predicted) {
        return {
          accented: surface.text,
          predicted: false,
          pos: [],
          noStress: false,
        };
      }

      const posOffset = (row * maxWords + word) * labelCount;
      const pos = decodePosRows(posData, posOffset, meta.labels, labelBridgeCache);
      const stressOffset = (row * maxWords + word) * maxChars * markCount;
      const noStressOffset = row * maxWords + word;
      const stressText = modelTokenText(surface);
      const stress = decodeStress(
        stressData,
        noStressData[noStressOffset] ?? -Infinity,
        stressOffset,
        stressText,
        meta.marks,
        maxChars,
      );
      const accentedModelText = stress.noStress
        ? stressText
        : applyStress(stressText, stress.pos, stress.mark);

      return {
        accented: applyModelAccentToSurface(surface, accentedModelText),
        predicted: true,
        pos,
        noStress: stress.noStress,
      };
    }),
  }));
}

function applyModelAccentToSurface(surface: SurfaceToken, accentedModelText: string): string {
  if (
    typeof surface.accentableStart === "number" &&
    typeof surface.accentableEnd === "number"
  ) {
    return `${surface.text.slice(0, surface.accentableStart)}${accentedModelText}${surface.text.slice(
      surface.accentableEnd,
    )}`;
  }

  return accentedModelText;
}

function requireOutput(outputs: OrtOutputs, name: string): OrtTensor {
  const tensor = outputs[name];
  if (!tensor) {
    throw new Error(`Missing ONNX output ${name}.`);
  }

  return tensor;
}

function numberTensorData(tensor: OrtTensor): ArrayLike<number> {
  if (tensor.data instanceof BigInt64Array || tensor.data instanceof BigUint64Array) {
    throw new Error("Expected numeric tensor data, received bigint tensor.");
  }

  return tensor.data;
}

async function createSessionWithProgressiveProxy(
  ort: OrtModule,
  modelBytes: Uint8Array,
  modelUrl: string,
  expectedBytes: number | null,
  modelByteLength: number,
  threads: number,
  onStatus: (status: LocalModelStatus) => void,
): Promise<{ session: OrtSession; mode: ExecutionMode; threads: number }> {
  let currentModelBytes = modelBytes;

  onStatus({ type: "session", bytes: modelByteLength, mode: "worker" });
  try {
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = threads;
    const session = await withTimeout(
      ort.InferenceSession.create(currentModelBytes, makeSessionOptions()),
      SESSION_PROXY_WATCHDOG_MS,
      "ORT proxy worker session create timed out.",
    );
    return { session, mode: "worker", threads };
  } catch (error) {
    console.warn("ORT proxy worker session failed; retrying on the main thread.", error);
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    onStatus({ type: "session", bytes: modelByteLength, mode: "fallback" });

    if (currentModelBytes.byteLength !== modelByteLength) {
      currentModelBytes = await refetchModelBytesForFallback(modelUrl, expectedBytes);
    }

    const session = await ort.InferenceSession.create(currentModelBytes, makeSessionOptions());
    return { session, mode: "main", threads: 1 };
  }
}

function makeSessionOptions(): {
  executionProviders: readonly ["wasm"];
  graphOptimizationLevel: "all";
} {
  return {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  };
}

async function refetchModelBytesForFallback(
  url: string,
  expectedBytes: number | null,
): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  if (expectedBytes && bytes.byteLength !== expectedBytes) {
    throw new Error(
      `Fallback model refetch returned ${bytes.byteLength} bytes, expected ${expectedBytes}.`,
    );
  }

  return bytes;
}

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  let timeoutId = 0;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = window.setTimeout(() => reject(new Error(message)), ms);
  });

  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timeoutId));
}

function configureTransformersOnnxRuntime(
  transformers: TransformersModule,
  wasmPaths: string,
): void {
  const onnx = transformers.env.backends?.onnx;
  if (!onnx?.wasm) {
    return;
  }

  onnx.wasm.wasmPaths = wasmPaths;
  onnx.wasm.proxy = true;
}

async function disposeOrtValues(values: OrtOutputs | OrtFeeds | null): Promise<void> {
  if (!values) {
    return;
  }

  for (const value of Object.values(values) as OrtTensor[]) {
    const dispose = value.dispose;
    if (typeof dispose !== "function") {
      continue;
    }

    try {
      await dispose.call(value);
    } catch (error) {
      console.debug("ORT value dispose failed; continuing.", error);
    }
  }
}

function int64Tensor(
  ort: OrtModule,
  values: readonly number[],
  dims: readonly number[],
): OrtTensor {
  const data = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    data[i] = BigInt(values[i]!);
  }

  return new ort.Tensor("int64", data, dims);
}

function adaptiveTokenBudget(
  tokenBudget: number,
  memory: MemoryStatus = readMemoryStatus(),
): number {
  if (
    memory.wasmMemoryCount > 0 &&
    memory.wasmRatio !== null &&
    memory.wasmRatio > WASM_HIGH_WATER_RATIO
  ) {
    return Math.max(MIN_TOKEN_BUDGET, Math.floor(tokenBudget / 2));
  }

  return tokenBudget;
}

function isMemoryAllocationError(error: unknown): boolean {
  const message = errorMessage(error).toLowerCase();
  return (
    message.includes("out of memory") ||
    message.includes("allocation failed") ||
    message.includes("failed to allocate") ||
    message.includes("could not allocate") ||
    message.includes("cannot enlarge memory") ||
    message.includes("array buffer allocation") ||
    (message.includes("wasm") && message.includes("memory"))
  );
}

function releaseSentenceInferenceInputs(sentence: PreparedSentence | undefined): void {
  if (!sentence) {
    return;
  }

  sentence.inputIds = null;
  sentence.firstSubword = null;
  sentence.lastSubword = null;
  sentence.subwordLength = 0;
}

function watchCacheStatus(
  engine: LocalAccentEngine,
  state: CacheWriteState | null,
): void {
  if (!state?.start) {
    return;
  }

  state.start().then((status) => {
    engine.setCacheStatus(status);
    window.__localAccentStats = engine.getStats();
  });
}

async function loadOrtRuntime(): Promise<OrtModule> {
  const url = "/local-model/runtime/ort.min.mjs";
  return (await import(/* @vite-ignore */ url)) as unknown as OrtModule;
}

async function loadTransformersRuntime(): Promise<TransformersModule> {
  const url = "/local-model/runtime/transformers.min.js";
  return (await import(/* @vite-ignore */ url)) as unknown as TransformersModule;
}

function installWasmMemoryTracker(): { read: () => Omit<MemoryStatus, "jsHeapBytes" | "jsHeapLimitBytes" | "wasmRatio"> } {
  const memories = new Map<WebAssembly.Memory, { maxBytes?: number }>();
  const wasm = WebAssembly as typeof WebAssembly & {
    Memory: typeof WebAssembly.Memory;
  };
  const alreadyInstalled = Reflect.get(wasm.Memory, "__localAccentTracked");
  if (alreadyInstalled) {
    return {
      read() {
        return {
          wasmBytes: 0,
          wasmMaxBytes: WASM_32BIT_CEILING_BYTES,
          wasmMemoryCount: 0,
        };
      },
    };
  }

  const NativeMemory = wasm.Memory;
  const remember = (memory: WebAssembly.Memory, maximumPages: number | null = null) => {
    const current = memories.get(memory) ?? {};
    const maxBytes =
      Number.isFinite(maximumPages) && maximumPages !== null
        ? Number(maximumPages) * WASM_PAGE_BYTES
        : current.maxBytes;
    memories.set(memory, { maxBytes });
  };
  const InstrumentedMemory = new Proxy(NativeMemory, {
    construct(target, args: [WebAssembly.MemoryDescriptor], newTarget) {
      const descriptor = args[0];
      const memory = Reflect.construct(target, args, newTarget) as WebAssembly.Memory;
      remember(memory, descriptor.maximum ?? null);
      return memory;
    },
  });

  Reflect.set(InstrumentedMemory, "__localAccentTracked", true);
  wasm.Memory = InstrumentedMemory;

  return {
    read() {
      const entries = Array.from(memories, ([memory, info]) => {
        const bytes = memory.buffer.byteLength;
        return {
          bytes,
          maxBytes:
            info.maxBytes && info.maxBytes >= bytes
              ? info.maxBytes
              : WASM_32BIT_CEILING_BYTES,
        };
      });
      const wasmBytes = entries.reduce((sum, item) => sum + item.bytes, 0);
      const wasmMaxBytes = entries.reduce((sum, item) => sum + item.maxBytes, 0);

      return {
        wasmBytes,
        wasmMaxBytes: wasmMaxBytes || WASM_32BIT_CEILING_BYTES,
        wasmMemoryCount: entries.length,
      };
    },
  };
}

function readMemoryStatus(): MemoryStatus {
  const wasm = wasmMemoryTracker.read();
  const performanceWithMemory = performance as Performance & {
    memory?: {
      usedJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  };
  const js = performanceWithMemory.memory
    ? {
        jsHeapBytes: performanceWithMemory.memory.usedJSHeapSize,
        jsHeapLimitBytes: performanceWithMemory.memory.jsHeapSizeLimit,
      }
    : { jsHeapBytes: null, jsHeapLimitBytes: null };

  return {
    ...wasm,
    ...js,
    wasmRatio: wasm.wasmMaxBytes > 0 ? wasm.wasmBytes / wasm.wasmMaxBytes : null,
  };
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
