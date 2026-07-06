# Browser Feasibility for the No-Dictionary ONNX Pipeline

Measured from this checkout during the SPEC28 implementation.

## Artifact Sizes

| artifact | bytes | MiB | note |
| --- | ---: | ---: | --- |
| tagger VDU INT8 ONNX | 151,556,354 | 144.54 | `local/tagger-hf/artifacts/litlat-gen2-onnx/int8/model_quantized.onnx` |
| tagger VDU runtime dir | 159,155,445 | 151.78 | model + tokenizer + config/head/labels metadata |
| tagger UD INT8 ONNX | 151,398,866 | 144.39 | measured alternate artifact |
| tagger v2 INT8 ONNX | 155,511,997 | 148.31 | measured alternate artifact |
| shared LitLat tokenizer | 7,275,706 | 6.94 | `tokenizer.json`; same size in artifacts and release dirs |
| release `hf-vdu` runtime files | 7,608,797 | 7.26 | no ONNX model in release dir |
| stress FP32 ONNX | 619,705,188 | 591.00 | `stress.onnx` |
| stress INT8 ONNX | 534,785,129 | 510.01 | parity-stable partial dynamic INT8: first 4 encoder layers |
| stress metadata | 1,770 | 0.00 | `stress.meta.json` |

Total download for the shipped two-model no-dict browser bundle, counting the shared tokenizer once: **693,942,344 bytes / 661.79 MiB**. With FP32 stress instead of the parity-stable INT8 stress artifact: **778,862,403 bytes / 742.78 MiB**.

Important caveat: full dynamic quantization of every supported stress-model node produced a much smaller file during tuning, but failed the 30-sentence agreement gate. The shipped `stress.int8.onnx` therefore records a partial dynamic-quantization scope in metadata.

## LitLat BERT Parameter Breakdown

Measured from `local/accentuator/data/stress_nn2/stress_nn2.pt`.

| component | parameters | share of encoder |
| --- | ---: | ---: |
| word embedding matrix, 84,201 x 768 | 64,666,368 | 42.91% |
| full embedding block | 65,063,424 | 43.17% |
| transformer body excluding embedding block | 85,645,056 | 56.83% |
| total LitLat encoder | 150,708,480 | 100.00% |
| stress head | 4,780,035 | outside encoder |

Vocabulary-prune measurement:

| measurement | value |
| --- | ---: |
| dictionary words scanned | 574,138 |
| LRT corpus word tokens scanned | 37,727 |
| tokenizer vocab entries | 84,198 |
| checkpoint embedding rows | 84,201 |
| non-special rows used by dictionary + LRT | 16,809 |
| rows kept including specials | 16,814 |
| prunable rows | 67,387 / 80.03% |
| prunable embedding parameters | 51,753,216 |
| FP32 embedding bytes saved | 197.42 MiB |
| INT8 embedding bytes saved | 49.36 MiB |

The prune helps, but it does not rescue the current two-model design: even a perfect Lithuanian-only vocab prune saves about 49 MiB from an INT8 embedding matrix, while the measured two-model bundle is about 662 MiB.

## Runtime Paths

ONNX Runtime Web supports a default WebAssembly path and a WebGPU execution provider. The official docs say WASM is the default/lightweight path, while WebGPU is intended for more compute-intensive models and client GPU use. The same docs note current Chrome/Edge availability for WebGPU and browser support caveats. [ORT WebGPU docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

For WASM, ORT Web has SIMD and multi-threaded builds; its deployment docs list `ort-wasm-simd-threaded.wasm` as the standard artifact, and the env docs say browser thread count defaults to half of `navigator.hardwareConcurrency` capped at 4 when threading is available and cross-origin isolation permits it. [ORT deploy docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html), [ORT env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)

Measured here, native desktop ONNX Runtime CPU with the two-model INT8 pipeline runs:

```text
500 tokens in 10.825s = 46.2 tokens/s
```

That is not a browser number. For browser WASM SIMD+threads, expect desktop-class throughput in the **low tens of tokens/s** for this two-BERT pipeline, with large variance from CPU, isolation headers, and thermal throttling. Mobile should be treated as **single-digit to low-tens tokens/s** plus long startup.

For WebGPU, the official Microsoft ORT WebGPU launch post reports large gains on compute-heavy models, including a 19x encoder speedup for Segment Anything on RTX 3060-class hardware, and frames WebGPU as the path when CPU browser inference falls short. That is not a BERT-base benchmark, so for this project WebGPU throughput is an estimate until measured in Chrome with our exact graphs. [Microsoft ORT WebGPU blog](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)

## Memory Ceiling

ORT Web large-model docs call out a 2 GiB protobuf limit for single ONNX files and a 4 GiB WebAssembly memory ceiling. They also recommend browser-side caching via Cache API or Origin Private File System for large models. [ORT large-model docs](https://onnxruntime.ai/docs/tutorials/web/large-models.html)

Engineering budget for WASM peak RAM should be **2-4x model bytes**: fetched bytes/cache or ArrayBuffers, ORT/WASM heap copies of initializers, arena/workspace allocations, and activations. For the measured 661.79 MiB INT8 bundle, that implies roughly **1.3-2.6 GiB peak browser memory** before app UI overhead. This fits high-memory desktops, is risky on laptops with many tabs, and is not a good mobile default.

## Verdict

"Load weights into the user's session and let Chrome do the job" is **not viable as the default product path today** for the current two-model no-dict design. It is viable only as an opt-in desktop demo for users who accept a roughly **662 MiB cached download**, a slow first load, and high RAM use.

The two-model design is the core blocker. It ships a tagger encoder and a second stress encoder, so the download and memory budget is roughly doubled. A future shared-encoder single model is the right browser architecture: one LitLat/modern encoder, two heads, one tokenizer, one ORT session. Pair that with vocabulary pruning and a browser-measured quantization recipe.

Ship-shape recommendation:

1. Build a desktop-only WASM demo behind explicit copy like "download ~700 MB once, cached locally." Use Cache API/OPFS and show progress.
2. Keep server-side inference as the default path.
3. Prioritize the shared-encoder joint model before serious client-side launch.
4. After the joint model exists, remeasure WASM SIMD+threads and WebGPU in Chrome/Edge on desktop and Android, then decide whether mobile deserves support.
