import { cloudflare } from "@cloudflare/vite-plugin";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
// Dev-only serving of the gitignored local-model bundle for Local mode
// (src/client/local/assets.ts LOCAL_MODEL_BASE). Generate the bundle with
// `uv run scripts/prepare_local_model.py`. Production/R2 wiring is separate.
const localModelDir = path.resolve(rootDir, "local-model");
const localModelPrefix = "/local-model/";

export default defineConfig({
  plugins: [forceHunspellCjs(), localModelDevServer(), cloudflare()],
  // Bundle hunspell-asm's CJS subtree, not its ESM one — see forceHunspellCjs().
  // Pre-bundle the CJS entry so dev serves it as browser ESM.
  optimizeDeps: {
    include: ["hunspell-asm/dist/cjs/index.js"],
  },
  // The spellcheck web worker is bundled as its own rollup sub-build in `vite
  // build`, which does NOT inherit root resolver plugins — register the CJS
  // forcer here too or the prod worker silently ships the broken ESM build.
  worker: {
    plugins: () => [forceHunspellCjs()],
  },
  server: {
    fs: {
      allow: [rootDir, localModelDir],
    },
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});

// hunspell-asm and its dep emscripten-wasm-loader ship broken ESM builds: they
// use the `import * as X; X()` anti-pattern for CJS deps (nanoid, its emscripten
// runtime), which becomes "calling a namespace object" at runtime under a
// bundler. Their CJS builds use require() and work. Rollup prefers the `module`
// (ESM) field, so force these specific packages to their CJS `main` instead.
// getroot has no `module` field, so it already resolves to CJS.
function forceHunspellCjs() {
  const cjsEntry: Record<string, string> = {
    "hunspell-asm": "node_modules/hunspell-asm/dist/cjs/index.js",
    "emscripten-wasm-loader": "node_modules/emscripten-wasm-loader/dist/cjs/index.js",
  };
  return {
    name: "force-hunspell-cjs",
    enforce: "pre" as const,
    resolveId(source: string) {
      const entry = cjsEntry[source];
      return entry ? path.resolve(rootDir, entry) : null;
    },
  };
}

function localModelDevServer() {
  return {
    name: "local-model-dev-server",
    enforce: "pre" as const,
    configureServer(server: import("vite").ViteDevServer) {
      server.middlewares.use(async (request, response, next) => {
        const requestPath = request.url?.split("?")[0] ?? "";
        if (!requestPath.startsWith(localModelPrefix)) {
          next();
          return;
        }

        const relative = decodeURIComponent(requestPath.slice(localModelPrefix.length));
        const filePath = path.resolve(localModelDir, relative);
        if (!filePath.startsWith(`${localModelDir}${path.sep}`)) {
          response.statusCode = 403;
          response.end("Forbidden");
          return;
        }

        let fileStat;
        try {
          fileStat = await stat(filePath);
        } catch {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }

        if (!fileStat.isFile()) {
          response.statusCode = 404;
          response.end("Not found");
          return;
        }

        response.setHeader("content-length", String(fileStat.size));
        response.setHeader("content-type", contentType(filePath));
        response.setHeader("cache-control", "no-store");
        response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
        response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        response.setHeader("Cross-Origin-Opener-Policy", "same-origin");

        if (request.method === "HEAD") {
          response.end();
          return;
        }

        const stream = createReadStream(filePath);
        stream.on("error", (error) => {
          response.destroy(error);
        });
        stream.pipe(response);
      });
    },
  };
}

function contentType(filePath: string): string {
  switch (path.extname(filePath)) {
    case ".json":
      return "application/json; charset=utf-8";
    case ".js":
    case ".mjs":
      return "text/javascript; charset=utf-8";
    case ".wasm":
      return "application/wasm";
    case ".onnx":
      return "application/octet-stream";
    default:
      return "application/octet-stream";
  }
}
