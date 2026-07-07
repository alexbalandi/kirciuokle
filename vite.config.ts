import { cloudflare } from "@cloudflare/vite-plugin";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const localModelDir = path.resolve(rootDir, "bundled_weights_pilot", "model-pruned");
// Dev-only serving for src/client/local/assets.ts LOCAL_MODEL_BASE.
// Production/R2 wiring intentionally stays out of this spec.
const localModelPrefix = "/local-model/";

export default defineConfig({
  plugins: [localModelDevServer(), cloudflare()],
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
