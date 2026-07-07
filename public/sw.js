// Service worker for the installable app — makes the app shell (HTML + hashed
// JS/CSS + icons) load offline. It is a browser service worker, NOT a Cloudflare
// Worker. Deliberately conservative:
//   * It NEVER rewrites responses — it caches and returns them verbatim, so the
//     COOP/COEP headers that give the page cross-origin isolation (required for the
//     threaded ONNX runtime / SharedArrayBuffer) are preserved.
//   * It bypasses the big self-managed assets entirely (the local model, the ORT
//     wasm, the spellcheck dictionaries, /api). Those have their own Cache API
//     storage — double-caching them here would waste quota and cause eviction.
const CACHE = "app-shell-v1";
const PRECACHE = [
  "/",
  "/manifest.webmanifest",
  "/favicon-32.png",
  "/apple-touch-icon.png",
  "/icon-192.png",
  "/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await cache.addAll(PRECACHE).catch(() => {});
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith("app-shell-") && key !== CACHE)
          .map((key) => caches.delete(key)),
      );
      await self.clients.claim();
    })(),
  );
});

// Large, self-managed, or dynamic responses the shell cache must not touch.
function shouldBypass(url) {
  const p = url.pathname;
  return (
    p.startsWith("/api/") ||
    p.startsWith("/local-model/") ||
    p === "/spellcheck-lt.txt" ||
    p === "/spellcheck-bigrams.txt" ||
    p === "/lt.dic" ||
    p === "/lt.aff" ||
    p.endsWith(".onnx") ||
    p.endsWith(".wasm")
  );
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || shouldBypass(url)) {
    return; // let the browser (and the app's own caches) handle it
  }

  // Navigations: network-first so updates land immediately, cached shell offline.
  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          const response = await fetch(request);
          const cache = await caches.open(CACHE);
          cache.put(request, response.clone());
          return response;
        } catch {
          const cache = await caches.open(CACHE);
          return (
            (await cache.match(request)) ||
            (await cache.match("/")) ||
            Response.error()
          );
        }
      })(),
    );
    return;
  }

  // Hashed static assets: cache-first, populate on first fetch. Only same-origin
  // "basic" responses are cached, so headers (COEP/CORP) are always preserved.
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      const hit = await cache.match(request);
      if (hit) {
        return hit;
      }
      try {
        const response = await fetch(request);
        if (response.ok && response.type === "basic") {
          cache.put(request, response.clone());
        }
        return response;
      } catch {
        return hit || Response.error();
      }
    })(),
  );
});
