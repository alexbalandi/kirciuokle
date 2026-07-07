const { chromium } = require("playwright");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8765/";
const rootDir = path.resolve(__dirname, "..", "..");
const sampleText = fs.readFileSync(
  path.join(rootDir, "docs", "SPEC49-sample.txt"),
  "utf8",
);
const screenshotPaths = [
  path.join(rootDir, "docs-popover-1.png"),
  path.join(rootDir, "docs-popover-2.png"),
  path.join(rootDir, "docs-popover-3.png"),
];

const TIMEOUTS = {
  page: 30_000,
  webAccent: 120_000,
  localReady: 600_000,
  localAccent: 240_000,
  ui: 15_000,
  wordInfo: 12_000,
};

const diagnostics = {
  console: [],
  pageErrors: [],
  failedRequests: [],
};

function assert(condition, message) {
  if (!condition) {
    throw new Error(`${message}\nDiagnostics:\n${JSON.stringify(diagnostics, null, 2)}`);
  }
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--js-flags=--expose-gc"],
  });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  attachDiagnostics(page);

  try {
    await page.addInitScript(() => {
      localStorage.setItem("lang", "ru");
      localStorage.setItem("accent-mode", "web");
      localStorage.setItem("accent-display", "all");
    });
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });

    await runWebSample(page);
    await captureAmbiguousRu(page);
    await capturePlainLt(page);

    const localServer = await startLocalStaticServer();
    try {
      const localPage = await browser.newPage({ viewport: { width: 1366, height: 900 } });
      attachDiagnostics(localPage);
      await localPage.addInitScript(() => {
        localStorage.setItem("lang", "ru");
        localStorage.setItem("accent-mode", "web");
        localStorage.setItem("accent-display", "all");
      });
      await localPage.goto(localServer.url, {
        waitUntil: "domcontentloaded",
        timeout: TIMEOUTS.page,
      });
      await captureLocalProbabilityRu(localPage);
      await localPage.close();
    } finally {
      await localServer.close();
    }
  } finally {
    await browser.close();
  }
}

function attachDiagnostics(page) {
  page.on("console", (message) => {
    diagnostics.console.push({
      type: message.type(),
      text: message.text().slice(0, 500),
    });
  });
  page.on("pageerror", (error) => {
    diagnostics.pageErrors.push(error.message);
  });
  page.on("requestfailed", (request) => {
    diagnostics.failedRequests.push({
      url: request.url(),
      failure: request.failure()?.errorText ?? "unknown",
    });
  });
}

async function runWebSample(page) {
  await page.fill("#source-text", sampleText, { timeout: TIMEOUTS.ui });
  await page.click("#accent-button", { timeout: TIMEOUTS.ui });
  await page.waitForFunction(
    () => !document.querySelector("#accent-button")?.disabled,
    null,
    { timeout: TIMEOUTS.webAccent },
  );
  await assertResultReady(page, "Web");
}

async function captureAmbiguousRu(page) {
  await page.click('[data-lang="ru"]', { timeout: TIMEOUTS.ui });
  await page.keyboard.press("Escape");

  const token = await openTokenWithPopover(
    page,
    page.locator("#result-output .token-ambiguous"),
    ".variant-popover .variant-headword",
    "No ambiguous token opened a SPEC49 popover.",
  );
  await assertPopoverGeometry(page, token);
  await assertFlatRows(page, { expectGloss: true, expectProbability: false });
  await page.screenshot({ path: screenshotPaths[0] });
}

async function capturePlainLt(page) {
  await page.click('[data-lang="lt"]', { timeout: TIMEOUTS.ui });
  await page.keyboard.press("Escape");

  const token = await openTokenWithPopover(
    page,
    page.locator("#result-output .token-plain"),
    ".variant-popover .variant-headword",
    "No plain token opened a fetched-word SPEC49 popover.",
  );
  await assertPopoverGeometry(page, token);
  await assertFlatRows(page, { expectGloss: false, expectProbability: false });
  await page.screenshot({ path: screenshotPaths[1] });
}

async function captureLocalProbabilityRu(page) {
  await page.evaluate(async () => {
    for (const key of await caches.keys()) {
      await caches.delete(key);
    }
    localStorage.setItem("lang", "ru");
    localStorage.setItem("accent-mode", "web");
  });
  await page.click('[data-lang="ru"]', { timeout: TIMEOUTS.ui });
  await page.click('[data-mode="local"]', { timeout: TIMEOUTS.ui });
  await page.waitForSelector("#local-status:not([hidden])", { timeout: TIMEOUTS.ui });
  await page.waitForFunction(
    () =>
      window.__localAccentReady === true ||
      Boolean(document.querySelector(".local-consent-button")),
    null,
    { timeout: TIMEOUTS.ui },
  );
  if ((await page.locator(".local-consent-button").count()) > 0) {
    await page.click(".local-consent-button", { timeout: TIMEOUTS.ui });
  }
  await page.waitForFunction(() => window.__localAccentReady === true, null, {
    timeout: TIMEOUTS.localReady,
  });

  await page.fill("#source-text", sampleText, { timeout: TIMEOUTS.ui });
  await page.click("#accent-button", { timeout: TIMEOUTS.ui });
  await page.waitForFunction(
    () => Boolean(window.__localAccentStats?.lastRun?.tokens),
    null,
    { timeout: TIMEOUTS.localAccent },
  );
  await assertResultReady(page, "Local");
  await page.keyboard.press("Escape");

  const token = await openTokenWithPopover(
    page,
    page.locator("#result-output button.token"),
    ".variant-popover .variant-probability",
    "No Local token opened a probability popover.",
  );
  await assertPopoverGeometry(page, token);
  await assertFlatRows(page, { expectGloss: true, expectProbability: true });
  await page.screenshot({ path: screenshotPaths[2] });
}

async function assertResultReady(page, mode) {
  const message = await page.locator("#form-message").textContent();
  assert(!message, `${mode} mode reported an error: ${message}`);

  const resultText = (await page.locator("#result-output").textContent()) ?? "";
  assert(resultText.length > 80, `${mode} mode did not render the sample output.`);
}

async function openTokenWithPopover(page, tokens, waitSelector, failureMessage) {
  const count = await tokens.count();
  assert(count > 0, failureMessage);

  for (let index = 0; index < Math.min(count, 18); index += 1) {
    const token = tokens.nth(index);
    if (!(await token.isVisible())) {
      continue;
    }

    await page.keyboard.press("Escape");
    await token.scrollIntoViewIfNeeded();
    await token.click({ timeout: TIMEOUTS.ui });
    try {
      await page.waitForSelector(waitSelector, { timeout: TIMEOUTS.wordInfo });
      return token;
    } catch {
      await page.keyboard.press("Escape");
    }
  }

  throw new Error(`${failureMessage}\nDiagnostics:\n${JSON.stringify(diagnostics, null, 2)}`);
}

async function assertPopoverGeometry(page, token) {
  const tokenBox = await token.boundingBox();
  const popoverBox = await page.locator(".variant-popover").boundingBox();
  const viewport = page.viewportSize();
  assert(tokenBox && popoverBox && viewport, "Could not read popover geometry.");

  const wordCenter = tokenBox.x + tokenBox.width / 2;
  const popoverCenter = popoverBox.x + popoverBox.width / 2;
  const clampedLeft = popoverBox.x <= 9;
  const clampedRight = popoverBox.x + popoverBox.width >= viewport.width - 9;

  if (!clampedLeft && !clampedRight) {
    assert(
      Math.abs(popoverCenter - wordCenter) <= 4,
      `Popover center drifted from word center by ${Math.abs(
        popoverCenter - wordCenter,
      ).toFixed(2)}px.`,
    );
  }

  const caretLeft = await page.locator(".variant-popover").evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).getPropertyValue("--popover-caret-left")),
  );
  assert(Number.isFinite(caretLeft), "Popover caret position was not exposed.");
  assert(
    Math.abs(popoverBox.x + caretLeft - wordCenter) <= 4,
    `Caret drifted from word center by ${Math.abs(
      popoverBox.x + caretLeft - wordCenter,
    ).toFixed(2)}px.`,
  );
  assert(
    Math.abs(popoverBox.width - 320) <= 1,
    `Popover width was ${popoverBox.width}px instead of fixed 320px.`,
  );
}

async function startLocalStaticServer() {
  const clientDir = path.join(rootDir, "dist", "client");
  const modelDir = path.join(rootDir, "bundled_weights_pilot", "model");
  assert(fs.existsSync(path.join(clientDir, "index.html")), "Run build before SPEC49.");

  const server = http.createServer((request, response) => {
    void serveStaticRequest(request, response, clientDir, modelDir);
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });

  const address = server.address();
  assert(address && typeof address === "object", "Could not bind local static server.");
  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

async function serveStaticRequest(request, response, clientDir, modelDir) {
  setIsolationHeaders(response);
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  const pathname = decodeURIComponent(url.pathname);
  const isModel = pathname.startsWith("/local-model/");
  const baseDir = isModel ? modelDir : clientDir;
  const relative = isModel
    ? pathname.slice("/local-model/".length)
    : pathname === "/"
      ? "index.html"
      : pathname.slice(1);
  let filePath = path.resolve(baseDir, relative);

  if (!filePath.startsWith(`${baseDir}${path.sep}`) && filePath !== baseDir) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  if (!isModel && !fs.existsSync(filePath)) {
    filePath = path.join(clientDir, "index.html");
  }

  if (isModel && relative === "manifest.json") {
    try {
      const manifest = JSON.parse(await fs.promises.readFile(filePath, "utf8"));
      manifest.default_model = "joint.full-int8.onnx";
      if (manifest.models?.["joint.full-int8.onnx"]) {
        manifest.models["joint.full-int8.onnx"].default = true;
      }
      if (manifest.models?.["joint.int8.onnx"]) {
        manifest.models["joint.int8.onnx"].default = false;
      }
      const body = Buffer.from(JSON.stringify(manifest));
      response.setHeader("content-length", String(body.length));
      response.setHeader("content-type", "application/json; charset=utf-8");
      response.setHeader("cache-control", "no-store");
      response.writeHead(200);
      if (request.method !== "HEAD") {
        response.end(body);
      } else {
        response.end();
      }
    } catch (error) {
      response.writeHead(500);
      response.end(String(error));
    }
    return;
  }

  let stat;
  try {
    stat = await fs.promises.stat(filePath);
  } catch {
    response.writeHead(404);
    response.end("Not found");
    return;
  }

  if (!stat.isFile()) {
    response.writeHead(404);
    response.end("Not found");
    return;
  }

  response.setHeader("content-length", String(stat.size));
  response.setHeader("content-type", contentType(filePath));
  response.setHeader("cache-control", "no-store");
  response.writeHead(200);

  if (request.method === "HEAD") {
    response.end();
    return;
  }

  fs.createReadStream(filePath, { highWaterMark: 1024 * 1024 }).pipe(response);
}

function setIsolationHeaders(response) {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
}

function contentType(filePath) {
  switch (path.extname(filePath)) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".js":
    case ".mjs":
      return "text/javascript; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".wasm":
      return "application/wasm";
    case ".onnx":
      return "application/octet-stream";
    case ".png":
      return "image/png";
    case ".svg":
      return "image/svg+xml";
    default:
      return "application/octet-stream";
  }
}

async function assertFlatRows(page, { expectGloss, expectProbability }) {
  const rows = await page.locator(".variant-popover .variant-row:not(.variant-status)")
    .evaluateAll((nodes) =>
      nodes.map((row) => {
        const headword = row.querySelectorAll(".variant-headword-line").length;
        const morphology = Array.from(row.querySelectorAll(".variant-morphology"));
        const gloss = Array.from(row.querySelectorAll(".variant-gloss"));
        const probability = row.querySelectorAll(".variant-probability").length;
        const legacy = row.querySelectorAll(
          "ruby, rt, .variant-info, .variant-reading, .probability-chip",
        ).length;

        return {
          headword,
          morphologyCount: morphology.length,
          morphologyText: morphology.map((node) => node.textContent ?? ""),
          morphologyChildren: morphology.reduce(
            (sum, node) => sum + node.children.length,
            0,
          ),
          glossCount: gloss.length,
          glossText: gloss.map((node) => node.textContent ?? ""),
          glossChildren: gloss.reduce((sum, node) => sum + node.children.length, 0),
          probability,
          legacy,
        };
      }),
    );

  assert(rows.length > 0, "Popover rendered no reading rows.");
  assert(rows.some((row) => row.morphologyCount === 1), "Popover rows had no morphology line.");

  for (const row of rows) {
    assert(row.headword === 1, "A reading row did not have exactly one headword line.");
    assert(row.morphologyCount <= 1, "A reading row stacked multiple morphology lines.");
    assert(row.glossCount <= 1, "A reading row stacked multiple gloss lines.");
    assert(row.morphologyChildren === 0, "Morphology line contains nested segment markup.");
    assert(row.glossChildren === 0, "Gloss line contains nested segment markup.");
    assert(row.legacy === 0, "Legacy stacked popover markup is still present.");

    if (row.morphologyCount === 1) {
      assert(
        !row.morphologyText[0].includes("\n"),
        "Morphology line contains explicit line breaks.",
      );
      assert(
        expectGloss ? row.glossCount === 1 : row.glossCount === 0,
        expectGloss
          ? "RU popover row missed the parallel translated gloss line."
          : "LT popover row rendered a translated gloss line.",
      );
    }
  }

  const hasMiddot = rows.some((row) =>
    row.morphologyText.some((text) => text.includes(" · ")),
  );
  assert(hasMiddot, "No morphology line used middot separators.");

  const probabilityCount = rows.reduce((sum, row) => sum + row.probability, 0);
  assert(
    expectProbability ? probabilityCount > 0 : probabilityCount === 0,
    expectProbability
      ? "Local probability popover did not show quiet percentages."
      : "Non-local popover unexpectedly showed probability percentages.",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
