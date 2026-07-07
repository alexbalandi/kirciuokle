const { chromium } = require("playwright");

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8765/";
const consentScreenshotPath = process.argv[3] ?? "docs-consent.png";
const SPEC_TEXT =
  "81-erių vilnietė pardavė butą ir nusikaltėliui atidavė 114 tūkst. eurų. " +
  "Tuo metu valstybės institucijos, nevyriausybinės organizacijos ir verslininkai suka galvas, " +
  "kaip dar padėti žmonėms nepakliūti į sukčių pinkles.";

const MODEL_CACHE = "main-local-accent-model-v1";
const MODEL_FILE = "joint.int8.onnx";
const MODEL_BYTES = 537_586_710;

const TIMEOUTS = {
  page: 30_000,
  localReady: 600_000,
  localAccent: 240_000,
  ui: 15_000,
};

const diagnostics = {
  console: [],
  pageErrors: [],
  failedRequests: [],
  localModelRequests: 0,
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

  try {
    await assertFreshConsentFlow(browser);
    await assertCachedSkipPath(browser);
  } finally {
    await browser.close();
  }
}

async function newPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  page.on("console", (message) => {
    diagnostics.console.push({
      type: message.type(),
      text: message.text().slice(0, 500),
    });
  });
  page.on("pageerror", (error) => {
    diagnostics.pageErrors.push(error.message);
  });
  page.on("request", (request) => {
    if (request.url().includes("/local-model/")) {
      diagnostics.localModelRequests += 1;
    }
  });
  page.on("requestfailed", (request) => {
    diagnostics.failedRequests.push({
      url: request.url(),
      failure: request.failure()?.errorText ?? "unknown",
    });
  });
  return page;
}

async function assertFreshConsentFlow(browser) {
  const page = await newPage(browser);
  try {
    await page.addInitScript(() => {
      localStorage.setItem("lang", "en");
      localStorage.setItem("accent-mode", "web");
      localStorage.setItem("accent-display", "all");
    });

    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });
    await page.evaluate(async (cacheName) => {
      for (const key of await caches.keys()) {
        await caches.delete(key);
      }
      localStorage.setItem("lang", "en");
      localStorage.setItem("accent-mode", "web");
      localStorage.setItem("accent-display", "all");
    }, MODEL_CACHE);
    await page.reload({ waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });

    diagnostics.localModelRequests = 0;
    await page.click('[data-mode="local"]', { timeout: TIMEOUTS.ui });
    await page.waitForSelector(".local-consent-card", { timeout: TIMEOUTS.ui });
    await page.waitForTimeout(500);

    assert(
      diagnostics.localModelRequests === 0,
      "Local mode requested /local-model/ assets before consent.",
    );

    const consentText = (await page.locator(".local-consent-card").textContent()) ?? "";
    assert(
      consentText.includes("To accentuate locally, the site downloads the model once"),
      "Consent card did not render the English consent copy.",
    );
    assert(consentText.includes("538 MB"), "Consent card did not show decimal MB.");
    assert(!consentText.includes("MiB"), "Consent card still showed MiB.");
    assert(
      consentText.includes("Download model (538 MB)"),
      "Consent button did not include the model size.",
    );

    await page.locator(".local-consent-card").screenshot({
      path: consentScreenshotPath,
    });

    await page.click(".local-consent-button", { timeout: TIMEOUTS.ui });
    await page.waitForFunction(() => window.__localAccentReady === true, null, {
      timeout: TIMEOUTS.localReady,
    });
    assert(
      diagnostics.localModelRequests > 0,
      "Consent click did not start local model asset requests.",
    );

    const readyText = (await page.locator("#local-status").textContent()) ?? "";
    assert(readyText.includes("538 MB"), "Ready status did not show decimal MB.");
    assert(!readyText.includes("MiB"), "Ready status still showed MiB.");

    await page.fill("#source-text", SPEC_TEXT, { timeout: TIMEOUTS.ui });
    await page.click("#accent-button", { timeout: TIMEOUTS.ui });
    await page.waitForFunction(
      () => Boolean(window.__localAccentStats?.lastRun?.tokens),
      null,
      { timeout: TIMEOUTS.localAccent },
    );

    const bodyText = (await page.locator("body").textContent()) ?? "";
    assert(bodyText.includes("538 MB"), "Page did not expose MB-sized model text.");
    assert(!bodyText.includes("MiB"), "Page exposed MiB in user-facing text.");

    const numeral = page.locator("#result-output .token-numeral").first();
    assert((await numeral.count()) === 1, "81-erių did not render as a numeral token.");
    assert(
      ((await numeral.textContent()) ?? "").startsWith("81-"),
      "Numeral token did not keep the number prefix.",
    );
    assert(
      !(await numeral.evaluate((node) => node.classList.contains("token-ambiguous"))),
      "Numeral token rendered with ambiguous styling.",
    );
    await numeral.click({ timeout: TIMEOUTS.ui });
    await page.waitForTimeout(500);
    assert(
      (await page.locator(".variant-popover").count()) === 0,
      "Numeral token opened a POS popover.",
    );
    assert(
      (await page.locator(".variant-probability").count()) === 0,
      "Numeral token produced probability percentages.",
    );

    await page.locator("#result-output button.token").first().click({
      timeout: TIMEOUTS.ui,
    });
    await page.waitForSelector(".variant-popover .variant-probability", {
      timeout: TIMEOUTS.ui,
    });
    const chips = await page.locator(".variant-probability").allTextContents();
    assert(chips.length > 0, "Local popover did not show probability chips.");
    for (const chip of chips) {
      const text = chip.trim();
      const numeric = Number(text.replace("%", ""));
      assert(Number.isFinite(numeric), `Invalid probability chip: ${text}`);
      if (numeric >= 10) {
        assert(/^\d+%$/.test(text), `Probability >=10% was not an integer: ${text}`);
      } else {
        assert(/^\d+\.\d%$/.test(text), `Probability <10% lacked one decimal: ${text}`);
      }
    }
  } finally {
    await page.close();
  }
}

async function assertCachedSkipPath(browser) {
  const page = await newPage(browser);
  try {
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });
    await page.evaluate(
      async ({ cacheName, modelFile, modelBytes }) => {
        for (const key of await caches.keys()) {
          await caches.delete(key);
        }

        const cache = await caches.open(cacheName);
        const modelUrl = new URL(`/local-model/${modelFile}`, location.href).href;
        await cache.put(
          new Request(`${modelUrl}?local-cache=chunks`, {
            credentials: "same-origin",
          }),
          new Response(
            JSON.stringify({
              version: 1,
              bytes: modelBytes,
              chunks: 1,
            }),
            { headers: { "content-type": "application/json" } },
          ),
        );

        localStorage.setItem("lang", "en");
        localStorage.setItem("accent-mode", "local");
      },
      { cacheName: MODEL_CACHE, modelFile: MODEL_FILE, modelBytes: MODEL_BYTES },
    );

    await page.route("**/local-model/**", (route) => route.abort());
    diagnostics.localModelRequests = 0;
    await page.reload({ waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });
    await page.waitForFunction(
      () =>
        !document.querySelector(".local-consent-card") &&
        document.querySelector("#local-status")?.textContent !==
          "Checking whether the model is already saved in this browser...",
      null,
      { timeout: TIMEOUTS.ui },
    ).catch(() => {});

    assert(
      (await page.locator(".local-consent-card").count()) === 0,
      "Cached Local mode showed the consent card instead of skipping it.",
    );
    assert(
      diagnostics.localModelRequests > 0,
      "Cached Local mode did not start loading the model automatically.",
    );
  } finally {
    await page.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
