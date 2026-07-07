const { chromium } = require("playwright");

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8767/";
const screenshotPath = process.argv[3] ?? "docs-local-mode.png";
const SPEC42_TEXT =
  "81-erių vilnietė pardavė butą ir nusikaltėliui atidavė 114 tūkst. eurų. " +
  "Tuo metu valstybės institucijos, nevyriausybinės organizacijos ir verslininkai suka galvas, " +
  "kaip dar padėti žmonėms nepakliūti į sukčių pinkles. Pavyzdžiui, verslai ant kvitų spausdina " +
  "patarimus bei numerį, kuriuo reikėtų skambinti įtarus, kad susiduria su nusikaltėliu. " +
  "Vis dėlto ekspertai pabrėžia, kad svarbiausia vadovautis kritiniu mąstymu ir elgtis atsakingai, " +
  "jog tokie atvejai nebepasikartotų.";

const TIMEOUTS = {
  page: 30_000,
  webAccent: 120_000,
  localReady: 600_000,
  localAccent: 240_000,
  ui: 15_000,
};

const diagnostics = {
  console: [],
  pageErrors: [],
  failedRequests: [],
  modelRequests: 0,
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
    if (request.url().includes("/local-model/joint")) {
      diagnostics.modelRequests += 1;
    }
  });
  page.on("requestfailed", (request) => {
    diagnostics.failedRequests.push({
      url: request.url(),
      failure: request.failure()?.errorText ?? "unknown",
    });
  });

  try {
    await page.addInitScript(() => {
      localStorage.setItem("accent-mode", "web");
      localStorage.setItem("accent-display", "all");
    });
    diagnostics.modelRequests = 0;
    await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });
    await page.waitForTimeout(1000);
    assert(diagnostics.modelRequests === 0, "Web mode loaded model bytes unexpectedly.");

    await assertWebMode(page);
    await assertLocalMode(page);
    await assertWebReloadDoesNotDownload(page);
  } finally {
    await browser.close();
  }
}

async function assertWebMode(page) {
  await page.fill("#source-text", SPEC42_TEXT, { timeout: TIMEOUTS.ui });
  await page.click("#accent-button", { timeout: TIMEOUTS.ui });
  await page.waitForFunction(
    () => !document.querySelector("#accent-button")?.disabled,
    null,
    { timeout: TIMEOUTS.webAccent },
  );

  const message = await page.locator("#form-message").textContent();
  assert(!message, `Web mode reported an error: ${message}`);

  const resultText = (await page.locator("#result-output").textContent()) ?? "";
  assert(resultText.length > 40, "Web mode did not render accented output.");
  assert(await page.locator("#local-stats-button").isHidden(), "Stats icon is visible in Web mode.");
  assert(
    !(await page.locator('[data-display="top"]').isDisabled()),
    "Display control is disabled in Web mode.",
  );

  await page.click('[data-display="top"]', { timeout: TIMEOUTS.ui });
  assert(
    await page.locator('[data-display="top"].is-active').count(),
    "Top display option did not activate in Web mode.",
  );
  await page.click('[data-display="all"]', { timeout: TIMEOUTS.ui });
  assert(
    await page.locator('[data-display="all"].is-active').count(),
    "All display option did not activate in Web mode.",
  );

  const token = page.locator("#result-output button.token").first();
  if ((await token.count()) > 0) {
    await token.click({ timeout: TIMEOUTS.ui });
    try {
      await page.waitForSelector(".variant-popover", { timeout: 5000 });
      assert(
        (await page.locator(".variant-probability").count()) === 0,
        "Web mode rendered probability percentages.",
      );
      await page.keyboard.press("Escape");
    } catch {
      diagnostics.console.push({
        type: "warn",
        text: "No clickable Web popover opened for the SPEC42 paragraph.",
      });
    }
  }
}

async function assertLocalMode(page) {
  await page.click('[data-mode="local"]', { timeout: TIMEOUTS.ui });
  await page.waitForTimeout(500);
  assert(
    diagnostics.modelRequests === 0,
    "Local mode requested model bytes before consent.",
  );
  if ((await page.locator(".local-consent-button").count()) > 0) {
    await page.click(".local-consent-button", { timeout: TIMEOUTS.ui });
  }
  await page.waitForFunction(() => window.__localAccentReady === true, null, {
    timeout: TIMEOUTS.localReady,
  });
  assert(await page.locator("#local-stats-button").isVisible(), "Stats icon is hidden in Local mode.");
  assert(
    await page.locator('[data-display="top"]').isDisabled(),
    "Top/all display control is not disabled in Local mode.",
  );
  assert(
    await page.locator('[data-display="top"].is-active').count(),
    "Local mode did not force the top display option.",
  );

  await page.fill("#source-text", SPEC42_TEXT, { timeout: TIMEOUTS.ui });
  await page.click("#accent-button", { timeout: TIMEOUTS.ui });
  await page.waitForFunction(
    () => Boolean(window.__localAccentStats?.lastRun?.tokens),
    null,
    { timeout: TIMEOUTS.localAccent },
  );

  const resultText = (await page.locator("#result-output").textContent()) ?? "";
  assert(resultText.length > 40, "Local mode did not render output.");

  await page.click("#local-stats-button", { timeout: TIMEOUTS.ui });
  await page.waitForSelector(".stats-popover", { timeout: TIMEOUTS.ui });
  assert(
    ((await page.locator(".stats-popover").textContent()) ?? "").length > 20,
    "Stats popover opened but was empty.",
  );
  await page.keyboard.press("Escape");

  await page.locator("#result-output .token").first().click({ timeout: TIMEOUTS.ui });
  await page.waitForSelector(".variant-popover .variant-probability", {
    timeout: TIMEOUTS.ui,
  });
  assert(
    ((await page.locator(".variant-popover").textContent()) ?? "").includes("%"),
    "Local popover did not show probability chips.",
  );
  await page.screenshot({ path: screenshotPath });
}

async function assertWebReloadDoesNotDownload(page) {
  await page.keyboard.press("Escape");
  await page.click('[data-mode="web"]', { timeout: TIMEOUTS.ui });
  diagnostics.modelRequests = 0;
  await page.reload({ waitUntil: "domcontentloaded", timeout: TIMEOUTS.page });
  await page.waitForTimeout(1000);

  assert(
    await page.locator('[data-mode="web"].is-active').count(),
    "Web mode did not persist across reload.",
  );
  assert(
    diagnostics.modelRequests === 0,
    "Reloading into Web mode requested local model bytes.",
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
