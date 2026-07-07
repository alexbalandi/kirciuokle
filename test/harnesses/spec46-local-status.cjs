const { chromium } = require("playwright");

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8767/";
const timeoutMs = Number(process.argv[3] ?? 180_000);

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  const consoleLines = [];
  page.on("console", (message) => {
    consoleLines.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    consoleLines.push(`pageerror: ${error.message}`);
  });
  page.on("requestfailed", (request) => {
    consoleLines.push(
      `requestfailed: ${request.url()} ${request.failure()?.errorText ?? "unknown"}`,
    );
  });

  await page.goto(baseUrl, { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.evaluate(() => {
    localStorage.setItem("accent-mode", "web");
  });
  await page.reload({ waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.click('[data-mode="local"]', { timeout: 15_000 });
  await page.waitForTimeout(500);
  if ((await page.locator(".local-consent-button").count()) > 0) {
    await page.click(".local-consent-button", { timeout: 15_000 });
  }

  const start = Date.now();
  let last = "";
  while (Date.now() - start < timeoutMs) {
    const state = await page.evaluate(() => ({
      ready: window.__localAccentReady === true,
      status: document.querySelector("#local-status")?.textContent ?? "",
      message: document.querySelector("#form-message")?.textContent ?? "",
      stats: window.__localAccentStats ?? null,
    }));
    const line = `${Math.round((Date.now() - start) / 1000)}s ready=${
      state.ready
    } status="${state.status}" message="${state.message}"`;
    if (line !== last) {
      console.log(line);
      last = line;
    }
    if (state.ready) {
      console.log(JSON.stringify(state.stats, null, 2));
      await browser.close();
      return;
    }
    await page.waitForTimeout(5000);
  }

  console.log("Console diagnostics:");
  console.log(consoleLines.slice(-80).join("\n"));
  await browser.close();
  process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
