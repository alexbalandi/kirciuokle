// Prebuild guard: the spellcheck dictionaries are gitignored build artifacts, so
// a fresh checkout (or a stale one) can build a site whose spellcheck silently
// 404s. Fail the build fast with a pointer to the regenerate script instead.
import { existsSync } from "node:fs";

const required = ["public/spellcheck-lt.txt", "public/spellcheck-bigrams.txt"];
const missing = required.filter((path) => !existsSync(path));

if (missing.length > 0) {
  console.error(`\n✖ Missing generated spellcheck assets: ${missing.join(", ")}`);
  console.error("  These are gitignored build artifacts (like the model). Regenerate:");
  console.error("    uv run scripts/regenerate_spellcheck_dicts.py\n");
  process.exit(1);
}
