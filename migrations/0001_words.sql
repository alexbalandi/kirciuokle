CREATE TABLE IF NOT EXISTS words (
  word TEXT PRIMARY KEY,
  variants TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  negative_until TEXT
);
