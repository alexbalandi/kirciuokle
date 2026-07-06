# SPEC39 — Corpus enrichment: modern Wikipedia + deeper classics

## Context

Next fine-tune round needs more training text (LRT is quarantined as
validation). Two prongs: topically diverse modern prose from
lt.wikipedia (CC BY-SA; also rich in foreign names — the
foreign-abstention regression's antidote) and more public-domain
classics.

Files: new `local/accentuator/fetch_wikipedia_corpus.py`; modify
`local/accentuator/fetch_literary_corpus.py` (whitelist + cap only).

## 1. fetch_wikipedia_corpus.py

- MediaWiki API on lt.wikipedia (throttle ≥0.5s, normal UA). Sample
  ARTICLE DIVERSITY: pull from a spread of categories/portals
  (science, history, geography, sports, culture, technology, politics,
  biology — aim ≥8 topic buckets, roughly balanced) using category
  members + `prop=extracts&explaintext` (plain text, no wikitext
  parsing pain).
- Per article: strip section headers, infobox residue, reference
  markers, parenthetical pronunciation clutter `(   )` artifacts;
  sentence-split as elsewhere; drop lines <3 word tokens; NFC.
- Skip list-heavy articles (heuristic: >40% of lines short or
  starting with bullets/digits).
- Budget `--max-tokens` default 280000, stop at article boundary;
  meta sidecar (title, category bucket, URL, tokens).
- FIREWALL against BOTH eval sets: drop sentences matching (exact or
  8-word shingle) the chrestomatija gold OR the LRT corpus
  (data/eval/lrt-corpus.txt) — LRT is validation; Wikipedia text
  occasionally quotes news. Report drops.

## 2. fetch_literary_corpus.py: widen the net

- Add verified-PD authors to the whitelist (death year in comment):
  Juozas Tumas-Vaižgantas (d.1933), Gabrielė Petkevičaitė-Bitė
  (d.1943), Kazys Binkis (d.1942), Pranas Vaičaitis (d.1901),
  Jurgis Baltrušaitis (d.1944, Lithuanian-language works only).
- Add `--exclude-works` (paths of previous meta sidecars) so a second
  run fetches only NEW works (the first 220k-token corpus must not be
  duplicated).
- Keep every existing guard (old orthography, firewall). Raise default
  --max-tokens to 260000 for the second pull.

## Pass criteria

1. Wikipedia smoke (--max-tokens 25000): bucket distribution printed,
   firewall report (both eval sets), 5 sample sentences pasted — clean
   encyclopedic prose.
2. Wikipedia full run (280k): same reports; ≥8 buckets each ≥5% share.
3. Literary second pull with --exclude-works pointing at the existing
   meta: zero overlap with pull #1 (assert by work title), new-author
   works present; report per-author tokens; total may land under the
   cap if wikisource runs dry — report what exists.
4. Corpora land in data/eval/ (gitignored) as wikipedia-corpus.txt and
   literary-corpus-2.txt with meta sidecars.

Do not commit.
