# SPEC35 — Public-domain literary corpus for the register fine-tune

## Goal

Fetch ~200k tokens of public-domain Lithuanian classic literature
(prose + poetry, the register the chrestomatija benchmark punishes us
on), cleaned and guarded, as `data/eval/literary-corpus.txt` (one
sentence per line, blank line between works) + a meta sidecar listing
work/author/source-URL per segment.

One new script `local/accentuator/fetch_literary_corpus.py`; modify
nothing else.

## Source: lt.wikisource (Vikišaltiniai) via the MediaWiki API

Use the API (`https://lt.wikisource.org/w/api.php`) — structured,
legal, no scraping: list category members / author pages, fetch page
wikitext or parsed text (`action=parse`/`prop=extracts` or
`action=query&prop=revisions&rvprop=content`), strip wiki markup.
Throttle ≥0.5s between requests, normal User-Agent.

## Author whitelist (public domain: died ≥70 years ago, verified)

Maironis (d.1932), Žemaitė (d.1921), Jonas Biliūnas (d.1907),
Šatrijos Ragana (d.1930), Vincas Kudirka (d.1899), Antanas
Baranauskas (d.1902), Vincas Krėvė-Mickevičius (d.1954), Jurgis
Savickis (d.1952), Vydūnas (d.1953), Motiejus Valančius (d.1875),
Balys Sruoga (d.1947), Salomėja Nėris (d.1945), Vytautas Mačernis
(d.1944), Julius Janonis (d.1917), Petras Cvirka (d.1947).

EXCLUDED deliberately (not yet PD or joint-authorship risk): Lazdynų
Pelėda (co-author d.1957), Antanas Vienuolis (d.1957), Ignas Šeinius
(d.1959). Do not add authors beyond the whitelist without a death-year
check in the meta sidecar.

Survey what lt.wikisource actually holds per author (category/author
pages), prefer a prose/poetry MIX (the benchmark is ~half poetry), and
take whole works up to the token budget (--max-tokens default 220000,
stop cleanly at a work boundary).

## Cleaning

- Strip wiki markup, footnotes, editorial headers, page-number
  artifacts; drop ALL-CAPS-only lines and lines shorter than 3 word
  tokens.
- Normalize to NFC; collapse whitespace; keep original punctuation.
- Sentence-split prose the same way the repo does elsewhere
  ([.!?…] + capital); for poetry keep VERSE LINES as units when
  sentence splitting would join many lines (a line with ≥3 word tokens
  is a unit).
- Old orthography guard: skip works whose text contains high rates of
  non-modern characters or obviously pre-modern spelling (heuristic:
  frequency of "ł", "ó" outside loanwords, "sz", "cz" digraphs —
  print per-work flags and skip flagged works, listing them).

## The benchmark firewall (CRITICAL)

Load `data/eval/chrestomatija-gold.jsonl`, strip accents+lowercase each
gold sentence, and DROP any corpus sentence whose normalized form
matches a gold sentence (exact match after whitespace collapse) OR
shares an 8-word shingle with any gold sentence. Report the number of
dropped sentences and which works they came from (a nonzero count means
the chrestomatija sampled that author — expected occasionally; the
firewall is why training stays clean).

## Pass criteria

1. Run with `--max-tokens 30000` (smoke): meta sidecar written, per-work
   list with author/death-year/URL/token-count printed, firewall report
   printed, 5 random sample sentences pasted.
2. Full run `--max-tokens 220000`: same reports; total tokens
   190k–230k; poetry share between 25% and 60% (report it; adjust the
   work mix if outside).
3. Sample sentences must be clean modern-orthography literary
   Lithuanian (no wiki markup, no footnote digits glued to words).

Do not commit. The corpus stays in the gitignored data dir.
