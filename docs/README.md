# Documentation map

This project built, in roughly this order: a Lithuanian accentuation web
app backed by VDU's kirčiuoklė; its own POS taggers; its own open
accentuation dictionary; a family of stress-guessing models; a single-pass
joint POS+stress model that now beats the multi-component pipeline; and a
calibrated teacher-labeler for manufacturing training data. These docs
exist so someone (including future us) can build on the work without
re-deriving it — or repeating the mistakes.

## Core references

| doc | contents |
|---|---|
| [ARCHITECTURES.md](ARCHITECTURES.md) | Exact model architectures: taggers, stress heads v1–v3, the joint model — dims, masks, losses, warm starts |
| [DATASETS.md](DATASETS.md) | Every dataset: source, license, how obtained, what it may and may not be used for |
| [EVALUATION.md](EVALUATION.md) | The evaluation methodology ladder: parity → silver → audited silver → gold; metric definitions used everywhere |
| [STORY.md](STORY.md) | The end-path narrative: how the pieces led to the joint model + teacher-labeler, how future training works, and the mistakes worth not repeating |
| [ACCENTUATOR.md](ACCENTUATOR.md) | The open dictionary pipeline (14 generation modules, rules, provenance tiers, parity QA) |
| [../local/README.md](../local/README.md) | The self-hosted replica, released tagger models, tagger benchmarks |

## Experiment log

Structured by theme, chronological within each file:

| file | theme |
|---|---|
| [experiments/01-taggers.md](experiments/01-taggers.md) | POS tagger campaign: encoders, corpora, released models |
| [experiments/02-dictionary.md](experiments/02-dictionary.md) | Open dictionary: sources, generation modules, parity rounds |
| [experiments/03-guessers.md](experiments/03-guessers.md) | OOV stress guessers: trie, Anbinderis rules, LIEPA, ensembles |
| [experiments/04-stress-models.md](experiments/04-stress-models.md) | Neural stress models v1–v3: conditioning, no-stress class |
| [experiments/05-joint-model.md](experiments/05-joint-model.md) | The joint POS+stress model: dataset projection, training, polish |
| [experiments/06-teacher-labeler.md](experiments/06-teacher-labeler.md) | The calibrated consensus teacher (in progress) |

## Specs

`SPEC*.md` files are implementation specs handed to the codex CLI — each
is self-contained with verified API contracts and numbered pass criteria.
They document what was built, in what order, and double as examples of
the orchestration workflow (human/Claude writes specs and adjudicates;
codex implements; every pass criterion is executed before review).

## Reports (committed numbers)

`../local/accentuator/reports/`: `parity-vdu.md` (dictionary vs VDU
cache), `guesser-bench.md` (OOV guesser bake-off),
`live-guess-eval.md` + `nodict-eval.md` + `nodict-eval-v3.md` (unseen
LRT text, raw + audited), `joint-eval.md` (joint model),
`chrestomatija-eval.md` (gold benchmark, all systems).
