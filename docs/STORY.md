# The end-path: how the pieces became one system

Written 2026-07-06, at the point where the joint model became the
project's best accentuator and the teacher-labeler was being built.
Read this to understand WHY the current design is what it is.

## The arc in six moves

**1. Serving first (the app).** A Cloudflare app accenting text via VDU
kirčiuoklė + UDPipe disambiguation, with a self-warming dictionary
cache. This gave us the production baseline — and, unnoticed at the
time, the future silver-truth generator.

**2. Own taggers.** Fine-tuned litlat-bert taggers on MATAS+ALKSNIS
(gold morphology, open licenses) reached UDPipe-class quality locally
(slots 86–89%). Lesson that stuck: bake-off encoders on identical data
(litlat-bert beat Lithuanian ModernBERT by 5.4pp); benchmark-gate every
change.

**3. Own dictionary.** 14 generation modules turn open sources (kaikki
accent classes + published accentology rules + VLKK normative data)
into 575k accented words at DISJOINT=0 against the VDU cache
(docs/ACCENTUATOR.md). Lessons: no answer beats wrong answer; adjudicate
every disagreement with a citation; the residue that no open source
covers is LEXICAL — pattern models cannot close it (proved next).

**4. The guesser bake-off.** For out-of-dictionary words we built and
raced: a naive suffix trie (51% on gap words), a faithful
Anbinderis & Kasparaitis 2010 replication (abstains honestly, 67% of
answered), LIEPA/phonology_engine (88%), neural stress models. Two
durable findings: (a) on gap words a LOOKUP (LIEPA's embedded lexicon)
beats every pattern model — the gap is lexical; (b) where two
independent systems AGREE, accuracy jumps to ~99.5% — the seed of the
teacher-labeler.

**5. Conditioning, then fusion.** v2 conditioned the stress model on
morphology labels (homograph switching became possible, +5.5pp
end-to-end through the real tagger); v3 added a learned no-stress class
for foreign words (+4.8pp on gap, foreign-abstention 55→63%). Then the
architectural bet: ONE encoder, one pass, both heads — trained on MATAS
gold morphology with stress PROJECTED from our own dictionary (silver
where it's unambiguous, masked where it isn't). The joint model beat
the entire two-model pipeline on every axis (87.9% vs 83.5% audited
LRT; 76% vs 63% foreign-abstention; 88.9% POS; half the parameters).
Why: no lossy word+label bottleneck — the stress head reads full
sentence context; and contextual running-text training beats isolated
dictionary rows.

**6. The teacher-labeler (current).** Gold data for accents effectively
does not exist to buy or download (docs/DATASETS.md, rejected-sources
list). But we hold several INDEPENDENT annotators (VDU+UDPipe silver at
95.8% on gold, joint 86.9%, LIEPA, dictionary+labels) whose agreement
strata can be CALIBRATED against the one gold set we recovered (the
chrestomatija, via the Wayback Machine). The teacher accepts a token's
label only when its agreement pattern's measured accuracy clears a
threshold; unlabeled tokens are masked in training (coverage optional,
purity compounding). That turns any new corpus — public-domain literary
classics for the register gap, unlimited LRT for modern text — into
training data with a purity certificate.

## How future training works (the loop)

1. Pick a corpus (public-domain classics / fresh LRT / anything).
2. `build_silver_truth.py` once (external, throttled, resumable).
3. `teacher/collect_layers.py` → every internal system's opinion.
4. `teacher/label.py` with calibrated strata → training jsonl + purity.
5. Fine-tune the joint model on a REHEARSAL MIXTURE (~80/20 new/MATAS
   rows — per-token loss masking makes this trivial) at polish-grade LR
   with per-epoch dev selection.
6. Re-run the untouched gold benchmark (chrestomatija) and audited LRT.
   Gains real → keep; benchmark flat → the teacher taught its own
   noise; investigate strata.

## Mistakes worth not repeating

- **Don't average references — adjudicate.** Every VDU/dictionary
  disagreement got a citation-backed verdict; both directions produced
  fixes (32 silver errors, 6 dictionary bugs from one audit).
- **Don't trust saturated dev sets.** MATAS dev (dictionary-projected)
  reads 99%+ and cannot rank checkpoints; judge on ALKSNIS/LRT/gold.
- **Don't count marks on NFC strings** (composed à hides its grave —
  this bug alone made LIEPA look like it answered 12% instead of 80%).
- **Don't let a smoke test overwrite a real checkpoint** (SPEC22's
  criterion clobbered v1; per-epoch atomic saves + .bak now standard).
- **Don't ship a guess without tier semantics.** Unmarked or
  double-marked outputs must abstain, not "answer"; provenance strings
  carry the tier so consumers can trust-filter.
- **Don't condition on free-text labels** — select from the closed
  label vocabulary the model trained on; tie-break toward fewest
  spurious slots.
- **Sequence accuracy comparisons across papers are indicative only**
  unless tokenization, sample length, and normalization match — do the
  back-arithmetic (their 0.711 seq ≈ 97.9% token at ~16 tokens/sample)
  before concluding anything.
- **Register is a real axis.** A model trained on news+dictionary
  paradigms loses ~9pp on poetry; the benchmark that exposes this must
  never become training data.
- **Loss curves lie at boundaries** if the logging window resets
  mid-average (the "tiny loss at epoch start" mirage — divide by the
  actual window size).
- **Crash insurance is cheap** (atomic per-epoch checkpoints) and the
  one time you need it, it saves a 3-hour GPU run — ask us how we know.
