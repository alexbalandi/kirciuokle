# Phase 11 — repair MATAS UD features from Jablonskis XPOS

Diagnosis from the first GPU run (modernBERT combined/first, 2 epochs on
MATAS+ALKSNIS): dev UPOS 89.0% but feats-exact only 59.8% / slots 79.8%.
Per-key coverage comparison proved the cause: **MATAS's UD FEATS column is
structurally incomplete** — AUX tokens carry no features at all
(Mood/Number/Person/Tense/VerbForm ≈ 0% vs ALKSNIS 70–100%), DET has no
Case/Gender/Number, NUM lacks NumForm/Case/Gender mostly, PRON lacks
Person/PronType, VERB under-carries Number/Polarity. Training teaches the
model to omit exactly what our pipeline scores. MATAS's Jablonskis XPOS
column, however, is complete (`vksm.asm.tiesiog.es.vns.3.` style) — the
missing UD features are recoverable from it.

## prep_corpus.py changes

1. `--feats-keys slots|all` (default `slots`): with `slots`, labels keep
   ONLY the scoring-relevant UD keys:
   `Case, Gender, Number, Tense, Person, Voice, Degree, VerbForm, Mood,
   Reflex` (Mood and Reflex kept because they are recoverable, consistent,
   and useful for disambiguation context; everything else — Definite,
   Polarity, PronType, NumType, NumForm, NameType, Abbr, Foreign, Hyph,
   Aspect, etc. — is dropped from labels in ALL sources). This shrinks the
   label space and removes cross-corpus convention noise.
2. `--repair-from-xpos` (default on for matas, no-op for alksnis): for each
   MATAS token, parse the Jablonskis XPOS (dot-separated segments) into UD
   features and MERGE into FEATS: existing UD keys win, missing keys are
   filled. Mapping table (module-level, documented):
   - Case: `V.`→Nom, `K.`→Gen, `N.`→Dat, `G.`→Acc, `Įn.`→Ins, `Vt.`→Loc,
     `Š.`→Voc, `Il.`→Ill (Illative; keep — ALKSNIS uses Case=Ill)
   - Gender: `vyr.`→Masc, `mot.`→Fem, `bevrd.`/`bev.`→Neut
   - Number: `vns.`→Sing, `dgs.`→Plur, `dvisk.`→Dual
   - Tense: `es.`→Pres, `būt.`→Past, `būt-k.`→Past, `būt-d.`→`Past` with
     `Aspect` dropped anyway (map plain Past), `būs.`→Fut
   - Person: `1.`→1, `2.`→2, `3.`→3 (segments exactly matching those
     digit-dot forms in verbal XPOS)
   - Mood: `tiesiog.`→Ind, `tar.`→Cnd, `liep.`→Imp
   - Voice: `veik.`→Act, `neveik.`→Pass, `reik.`→Nec
   - VerbForm: `asm.`→Fin, `bndr.`→Inf, `dlv.`→Part, `pusd.`→Conv,
     `padlv.`→Ger, `būdn.`→Part (approximation)
   - Reflex: `sngr.`→Yes
   Unknown segments are ignored. Only fill keys that survive
   `--feats-keys`.
   IMPORTANT verification hook: after prep, print a per-UPOS coverage
   table for the kept keys in the produced TRAIN split vs ALKSNIS dev
   (reuse the logic below), so the repair is visible in numbers.
3. Keep canonicalization/dedup/leakage guard as-is. Print label-set size
   before/after (expect a large drop from 2,566).

## Validation utility

`local/tagger-hf/coverage_diff.py`: the per-UPOS FEATS-key coverage
comparison used for the diagnosis (two conllu/jsonl inputs, prints keys
with coverage deltas >N%). Runnable standalone; prep imports or shells the
same logic for its post-repair report.

## Quality bar

- py_compile; selfcheck still passes; `--help` everywhere.
- Do NOT run the full prep or any training (orchestrator does).
- No changes outside `local/tagger-hf/`. No git.
