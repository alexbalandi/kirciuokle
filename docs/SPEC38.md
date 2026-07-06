# SPEC38 — Teacher v2: independent POS voter (UDPipe lineage)

## Context

Teacher v1's POS track collapsed to 0% coverage: its two voters (joint
model, released -vdu tagger) share a warm-start lineage, so their
agreement stratum measures only 85.9% on ALKSNIS gold — correlated
voters don't purify. Fix: capture the raw UDPipe (LINDAT) tags — a
genuinely independent lineage — during the silver build, add them as a
third POS layer, recalibrate.

Files: modify `local/accentuator/build_silver_truth.py` (one added
output field), `local/accentuator/teacher/collect_layers.py`,
`local/accentuator/teacher/calibrate.py`,
`local/accentuator/teacher/label.py`. Nothing else.

## 1. build_silver_truth.py: dump raw UDPipe tags

The pipeline already receives UDPipe CoNLL-U per chunk (via
accent_text machinery). Add to each token row: `"ud": {"upos": ...,
"feats": "<raw FEATS string>"}` (null when the aligner has no UDPipe
token for it). BACKWARD COMPATIBLE: all existing consumers read named
keys and must keep working; existing silver files without the field
must still load everywhere (treat missing as null). Do not re-fetch
existing silver files.

## 2. collect_layers.py: `udpipe` POS layer

From the silver jsonl's new `ud` field, derive slot tags via the
existing token_tags-equivalent mapping (UPOS+FEATS → slots — reuse
kirciuokle.disambiguate.token_tags by constructing its Token input or
replicating its mapping; keep DET→PRON / AUX→VERB conventions).
Store per token: `udpipe_slots` alongside the existing joint/tagger
POS opinions.

## 3. calibrate.py: three-voter POS strata

POS agreement patterns over {joint, tagger, udpipe} — compare at the
SLOT projection level (that is what drives accentuation variant
matching; keep full-label accuracy as a secondary column where the
label spaces allow). Calibrate on the ALKSNIS gold test as before.
For the ALKSNIS corpus the udpipe layer needs UDPipe tags for those
sentences: run them through the LINDAT UDPipe REST API once
(scripts/accent_text.py holds the endpoint + model constants; ~684
sentences, throttle ≥1s, cache to data/teacher/alksnis-udpipe.jsonl,
resumable). Emit the new pos-strata.json.

## 4. label.py: POS acceptance at slot level

Accept POS for a token when its three-voter pattern clears
`--min-pos-accuracy` (default 0.95) — the accepted label is the JOINT
model's full label (it's in the checkpoint's label space; the other
voters act as verifiers). Also ensure NO-STRESS agreement flows: where
silver leaves a token unmarked AND the joint model predicts no-stress,
emit an accepted no-stress target (this stratum must appear in the
accent calibration table too — foreign-name-rich corpora depend on it).

## Pass criteria

1. Silver round-trip: a 30-line smoke corpus through build_silver_truth
   shows the `ud` field populated; an OLD silver file (chrestomatija's)
   still loads in collect_layers with udpipe layer absent.
2. ALKSNIS UDPipe cache built (resumable; paste 3 sample rows).
3. calibrate.py prints the three-voter POS table on ALKSNIS gold —
   paste it in full. The all-three-agree stratum must exceed the
   two-voter 85.9% (expect ≥93% at slot level; report whatever it is).
4. label.py smoke on the chrestomatija layers (--allow flags as
   needed): report POS coverage + purity at 0.95 — nonzero coverage
   expected if criterion 3's stratum clears the bar; if it does not
   clear 0.95, report the coverage at 0.90 too and say so plainly.

Do not commit.
