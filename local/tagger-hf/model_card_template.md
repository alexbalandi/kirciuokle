---
language: lt
license: cc-by-sa-4.0
base_model: EMBEDDIA/litlat-bert
pipeline_tag: token-classification
datasets:
- MATAS v3.0
- UD_Lithuanian-ALKSNIS
tags:
- lithuanian
- morphology
- pos
- upos
- feats
- accentuation
widget:
- text: "Čia yra graži lietuviška diena."
- text: "Vilniuje studentai skaito naują tekstą."
---

# Lithuanian UPOS/FEATS Tagger

## Summary

This is a Lithuanian token-classification tagger for UPOS and morphological
FEATS, packaged from run `{{run_name}}` on {{export_date}}. It is intended for
Lithuanian accentuation pipelines and for general Lithuanian morphosyntactic
tagging where a compact Hugging Face or ONNX artifact is useful.

- Base encoder: `{{base_model}}`
- Head: `{{head}}`
- Subword pooling: `{{pooling}}`
- Max length: `{{max_length}}`
- Release model id placeholder: `{{model_id}}`

## Intended Use

Use this model as a morphology tagger in Lithuanian accentuation systems,
especially when the downstream decision logic needs UPOS plus Case, Gender,
Number, Tense, Person, Voice, Degree, VerbForm, Mood, or Reflex. It can also be
used as a general Lithuanian UPOS/FEATS tagger, with the convention notes below.

## Convention Deviations From Strict UD

- DET -> PRON: Lithuanian traditional grammar and the VDU dictionary convention
  do not use a separate determiner category.
- AUX -> VERB: VDU morphology treats auxiliaries as verbs; this avoids
  contradictory labels for the accentuation pipeline.
- Slots-only FEATS: scoring and exported labels focus on the FEATS slots used by
  downstream disambiguation.
- Form-as-lemma except `yra` -> `būti`: the model is a tagger, not a
  lemmatizer; the copular exception preserves the accentuation pipeline's
  important lemma distinction.

## Training Data & Lineage

Training uses MATAS v3.0 and UD_Lithuanian-ALKSNIS. The preparation pipeline
applies deterministic XPOS-to-UD FEATS repair from Jablonskis tags and
constrained self-fill for missing Number and Person where configured. The clean
release lineage is explicitly UDPipe-free; UDPipe is used as a production
baseline and benchmark reference, not as a teacher for this released model.

## Benchmarks

### Final Run Metrics

{{final_metrics}}

### Official CoNLL-18 UD Table

```text
{{official_conll18_table}}
```

### VDU-convention Projection

```text
{{vdu_conll18_table}}
```

### Benchmark Facts

- Benchmark slots: {{bench_slots}}

{{bench_json_table}}

### Speed

{{speed_table}}

### Comparisons

{{comparison_table}}

### Additional Facts

{{facts_table}}

## How To Use

### Transformers

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "{{model_id}}"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_id)

words = ["Čia", "yra", "tekstas", "."]
encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
outputs = model(**encoded)
predicted_ids = outputs.logits.argmax(dim=-1)
```

Use `head_config.json` and `labels.json` from the model folder to map subword
predictions back to word-level `UPOS|FEATS` labels with the configured pooling.

### ONNX Runtime

```python
import onnxruntime as ort
from transformers import AutoTokenizer

model_dir = "./onnx"
tokenizer = AutoTokenizer.from_pretrained(".", use_fast=True)
session = ort.InferenceSession(f"{model_dir}/model_quantized.onnx")

words = ["Čia", "yra", "tekstas", "."]
encoded = tokenizer(words, is_split_into_words=True, return_tensors="np")
inputs = {item.name: encoded[item.name] for item in session.get_inputs()}
outputs = session.run(None, inputs)
```

### UDPipe-protocol Sidecar

```sh
docker build -f local/tagger-hf/Dockerfile -t kirciuokle-tagger-hf .
docker run --rm -p 8001:8001 -v "$PWD/onnx:/model:ro" kirciuokle-tagger-hf
```

The sidecar exposes `POST /process` with UDPipe-compatible form fields and
returns `{"result": "<conllu>"}`.

## Limitations

This model does not produce dependency parses and is not a general lemmatizer.
Tokenization mismatches against UD gold data count against official CoNLL-18
word and morphology scores. Labels follow the VDU-oriented convention described
above, so strict UD consumers should account for DET/PRON and AUX/VERB
differences.

## Attributions and Citations

- VDU kirčiuoklė, Vytautas Magnus University Centre of Computational
  Linguistics: source service for accentuation dictionary responses.
- kirtis.info: original inspiration, using the same underlying VDU data.
- MATAS v3.0, Rimkutė, Bielinskienė, Boizou, Dadurkevičius, Kovalevskaitė, and
  Utka, CLARIN-LT hdl:20.500.11821/61, CC BY 4.0.
- UD_Lithuanian-ALKSNIS, Vytautas Magnus University via Universal Dependencies,
  CC BY-SA 4.0.
- EMBEDDIA/litlat-bert by Ulčar and Robnik-Šikonja, EMBEDDIA project,
  CC BY-SA 4.0.
- VSSA-SDSA/LT-MLKM-modernBERT, Apache-2.0, evaluated as an encoder candidate.
- UDPipe 2 by Milan Straka, UFAL, via LINDAT-CLARIN; code MPL-2.0, models
  CC BY-NC-SA. Used as a production baseline and benchmark reference.
- Stanza by Stanford NLP, Apache-2.0, benchmarked.
- LIEPA `phonology_engine`, evaluated offline as an accentuation candidate.
- hermitdave/FrequencyWords, MIT, derived from OpenSubtitles data.
- Lithuanian Wikipedia text, CC BY-SA, used for evaluation corpus text.
- CoNLL 2018 shared task evaluation script, downloaded at runtime rather than
  vendored.

Released weights are CC BY-SA 4.0, inherited from EMBEDDIA/litlat-bert and
UD_Lithuanian-ALKSNIS. Repository code is public domain under The Unlicense.
