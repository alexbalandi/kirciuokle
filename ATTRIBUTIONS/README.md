# Attributions

This page collects the data, models, services, and benchmark tools this
project builds on. Each entry names the resource, its license or terms as
understood by this project, how it is used, the attribution line to reproduce,
and a citation when one is standard.

## VDU kirčiuoklė

- What it is: Lithuanian accentuation dictionary service from Vytautas Magnus
  University, Centre of Computational Linguistics, available at kalbu.vdu.lt.
- License or terms: upstream service/data terms remain with VDU.
- How this project uses it: the Worker and local tools consume the public web
  API for accent placement and variant morphology. The D1 and SQLite
  dictionaries are local caches of VDU responses and are not redistributed with
  this repository.
- Attribution line: Accentuation data from VDU kirčiuoklė, Vytautas Magnus
  University Centre of Computational Linguistics, https://kalbu.vdu.lt/.
- Citation: VDU kirčiuoklė, Vytautas Magnus University Centre of Computational
  Linguistics, https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/.

## kirtis.info

- What it is: the original public Lithuanian accentuation site that inspired
  this project, using the same underlying VDU data.
- License or terms: upstream site/data terms remain with kirtis.info and VDU.
- How this project uses it: inspiration and comparison point only; this
  repository does not redistribute its data.
- Attribution line: Inspired by kirtis.info, https://kirtis.info/.
- Citation: kirtis.info, https://kirtis.info/.

## MATAS v3.0

- What it is: a Lithuanian morphologically annotated corpus distributed through
  CLARIN-LT.
- License: CC BY 4.0.
- How this project uses it: tagger training data. The local preparation pipeline
  reconstructs missing UD FEATS from Jablonskis XPOS, applies
  constrained-decoding self-fills for Number and Person where needed, and
  normalizes DET to PRON and AUX to VERB to match the VDU convention used by the
  accentuation pipeline.
- Attribution line: MATAS v3.0, Rimkutė, Bielinskienė, Boizou, Dadurkevičius,
  Kovalevskaitė, and Utka, CLARIN-LT hdl:20.500.11821/61, CC BY 4.0.
- Citation:

```bibtex
@misc{matas3,
  title = {MATAS v3.0},
  author = {Rimkutė, Erika and Bielinskienė, Agnė and Boizou, Loic and Dadurkevičius, Virginijus and Kovalevskaitė, Jolanta and Utka, Andrius},
  publisher = {CLARIN-LT},
  howpublished = {\url{hdl:20.500.11821/61}},
  license = {CC BY 4.0}
}
```

## UD_Lithuanian-ALKSNIS

- What it is: the Lithuanian ALKSNIS Universal Dependencies treebank from VDU,
  distributed through Universal Dependencies.
- License: CC BY-SA 4.0.
- How this project uses it: gold development/test data and optional training
  data for the Lithuanian tagger.
- Attribution line: UD_Lithuanian-ALKSNIS, Vytautas Magnus University, via
  Universal Dependencies, CC BY-SA 4.0.
- Citation:

```bibtex
@misc{ud_lithuanian_alksnis,
  title = {UD Lithuanian ALKSNIS},
  author = {{Universal Dependencies contributors}},
  howpublished = {\url{https://github.com/UniversalDependencies/UD_Lithuanian-ALKSNIS}},
  license = {CC BY-SA 4.0}
}
```

## EMBEDDIA/litlat-bert

- What it is: a Lithuanian-Latvian-English BERT encoder released by the
  EMBEDDIA project.
- License: CC BY-SA 4.0.
- How this project uses it: base encoder for the released tagger. Because the
  base model and ALKSNIS data are share-alike, the fine-tuned tagger weights
  are released as CC BY-SA 4.0.
- Attribution line: Base encoder EMBEDDIA/litlat-bert by Ulčar and
  Robnik-Šikonja, EMBEDDIA project, CC BY-SA 4.0.
- Citation:

```bibtex
@misc{embeddia_litlat_bert,
  title = {EMBEDDIA/litlat-bert},
  author = {Ulčar, Matej and Robnik-Šikonja, Marko},
  howpublished = {\url{https://huggingface.co/EMBEDDIA/litlat-bert}},
  note = {EMBEDDIA project},
  license = {CC BY-SA 4.0}
}
```

## VSSA-SDSA/LT-MLKM-modernBERT

- What it is: a Lithuanian ModernBERT encoder candidate.
- License: Apache-2.0.
- How this project uses it: evaluated as an encoder candidate. It is not the
  base of the released clean model.
- Attribution line: Encoder candidate VSSA-SDSA/LT-MLKM-modernBERT, Apache-2.0.
- Citation: VSSA-SDSA/LT-MLKM-modernBERT,
  https://huggingface.co/VSSA-SDSA/LT-MLKM-modernBERT.

## UDPipe 2 / LINDAT-CLARIN

- What it is: UDPipe 2 by Straka/UFAL, served through LINDAT-CLARIN.
- License: UDPipe 2 code is MPL-2.0; UDPipe models are CC BY-NC-SA.
- How this project uses it: production tagger through the public REST service,
  the quality bar for benchmarks, and the teacher for one internal model
  variant named teacher2. That teacher2 lineage is not part of the released
  clean model.
- Attribution line: Tagging by UDPipe 2, Milan Straka, UFAL, via
  LINDAT-CLARIN; model terms CC BY-NC-SA.
- Citation:

```bibtex
@inproceedings{straka-2018-udpipe-2,
  title = {UDPipe 2.0 Prototype at CoNLL 2018 UD Shared Task},
  author = {Straka, Milan},
  booktitle = {Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {197--207}
}
```

## Stanza

- What it is: Stanford NLP's Python natural language processing toolkit.
- License: Apache-2.0.
- How this project uses it: benchmarked as an alternate Lithuanian tagger.
- Attribution line: Stanza by Stanford NLP, Apache-2.0.
- Citation:

```bibtex
@inproceedings{qi-etal-2020-stanza,
  title = {Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
  author = {Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year = {2020},
  publisher = {Association for Computational Linguistics},
  pages = {101--108}
}
```

## LIEPA phonology_engine

- What it is: a Lithuanian phonology/accentuation engine from the LIEPA
  ecosystem.
- License or terms: upstream terms remain with the LIEPA project.
- How this project uses it: evaluated offline as an accentuation-engine
  candidate.
- Attribution line: Offline accentuation candidate from LIEPA phonology_engine.
- Citation: LIEPA project resources, https://liepa.rastija.lt/.

## hermitdave/FrequencyWords

- What it is: frequency word lists derived from OpenSubtitles data.
- License: MIT for the repository; underlying OpenSubtitles data keeps its own
  terms.
- How this project uses it: seed wordlist for dictionary warming.
- Attribution line: Frequency word list from hermitdave/FrequencyWords, MIT,
  derived from OpenSubtitles data.
- Citation: hermitdave/FrequencyWords,
  https://github.com/hermitdave/FrequencyWords.

## Lithuanian Wikipedia

- What it is: Lithuanian Wikipedia text.
- License: CC BY-SA.
- How this project uses it: evaluation corpus text.
- Attribution line: Evaluation text includes Lithuanian Wikipedia material,
  CC BY-SA, Wikimedia Foundation contributors.
- Citation: Lithuanian Wikipedia, https://lt.wikipedia.org/.

## CoNLL 2018 Shared Task evaluation script

- What it is: the official Universal Dependencies CoNLL 2018 evaluation script.
- License or terms: upstream script terms remain with Universal Dependencies /
  CoNLL 2018 Shared Task maintainers.
- How this project uses it: downloaded at runtime into `data/raw/` and executed
  programmatically for official UD metrics. It is not vendored in this
  repository.
- Attribution line: Evaluation with the official CoNLL 2018 UD shared task
  script, https://universaldependencies.org/conll18/conll18_ud_eval.py.
- Citation:

```bibtex
@misc{conll18_ud_eval,
  title = {CoNLL 2018 Shared Task Universal Dependencies Evaluation Script},
  author = {{CoNLL 2018 Shared Task organizers}},
  howpublished = {\url{https://universaldependencies.org/conll18/conll18_ud_eval.py}},
  year = {2018}
}
```

## Scope Note

Repository code is public domain under The Unlicense. Data, services, corpora,
and models listed above keep their own licenses and terms. Released tagger
weights are CC BY-SA 4.0, inherited from EMBEDDIA/litlat-bert and
UD_Lithuanian-ALKSNIS.
