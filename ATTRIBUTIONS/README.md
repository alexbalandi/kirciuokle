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

- What it is: a Lithuanian accentuation/phonology engine — the Python package
  `phonology_engine` (BSD license) wrapping the native text-processing and
  stress components of the LIEPA speech synthesizer.
- License or terms: the package is BSD-licensed; the wrapped native
  components originate from the state-funded LIEPA project.
- How this project uses it: the open accentuator's lowest-confidence
  **guesser tier** (`local/accentuator/guess_uncovered.py`) for words no open
  dictionary source covers. Benchmarked against the VDU cache: 87.9%
  exact-variant / 95.3% stress-position agreement on dictionary-gap words.
  Guesses live in a separate artifact with `liepa-guess` provenance and are
  never merged into the zero-disagreement main dictionary.
- Attribution line: Guess-tier accentuation by phonology_engine (BSD),
  wrapping LIEPA speech-synthesizer components, https://github.com/aleksas/phonology_engine.
- Citation: LIEPA project resources, https://liepa.rastija.lt/;
  phonology_engine, https://pypi.org/project/phonology-engine/.

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

## VLKK vardai database (vardai.vlkk.lt)

- What it is: the State Commission of the Lithuanian Language's database of
  recommended given names with accented, declined, stress-classed forms.
- License or terms: normative recommendations of a state institution; used
  as the project's declared normative authority for given names, fetched
  politely and cited.
- How this project uses it: `local/accentuator/fetch_vlkk_names.py` collects
  accented nominatives from the letter indexes and, for names attested in the
  project's word lists, the kirčiuotė and singular paradigm from per-name
  pages. Given names in the generated dictionary carry
  `open-accentuator:vlkk-vardai:` provenance and take precedence over
  Wiktionary name entries.
- Attribution line: Given-name accentuation from the VLKK names database,
  Valstybinė lietuvių kalbos komisija, https://vardai.vlkk.lt/.
- Citation: Vardai VLKK, Valstybinė lietuvių kalbos komisija,
  https://vardai.vlkk.lt/.

## Wiktionary via kaikki.org (wiktextract)

- What it is: machine-readable English Wiktionary extractions by Tatu Ylonen's
  wiktextract project, served at kaikki.org; the Lithuanian entries carry
  accent-class tags and accented inflection tables.
- License: Wiktionary text is dual-licensed CC BY-SA 4.0 and GFDL; wiktextract
  is open source.
- How this project uses it: the open accentuator (`local/accentuator/`) reads
  accent classes, accented headwords, and inflection tables from the kaikki
  Lithuanian dump as its lexical facts. Known bad entries are excluded (never
  corrected from closed sources) via `local/accentuator/parity_vetoes.json`.
- Attribution line: Lexical accent data from Wiktionary contributors via
  kaikki.org (wiktextract by Tatu Ylonen), CC BY-SA 4.0.
- Citation:

```bibtex
@inproceedings{ylonen-2022-wiktextract,
  title = {Wiktextract: Wiktionary as Machine-Readable Structured Data},
  author = {Ylonen, Tatu},
  booktitle = {Proceedings of the 13th Language Resources and Evaluation Conference},
  year = {2022},
  pages = {1317--1325}
}
```

## Kazlauskienė, Raškinis, Norkevičius, Vaičiūnas (2010), VDU accentuation monograph

- What it is: "Automatinis lietuvių kalbos žodžių skiemenavimas, kirčiavimas,
  transkribavimas" (VDU, 2010) — the published, freely downloadable
  description of the VDU kirčiuoklė's algorithms and resource structure.
- License or terms: published scholarship; used as textbook knowledge. No
  VDU data files are used — only the published rules and the Appendix C
  suffix table.
- How this project uses it: the open accentuator's suffix-derivation module
  (`local/accentuator/suffix_rules.py`) implements the self-accented suffix
  rules (§3.2.4, §3.3.7, Appendix C); inflection endings are induced from
  Wiktionary paradigms, not taken from VDU.
- Attribution line: Suffix accentuation rules after Kazlauskienė, Raškinis,
  Norkevičius, Vaičiūnas (2010), Vytauto Didžiojo universitetas.
- Citation:

```bibtex
@book{kazlauskiene2010automatinis,
  title = {Automatinis lietuvi{\ų} kalbos {\ž}od{\ž}i{\ų} skiemenavimas, kir{\č}iavimas, transkribavimas},
  author = {Kazlauskien{\.e}, Asta and Ra{\š}kinis, Gailius and Norkevi{\č}ius, Giedrius and Vai{\č}i{\=u}nas, Airenas},
  publisher = {Vytauto Did{\ž}iojo universiteto leidykla},
  year = {2010},
  url = {http://donelaitis.vdu.lt/lkk/pdf/alka_SKT.pdf}
}
```

## Kushnir (2019), Prosodic Patterns in Lithuanian Morphology

- What it is: Yuriy Kushnir's doctoral dissertation (Leipzig University, 2019),
  a formal analysis of Lithuanian accentuation covering nominal accent classes,
  Saussure's Law, dominance, and verbal prosody.
- License or terms: published scholarship; used as textbook knowledge.
- How this project uses it: the open accentuator implements its conditioning of
  verbal stress retraction (§4.4.2, §4.4.5: 1/2sg stress retracts to the prefix
  exactly when the tense's root allomorph is weak — past theme -ė in primary
  verbs; -o presents and -yti pasts are always strong) and its confirmation
  that future 1/2sg and conditional forms never undergo Saussure's shift.
- Attribution line: Verbal stress-retraction conditioning after Kushnir (2019),
  Prosodic Patterns in Lithuanian Morphology, Leipzig University.
- Citation:

```bibtex
@phdthesis{kushnir2019prosodic,
  title = {Prosodic Patterns in Lithuanian Morphology},
  author = {Kushnir, Yuriy},
  school = {Universit{\"a}t Leipzig},
  year = {2019},
  url = {https://yuriykushnir.com/documents/Y_Kushnir_Dissertation.pdf}
}
```

## Lithuanian hunspell dictionary (ispell-lt)

- What it is: the Lithuanian hunspell/myspell affix and dictionary files (`.aff`
  + `.dic`) originating from the ispell-lt project, obtained through the
  `wooorm/dictionaries` distribution.
- License: BSD-3-Clause, Copyright (c) 2000-2020 Albertas Agejevas and
  contributors.
- How this project uses it: shipped as `public/lt.aff` and `public/lt.dic`
  (gitignored build artifacts fetched by
  `scripts/regenerate_spellcheck_dicts.py`) and loaded by the client-side
  spellcheck web worker as the authoritative "is this a valid Lithuanian word"
  morphology check, so real inflected text is not false-flagged. It is not used
  for accentuation.
- Attribution line: Lithuanian spellcheck dictionary from ispell-lt (Albertas
  Agejevas and contributors) via wooorm/dictionaries, BSD-3-Clause.
- Citation: dictionaries/lt, https://github.com/wooorm/dictionaries.

## hunspell-asm and Hunspell

- What it is: `hunspell-asm` (the Hunspell spell checker compiled to
  WebAssembly with a JavaScript wrapper) and the underlying Hunspell engine.
- License: hunspell-asm is MIT; Hunspell is tri-licensed GPL-2.0 / LGPL-2.1 /
  MPL-1.1 and is used here as a dynamically-loaded library under its
  LGPL/MPL terms.
- How this project uses it: bundled into the spellcheck web worker to run the
  Lithuanian hunspell dictionary entirely in the browser, so nothing leaves the
  user's device. The published ESM build calls CJS deps as if callable, which
  breaks under bundlers, so the CJS build is bundled instead (see
  `vite.config.ts`).
- Attribution line: In-browser spell checking by hunspell-asm (MIT) wrapping
  Hunspell (GPL/LGPL/MPL), https://github.com/kwonoj/hunspell-asm.
- Citation: hunspell-asm, https://github.com/kwonoj/hunspell-asm; Hunspell,
  https://hunspell.github.io/.

## Scope Note

Repository code is public domain under The Unlicense. Data, services, corpora,
and models listed above keep their own licenses and terms. Released tagger
weights are CC BY-SA 4.0, inherited from EMBEDDIA/litlat-bert and
UD_Lithuanian-ALKSNIS.
