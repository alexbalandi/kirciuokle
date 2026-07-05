# No-Dictionary Pipeline Evaluation

## Corpus
- corpus: `local/accentuator/data/eval/lrt-corpus.txt`
- silver: `local/accentuator/data/eval/lrt-silver.jsonl`
- generated DB: `local/accentuator/data/generated.sqlite`
- stress checkpoint: `local/accentuator/data/stress_nn2/stress_nn2.pt`
- silver tokens: 37,736
- aligned tokens: 37,718
- skipped silver tokens: 18 (0.05%)
- skipped tagger tokens: 9
- label vocabulary: 875
- audit overlay: `local/accentuator/data/eval/lrt-silver-audit.json` (453 entries)

## Pipelines (Raw Silver)

Token exact and position are measured over answered tokens. Type exact is measured over answered first-seen word types.

| pipeline | min confidence | answered | token exact | token position | type exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| nodict | 0 | 37,195/37,718 (98.6%) | 29,426/37,195 (79.1%) | 30,342/37,195 (81.6%) | 8,545/11,238 (76.0%) |
| nodict | 0.9 | 21,857/37,718 (57.9%) | 18,968/21,857 (86.8%) | 19,178/21,857 (87.7%) | 7,209/8,448 (85.3%) |
| nodict-uncond | 0 | 37,195/37,718 (98.6%) | 27,394/37,195 (73.6%) | 29,148/37,195 (78.4%) | 8,472/11,238 (75.4%) |
| liepa | n/a | 30,011/37,718 (79.6%) | 28,724/30,011 (95.7%) | 29,386/30,011 (97.9%) | 9,697/10,294 (94.2%) |
| dict | n/a | 29,007/37,718 (76.9%) | 27,414/29,007 (94.5%) | 27,495/29,007 (94.8%) | 6,330/6,596 (96.0%) |

## Pipelines (Audited Silver)

| pipeline | min confidence | answered | token exact | token position | type exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| nodict | 0 | 36,717/36,717 (100.0%) | 29,775/36,717 (81.1%) | 30,688/36,717 (83.6%) | 8,669/11,010 (78.7%) |
| nodict | 0.9 | 21,672/36,717 (59.0%) | 19,149/21,672 (88.4%) | 19,357/21,672 (89.3%) | 7,291/8,359 (87.2%) |
| nodict-uncond | 0 | 36,717/36,717 (100.0%) | 27,745/36,717 (75.6%) | 29,497/36,717 (80.3%) | 8,603/11,010 (78.1%) |
| liepa | n/a | 29,970/36,717 (81.6%) | 28,819/29,970 (96.2%) | 29,491/29,970 (98.4%) | 9,742/10,277 (94.8%) |
| dict | n/a | 28,989/36,717 (79.0%) | 27,807/28,989 (95.9%) | 27,892/28,989 (96.2%) | 6,485/6,593 (98.4%) |

## Audit Diagnostics

| pipeline | min confidence | excluded tokens | foreign-unmarked | desired unmarked/abstained |
| --- | ---: | ---: | ---: | ---: |
| nodict | 0 | 58 | 943 | 523/943 (55.5%) |
| nodict | 0.9 | 58 | 943 | 779/943 (82.6%) |
| nodict-uncond | 0 | 58 | 943 | 523/943 (55.5%) |
| liepa | n/a | 58 | 943 | 908/943 (96.3%) |
| dict | n/a | 58 | 943 | 943/943 (100.0%) |

## Nodict Disagreements

| word | silver | nodict | label |
| --- | --- | --- | --- |
| apsaugos | apsaugõs | apsaũgos | dkt., aukšt., mot. g., vns., kilm. |
| lietuvos | lietuvõs | lietùvos | dkt., aukšt., mot. g., vns., kilm. |
| šarvinių | šarvìnių | šar̃vinių | bdv., aukšt., mot. g., dgs., kilm. |
| netaikydama | netáikydama | netaikýdama | vksm., mot. g., vns., gal., neveik. r., dlv., reik. |
| resursų | resùrsų | resursų̃ | dkt., aukšt., vyr. g., dgs., kilm. |
| safariland | safariland | safarilañd | dkt. |
| lithuania | lithuania | lithùania | dkt. |
| kreipėsi | kreĩpėsi | kreipė́si | vksm., būt. k. l., 3 asm. |
| leidimą | leidìmą | léidimą | dkt., aukšt., vyr. g., vns., gal. |
| teigimu | teigimù | teĩgimu | dkt., aukšt., vyr. g., vns., įnag. |
| taikyti | táikyti | taikýti | vksm. |
| veikimo | veikìmo | veĩkimo | dkt., aukšt., vyr. g., vns., kilm. |
| kiekviena | kiekvienà | kiekvíena | įv., aukšt., mot. g., vns., vard. |
| narė | narė̃ | narė́ | dkt., aukšt., mot. g., vns., vard. |
| mano | màno | mãno | vksm., es. l., 3 asm. |
| savo | sàvo | savõ | įv., dvisk., kilm. |
| karinės | karìnės | karinė̃s | bdv., aukšt., mot. g., vns., kilm. |
| paskirties | paskirtiẽs | paskir̃ties | dkt., aukšt., mot. g., vns., kilm. |
| medžiagų | mẽdžiagų | medžiagų̃ | dkt., aukšt., mot. g., dgs., kilm. |
| išimties | išimtiẽs | išimtìes | dkt., aukšt., mot. g., vns., kilm. |
| taikymu | táikymu | taĩkymu | dkt., aukšt., vyr. g., vns., įnag. |
| siekiama | siekiamà | síekiama | jstk., nelygin., bev. g. |
| esminį | esmìnį | èsminį | bdv., aukšt., vyr. g., vns., gal. |
| galią | gãlią | gal̃ią | dkt., aukšt., mot. g., vns., gal. |
| poreikį | póreikį | porèikį | dkt., aukšt., vyr. g., vns., gal. |
