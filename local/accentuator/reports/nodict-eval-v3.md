# No-Dictionary Pipeline Evaluation

## Corpus
- corpus: `local/accentuator/data/eval/lrt-corpus.txt`
- silver: `local/accentuator/data/eval/lrt-silver.jsonl`
- generated DB: `local/accentuator/data/generated.sqlite`
- stress checkpoint: `local/accentuator/data/stress_nn3/stress_nn3.pt`
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
| nodict | 0 | 37,718/37,718 (100.0%) | 30,869/37,718 (81.8%) | 31,498/37,718 (83.5%) | 8,762/11,281 (77.7%) |
| nodict | 0.9 | 25,235/37,718 (66.9%) | 22,176/25,235 (87.9%) | 21,862/25,235 (86.6%) | 7,758/9,132 (85.0%) |
| nodict-uncond | 0 | 37,718/37,718 (100.0%) | 27,903/37,718 (74.0%) | 28,420/37,718 (75.3%) | 8,663/11,281 (76.8%) |
| liepa | n/a | 30,011/37,718 (79.6%) | 28,724/30,011 (95.7%) | 29,386/30,011 (97.9%) | 9,697/10,294 (94.2%) |
| dict | n/a | 28,978/37,718 (76.8%) | 27,411/28,978 (94.6%) | 27,489/28,978 (94.9%) | 6,329/6,585 (96.1%) |

## Pipelines (Audited Silver)

| pipeline | min confidence | answered | token exact | token position | type exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| nodict | 0 | 36,717/36,717 (100.0%) | 30,650/36,717 (83.5%) | 31,870/36,717 (86.8%) | 8,812/11,010 (80.0%) |
| nodict | 0.9 | 24,506/36,717 (66.7%) | 21,806/24,506 (89.0%) | 22,038/24,506 (89.9%) | 7,797/8,990 (86.7%) |
| nodict-uncond | 0 | 36,717/36,717 (100.0%) | 27,761/36,717 (75.6%) | 28,807/36,717 (78.5%) | 8,672/11,010 (78.8%) |
| liepa | n/a | 29,970/36,717 (81.6%) | 28,819/29,970 (96.2%) | 29,491/29,970 (98.4%) | 9,742/10,277 (94.8%) |
| dict | n/a | 28,956/36,717 (78.9%) | 27,804/28,956 (96.0%) | 27,886/28,956 (96.3%) | 6,484/6,580 (98.5%) |

## Audit Diagnostics

| pipeline | min confidence | excluded tokens | foreign-unmarked | desired unmarked/abstained |
| --- | ---: | ---: | ---: | ---: |
| nodict | 0 | 58 | 943 | 592/943 (62.8%) |
| nodict | 0.9 | 58 | 943 | 796/943 (84.4%) |
| nodict-uncond | 0 | 58 | 943 | 707/943 (75.0%) |
| liepa | n/a | 58 | 943 | 908/943 (96.3%) |
| dict | n/a | 58 | 943 | 943/943 (100.0%) |

## Nodict Disagreements

| word | silver | nodict | label |
| --- | --- | --- | --- |
| šarvinių | šarvìnių | šar̃vinių | bdv., aukšt., mot. g., dgs., kilm. |
| netaikydama | netáikydama | netaikýdama | vksm., mot. g., vns., gal., neveik. r., dlv., reik. |
| agentūra | agentūrà | agentūrã | dkt., aukšt., mot. g., vns., vard. |
| safariland | safariland | safarilañd | dkt. |
| taikyti | táikyti | taikýti | vksm. |
| veikimo | veikìmo | veĩkimo | dkt., aukšt., vyr. g., vns., kilm. |
| kiekviena | kiekvienà | kiekvíena | įv., aukšt., mot. g., vns., vard. |
| narė | narė̃ | narė́ | dkt., aukšt., mot. g., vns., vard. |
| mano | màno | mãno | vksm., es. l., 3 asm. |
| būtinas | bū́tinas | būtinãs | bdv., aukšt., mot. g., dgs., gal. |
| savo | sàvo | sãvo | įv., dvisk., kilm. |
| susijusiems | susìjusiems | susijùsiems | vksm., vyr. g., dgs., naud., būt. k. l., būt. d. l., veik. r., dlv. |
| taikymu | táikymu | taĩkymu | dkt., aukšt., vyr. g., vns., įnag. |
| siekiama | siekiamà | síekiama | jstk., nelygin., bev. g. |
| esminį | esmìnį | èsminį | bdv., aukšt., vyr. g., vns., gal. |
| galią | gãlią | gal̃ią | dkt., aukšt., mot. g., vns., gal. |
| regione | regionè | regiòne | dkt., aukšt., vyr. g., vns., viet. |
| poreikį | póreikį | porèikį | dkt., aukšt., vyr. g., vns., gal. |
| aprūpinti | aprū́pinti | aprū̃pinti | vksm. |
| šarvinėmis | šarvìnėmis | šar̃vinėmis | bdv., aukšt., mot. g., dgs., įnag. |
| gyvybes | gyvýbes | gyvybès | dkt. tikr., mot. g., dgs., gal. |
| teikime | teikimè | teĩkime | dkt., aukšt., vyr. g., vns., viet. |
| itin | ìtin | itiñ | prv. |
| iv | iv | ìv | sktv., pagr. |
| jav | jav | jãv | dkt. |
