# No-Dictionary Pipeline Evaluation

## Corpus
- corpus: `local/accentuator/data/eval/lrt-smoke.txt`
- silver: `local/accentuator/data/eval/lrt-smoke-silver.jsonl`
- generated DB: `local/accentuator/data/generated.sqlite`
- stress checkpoint: `local/accentuator/data/stress_nn2/stress_nn2.pt`
- silver tokens: 5,020
- aligned tokens: 5,016
- skipped silver tokens: 4 (0.08%)
- skipped tagger tokens: 2
- label vocabulary: 875

## Pipelines

Token exact and position are measured over answered tokens. Type exact is measured over answered first-seen word types.

| pipeline | min confidence | answered | token exact | token position | type exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| nodict | 0 | 4,932/5,016 (98.3%) | 1,269/4,932 (25.7%) | 2,198/4,932 (44.6%) | 419/2,309 (18.1%) |
| nodict | 0.9 | 112/5,016 (2.2%) | 101/112 (90.2%) | 102/112 (91.1%) | 9/12 (75.0%) |
| nodict-uncond | 0 | 4,932/5,016 (98.3%) | 1,286/4,932 (26.1%) | 2,224/4,932 (45.1%) | 424/2,309 (18.4%) |
| liepa | n/a | 680/5,016 (13.6%) | 640/680 (94.1%) | 656/680 (96.5%) | 295/325 (90.8%) |
| dict | n/a | 3,822/5,016 (76.2%) | 3,653/3,822 (95.6%) | 3,656/3,822 (95.7%) | 1,479/1,536 (96.3%) |

## Nodict Disagreements

| word | silver | nodict | label |
| --- | --- | --- | --- |
| italijoje | itãlijoje | italijóje | dkt., aukšt., mot. g., vns., viet. |
| lietuvos | lietuvõs | lietuvós | dkt., aukšt., mot. g., vns., kilm. |
| krepšinio | krepšìnio | krepšinió | dkt., aukšt., vyr. g., vns., kilm. |
| rinktinė | rinktìnė | rinktinė́ | dkt., aukšt., mot. g., vns., vard. |
| viso | vìso | visó | (empty) |
| yra | yrà | ýra | vksm., mot. g., vns., gal., es. l., neveik. r., dlv. |
| žaidusi | žaĩdusi | žaidùsi | vksm., mot. g., vns., vard., būt. k. l., būt. d. l., veik. r., dlv. |
| rungtynių | rungtỹnių | rungtýnių | dkt., aukšt., mot. g., dgs., kilm. |
| tačiau | tačiaũ | tačiaú | jng. |
| bolonijoje | bolònijoje | bolónijoje | dkt., aukšt., mot. g., vns., viet. |
| pirmą | pìrmą | pirmą̃ | sktv., vns., gal. |
| kartą | kar̃tą | kartą̃ | dkt., aukšt., vyr. g., vns., gal. |
| istorijoje | istòrijoje | istor̃ijoje | dkt., aukšt., mot. g., vns., viet. |
| rašo | rãšo | rašó | vksm., mot. g., vns., gal., es. l., neveik. r., dlv. |
| asociacija | asociãcija | asociácija | dkt., aukšt., mot. g., vns., vard. |
| krepšinis | krepšìnis | krepšiñis | dkt., aukšt., vyr. g., vns., vard. |
| įdomu | įdomù | į̃domu | bdv., bev. g. |
| būtent | bū́tent | būteñt | jng. |
| prieš | priẽš | priéš | prl. |
| italijos | itãlijos | italijós | dkt., aukšt., mot. g., vns., kilm. |
| krepšininkus | krẽpšininkus | krepšiñinkus | dkt. tikr., vyr. g., dgs., gal. |
| savo | sàvo | savó | įv., aukšt., mot. g., dgs., kilm. |
| pirmąją | pìrmąją | pirmąją̃ | sktv., mot. g., gal. |
| oficialią | oficiãlią | oficiálią | bdv., aukšt., mot. g., vns., gal. |
| pergalę | pérgalę | pergalę́ | dkt., aukšt., mot. g., vns., gal. |
