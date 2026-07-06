# Chrestomatija Gold Evaluation

## Corpus
- gold: `local/accentuator/data/eval/chrestomatija-gold.jsonl`
- generated DB: `local/accentuator/data/generated.sqlite`
- joint checkpoint: `local/accentuator/joint/checkpoints/joint_v3.best.pt`
- sentence cap: none
- extracted sentences scored: 2,969
- word tokens: 43,209
- word types: 14,946
- stress marks: 43,074
- pages: 189 (11-199)

## Metrics

Token exact is measured over all gold word tokens; an unmarked gold token counts exact when the system leaves it unmarked or abstains. Token position is measured over answered tokens. Sentence sequence accuracy is exact only when every word token in the sentence is exact.

| system | status | answered | token exact | token position | sentence sequence | time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| joint | ok | 42,849/43,209 (99.2%) | 39,203/43,209 (90.7%) | 39,805/42,849 (92.9%) | 1,196/2,969 (0.403; 40.3%) | 36.1s |
| dict-default | skipped: RuntimeError: --skip-dict | 0/43,209 (0.0%) | n/a | n/a | n/a | 0.0s |
| liepa | skipped: RuntimeError: --skip-liepa | 0/43,209 (0.0%) | n/a | n/a | n/a | 0.0s |

Thesis context: the 2026 VU thesis reports sentence-level sequence accuracy 0.711 for its transformer and 0.702 for VDU Kirciuoklis on 2,303 Chrestomatija samples. Tokenization and normalization protocols may differ from this reimplementation, so the cross-paper comparison is indicative.

## Alignment Diagnostics

| system | skipped gold tokens | skipped model tokens |
| --- | ---: | ---: |
| joint | 360 | 0 |
| dict-default | 0 | 0 |
| liepa | 0 | 0 |

## Sample Disagreements

| system | page | word | gold | predicted | sentence excerpt |
| --- | ---: | --- | --- | --- | --- |
| joint | 11 | nūn | nū̃n | nūñ | Jaũ nū̃n, kõ tėvaĩ niekadà neregė́jo, Nū̃n šìtai vìs jū́sump atė̃jo. |
| joint | 11 | jūsump | jū́sump | jūsum̃p | Jaũ nū̃n, kõ tėvaĩ niekadà neregė́jo, Nū̃n šìtai vìs jū́sump atė̃jo. |
| joint | 11 | jūsump | jū́sump | jū̃sump | Veizdė́kit ir̃ dabókitėsi, žmónes vìsos, Šìtai eĩt jū́sump žõdis dangaũs karalỹstos. |
| joint | 11 | maloniai | maloniaĩ | malõniai | Maloniaĩ ir̃ sù džiaugsmù tą̃ žõdį priim̃kit, Õ jū́sų ū́kiuose šeimýną mókykit. |
| joint | 11 | džiaugsmu | džiaugsmù | džiaũgsmu | Maloniaĩ ir̃ sù džiaugsmù tą̃ žõdį priim̃kit, Õ jū́sų ū́kiuose šeimýną mókykit. |
| joint | 11 | tur | tur̃ | tùr | Sū́nūs, dùkterys jū́sų tur̃ tataĩ mokė́ti, Visà šìrdžia tur̃ tą̃ Diẽvo žõdį mylė́ti. |
| joint | 11 | širdžia | šìrdžia | širdžià | Sū́nūs, dùkterys jū́sų tur̃ tataĩ mokė́ti, Visà šìrdžia tur̃ tą̃ Diẽvo žõdį mylė́ti. |
| joint | 11 | palaimą | paláimą | pal̃aimą | Jéi, bróliai sẽserys, tuõs žodžiùs nepapeĩksit, Diẽvą Tė́vą ir̃ Sū́nų sáu míelu padarýsit Ir̃ pašlóvinti põ akimìs Diẽv… |
| joint | 11 | šituo | šìtuo | šituõ | Šìtuo mókslu Diẽvą tikraĩ pažìnsit Ir̃ Dangaũs karalỹstosp prisiar̃tinsit. |
| joint | 11 | prisiartinsit | prisiar̃tinsit | prisiártinsit | Šìtuo mókslu Diẽvą tikraĩ pažìnsit Ir̃ Dangaũs karalỹstosp prisiar̃tinsit. |
| joint | 12 | laukas | laũkas | láukas | LAŨKAS, kẽlias, píeva, krỹžius, Šìlo júosta mėlynà, Debesė̃lių tánkus ìžas Ir̃ graudì graudì dainà. |
| joint | 12 | mėlyna | mėlynà | mė́lyna | LAŨKAS, kẽlias, píeva, krỹžius, Šìlo júosta mėlynà, Debesė̃lių tánkus ìžas Ir̃ graudì graudì dainà. |
| joint | 12 | debesėlių | debesė̃lių | debesė́lių | LAŨKAS, kẽlias, píeva, krỹžius, Šìlo júosta mėlynà, Debesė̃lių tánkus ìžas Ir̃ graudì graudì dainà. |
| joint | 12 | ižas | ìžas | ižãs | LAŨKAS, kẽlias, píeva, krỹžius, Šìlo júosta mėlynà, Debesė̃lių tánkus ìžas Ir̃ graudì graudì dainà. |
| joint | 12 | prietemy | prietemỹ | príetemy | Bė́ga kẽlias, ir̃ beržẽliai Liñksta vė́jo pučiamì; Samanótas stógas žãlias Ir̃ šuñs bal̃sas prietemỹ. Õ toliaũ – paskeñ… |

Report path: `local/accentuator/reports/chrestomatija-eval-v3.md`
