# Live Guess Evaluation

## Corpus
- silver tokens: 37,736
- silver word types: 11,290
- ambiguous silver tokens: 5,879
- dictionary OOV tokens: 8,725 (23.1%)
- dictionary OOV types: 4,694 (41.6%)
- live backend cascade: `nn&liepa+liepa`
- live tier conditioned vs unconditioned: conditioned
- sample live (word, label) pairs: (šarvinių, bdv., aukšt., mot. g., dgs., kilm.), (netaikydama, vksm., mot. g., vns., gal., neveik. r., dlv., reik.), (įsigytų, vksm., liep., 3 asm.), (safariland, dkt.), (lithuania, dkt.)
- generated DB: `C:\accentuation_lt\local\accentuator\data\generated.sqlite`
- guesses DB: `C:\accentuation_lt\local\accentuator\data\guesses.sqlite`
- audit overlay: `C:\accentuation_lt\local\accentuator\data\eval\lrt-silver-audit.json` (453 entries)
- live label skipped silver tokens: 18 (0.05%)
- live label skipped tagger tokens: 9

## Tiers (Raw Silver)
| tier | tokens | types | token exact | token position | type exact | type position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dict | 29,011 (76.9%) | 6,596 (58.4%) | 26,195/29,011 (90.3%) | 26,321/29,011 (90.7%) | 6,014/6,596 (91.2%) | 6,037/6,596 (91.5%) |
| precomputed-guess | 3,561 (9.4%) | 1,582 (14.0%) | 3,403/3,561 (95.6%) | 3,482/3,561 (97.8%) | 1,500/1,582 (94.8%) | 1,545/1,582 (97.7%) |
| live-guess | 3,281 (8.7%) | 2,437 (21.6%) | 3,069/3,281 (93.5%) | 3,187/3,281 (97.1%) | 2,270/2,437 (93.1%) | 2,368/2,437 (97.2%) |
| unanswered | 1,883 (5.0%) | 675 (6.0%) | n/a | n/a | n/a | n/a |

## Tiers (Audited Silver)
| tier | tokens | types | token exact | token position | type exact | type position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dict | 28,993 (76.8%) | 6,593 (58.4%) | 26,585/28,993 (91.7%) | 26,713/28,993 (92.1%) | 6,167/6,593 (93.5%) | 6,184/6,593 (93.8%) |
| precomputed-guess | 3,532 (9.4%) | 1,573 (13.9%) | 3,405/3,532 (96.4%) | 3,484/3,532 (98.6%) | 1,501/1,573 (95.4%) | 1,546/1,573 (98.3%) |
| live-guess | 3,269 (8.7%) | 2,429 (21.5%) | 3,076/3,269 (94.1%) | 3,193/3,269 (97.7%) | 2,277/2,429 (93.7%) | 2,374/2,429 (97.7%) |
| unanswered | 927 (2.5%) | 415 (3.7%) | n/a | n/a | n/a | n/a |

## Audit Diagnostics
- excluded tokens: 58
- foreign-unmarked tokens: 957
- foreign-unmarked desired unmarked/abstained: 922 (96.3%)

## Live-Guess Disagreements
- nutolusios: live=nutõlusios silver=nutólusios
- sudedamosios: live=sùdedamosios silver=sudedamõsios
- sodra: live=sódra silver=sodrà
- sodros: live=sódros silver=sõdros
- supažindina: live=supažíndina silver=supažìndina
- darbingais: live=darbíngais silver=darbìngais
- sodrą: live=sódrą silver=sõdrą
- skyrimo: live=skyrìmo silver=skýrimo
- štombergas: live=štómbergas silver=štòmbergas
- kuzminskas: live=kuzmínskas silver=kuzmìnskas
- rietyje: live=rietyjè silver=rietyje
- karnišovas: live=karnišóvas silver=karnišòvas
- prisikursime: live=prisikúrsime silver=prisikùrsime
- paskirstant: live=paskírstant silver=paskìrstant
- apsibrėžti: live=apsibrė́žti silver=apsibrėžtì
- delsimą: live=del̃simą silver=delsìmą
- stingdo: live=stíngdo silver=stìngdo
- nepulsi: live=nepúlsi silver=nepùlsi
- prisijaukink: live=prisijaukínk silver=prisijaukìnk
- paskirtojo: live=paskírtojo silver=paskìrtojo
