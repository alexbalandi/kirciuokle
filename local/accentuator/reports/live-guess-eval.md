# Live Guess Evaluation

## Corpus
- silver tokens: 5,020
- silver word types: 2,327
- ambiguous silver tokens: 767
- dictionary OOV tokens: 1,196 (23.8%)
- dictionary OOV types: 791 (34.0%)
- live backend cascade: `nn&liepa+liepa`
- generated DB: `C:\accentuation_lt\local\accentuator\data\generated.sqlite`
- guesses DB: `C:\accentuation_lt\local\accentuator\data\guesses.sqlite`

## Tiers
| tier | tokens | types | token exact | token position | type exact | type position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dict | 3,824 (76.2%) | 1,536 (66.0%) | 3,461/3,824 (90.5%) | 3,475/3,824 (90.9%) | 1,390/1,536 (90.5%) | 1,396/1,536 (90.9%) |
| precomputed-guess | 545 (10.9%) | 346 (14.9%) | 476/545 (87.3%) | 481/545 (88.3%) | 311/346 (89.9%) | 312/346 (90.2%) |
| live-guess | 565 (11.3%) | 427 (18.3%) | 439/565 (77.7%) | 416/565 (73.6%) | 348/427 (81.5%) | 331/427 (77.5%) |
| unanswered | 86 (1.7%) | 18 (0.8%) | n/a | n/a | n/a | n/a |

## Live-Guess Disagreements
- štombergas: live=štómbergas silver=štòmbergas
- kuzminskas: live=kuzmínskas silver=kuzmìnskas
- rietyje: live=rietyjè silver=rietyje
- žaista: live=žaĩstà silver=žaĩsta
- vienerias: live=vienerias silver=víenerias
- lavrinovičius: live=lavrinovičius silver=lavrinòvičius
- jasikevičius: live=jasikevičius silver=jasikẽvičius
- šiškauskas: live=šiškauskas silver=šiškáuskas
- eurelijus: live=eurelijus silver=eurèlijus
- karnišovas: live=karnišóvas silver=karnišòvas
- skambesnės: live=skambèsnė̃s silver=skambèsnės
- aštuntfinalis: live=aštuntfinalis silver=aštuñtfinalis
- liubliana: live=liubliana silver=liublianà
- latauskienė: live=latauskienė silver=latáuskienė
- latauskienės: live=latauskienės silver=latáuskienės
- neretai: live=nerẽtaĩ silver=neretaĩ
- prisikursime: live=prisikúrsime silver=prisikùrsime
- vadybinės: live=vadybinės silver=vadýbinės
- paskirstant: live=paskírstant silver=paskìrstant
- apsibrėžti: live=apsibrė́žti silver=apsibrėžtì
