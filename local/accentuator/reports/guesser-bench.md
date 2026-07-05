# Out-of-dictionary stress guesser benchmark

- training types: 523,828 (generated.sqlite, default forms)
- `held`: 10,690 in-domain held-out types (seed 20260705)
- `gap`: 2,751 VDU-cache words the dictionary does not cover
- metrics of answered; `exact-over-all` = answered x exact

| candidate | slice | answered | exact | position | exact-over-all |
|---|---|---:|---:|---:|---:|
| trie | held | 100.0% | 88.1% | 91.4% | 88.1% |
| trie | gap | 100.0% | 50.5% | 61.7% | 50.5% |
| anbinderis | held | 97.6% | 97.2% | 97.5% | 94.8% |
| anbinderis | gap | 63.0% | 67.0% | 71.9% | 42.2% |
| liepa | held | 100.0% | 78.8% | 84.7% | 78.8% |
| liepa | gap | 100.0% | 88.1% | 95.3% | 88.0% |
| agree(nn,liepa) | held | 77.6% | 99.7% | 99.8% | 77.4% |
| agree(nn,liepa) | gap | 50.5% | 99.5% | 99.6% | 50.3% |
| agree->liepa | held | 100.0% | 78.8% | 84.7% | 78.8% |
| agree->liepa | gap | 100.0% | 88.1% | 95.3% | 88.0% |
| nn@0 | held | 100.0% | 97.9% | 98.1% | 97.9% |
| nn@0 | gap | 100.0% | 59.5% | 66.9% | 59.5% |
