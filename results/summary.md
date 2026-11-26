| Model | Agent | n | pass@1 | 95% CI | Env steps / task | Tokens / task | Median s / task | Hit 50-step cap | Cost / task |
|---|---|---|---|---|---|---|---|---|---|
| `nemotron-3.5-super-vl-120b-a12b` | baseline | 63 | **74.6%** | 62.7-83.7% | 5.6 | 15,174 | 57 | 0 | $0 (free tier) |
| `nemotron-3.5-super-vl-120b-a12b` | harness (all) | 62 | **79.0%** | 67.4-87.3% | 5.4 | 18,996 | 54 | 0 | $0 (free tier) |
| `nemotron-3-ultra-550b-a55b` | baseline | 16 | **56.2%** | 33.2-76.9% | 7.6 | 21,776 | 63 | 0 | $0 (free tier) |
| `nemotron-3-ultra-550b-a55b` | harness (all) | 2 | **100.0%** | 34.2-100.0% | 4.0 | 14,932 | 150 | 0 | $0 (free tier) |

Paired: the same questions, answered by both agents.

| Model | Questions | Baseline | Harness | Gain | Only harness right | Only baseline right | Tokens / task, baseline -> harness |
|---|---|---|---|---|---|---|---|
| `nemotron-3.5-super-vl-120b-a12b` | 60 | 46/60 (76.7%) | 48/60 (80.0%) | +3.3 pts | 5 | 3 | 14,976 -> 18,446 |
| `nemotron-3-ultra-550b-a55b` | 2 | 1/2 (50.0%) | 2/2 (100.0%) | +50.0 pts | 1 | 0 | 51,316 -> 14,932 |
