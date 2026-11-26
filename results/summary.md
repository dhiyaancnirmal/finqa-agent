| Model | Agent | n | pass@1 | 95% CI | Env steps / task | Tokens / task | Median s / task | Hit 50-step cap | Cost / task |
|---|---|---|---|---|---|---|---|---|---|
| `nemotron-3.5-lightning-30b-a3b` | baseline | 6 | **0.0%** | 0.0-39.0% | 29.3 | 251,656 | 580 | 3 | $0 (free tier) |
| `nemotron-3.5-lightning-30b-a3b` | harness (all) | 17 | **64.7%** | 41.3-82.7% | 11.6 | 55,509 | 128 | 0 | $0 (free tier) |
| `nemotron-3.5-super-vl-120b-a12b` | baseline | 79 | **68.4%** | 57.5-77.6% | 7.5 | 37,443 | 67 | 3 | $0 (free tier) |
| `nemotron-3.5-super-vl-120b-a12b` | harness (all) | 98 | **73.5%** | 64.0-81.2% | 5.9 | 21,897 | 71 | 0 | $0 (free tier) |
| `nemotron-3-ultra-550b-a55b` | baseline | 20 | **50.0%** | 29.9-70.1% | 9.2 | 28,639 | 71 | 0 | $0 (free tier) |
| `nemotron-3-ultra-550b-a55b` | harness (all) | 8 | **100.0%** | 67.6-100.0% | 4.1 | 15,701 | 152 | 0 | $0 (free tier) |

Paired: the same questions, answered by both agents.

| Model | Questions | Baseline | Harness | Gain | Only harness right | Only baseline right | Tokens / task, baseline -> harness |
|---|---|---|---|---|---|---|---|
| `nemotron-3.5-lightning-30b-a3b` | 6 | 0/6 (0.0%) | 5/6 (83.3%) | +83.3 pts | 5 | 0 | 251,656 -> 51,195 |
| `nemotron-3.5-super-vl-120b-a12b` | 79 | 54/79 (68.4%) | 60/79 (75.9%) | +7.6 pts | 9 | 3 | 37,443 -> 22,651 |
| `nemotron-3-ultra-550b-a55b` | 8 | 5/8 (62.5%) | 8/8 (100.0%) | +37.5 pts | 3 | 0 | 41,857 -> 15,701 |
