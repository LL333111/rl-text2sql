| Row | Val acc | Spider dev acc | Spider dev executable | BIRD dev acc | BIRD dev executable | Mean output tokens | Training GPU-hours |
|---|---|---|---|---|---|---|---|
| 1. Qwen2.5-Coder-1.5B zero-shot | 66.0 | 62.2 | 82.5 | 28.7 | 55.3 | 38.6 | 0.00 |
| 2. + LoRA SFT (best on val) | 78.7 | 69.5 | 90.0 | 29.6 | 64.3 | 39.0 | 2.10 (of 3.00) |
| 2. + LoRA SFT (end of budget) | 75.7 | 70.8 | 91.3 | 30.3 | 64.5 | 38.6 | 3.00 (of 3.00) |
| 3. + GRPO binary reward (best on val) | 78.1 | 70.2 | 92.8 | 35.3 | 71.9 | 29.2 | 2.10 (of 3.00) |
| 3. + GRPO binary reward (end of budget) | 77.1 | 71.9 | 93.0 | 32.4 | 63.2 | 26.8 | 3.00 (of 3.00) |
| Abl. GRPO partial-credit reward (best on val) | 77.3 | 72.8 | 93.0 | 34.2 | 69.3 | 30.9 | 1.51 (of 3.00) |
| Abl. GRPO partial-credit reward (end of budget) | 76.7 | 72.2 | 94.8 | 32.3 | 68.4 | 28.7 | 3.00 (of 3.00) |
| Ref. Qwen2.5-Coder-7B zero-shot | 85.8 | 76.8 | 97.2 | 53.8 | 90.9 | 39.0 | 0.00 |
