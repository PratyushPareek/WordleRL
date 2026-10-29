# 8k-token run (2026-10-29, A100-SXM4-40GB, commit 89c01f8): stopped after step 31 of 50

Qwen3-4B, thinking on, 8,192 tokens per game, plain sampling, per-token vLLM correction.
- Baseline (20 held-out words): 2/20 solved, 16/20 ran out of budget.
- Evaluation at step 25 (same 20 words): 3/20 solved, 17/20 hit the budget. No real improvement.
- Games that ran out made ~3 guesses at ~2,200-2,600 tokens of thinking per guess; solved games used ~1,000-1,400
  per guess. See budget_analysis.md and steps_by_word.md.
Stopped at step 31 to rerun with a 16,384-token budget. Session cost ~17 compute units.
Checkpoints 15 and 30 are kept locally (not in git).
