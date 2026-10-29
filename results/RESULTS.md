# Results so far (2026-10-30)

Qwen3-4B, thinking on, low-rank LoRA GRPO (rank 1, alpha 32, lr 1e-5), 16 games per step (2 words x 8),
plain sampling (temperature 1.0), per-token vLLM correction. Training words: 100; evaluation: the same 20 held-out
words, one game each.

## Overall

| Evaluation (20 held-out words) | Budget | Solved | Out of budget | Stopped early below budget | Mean tokens |
|---|---|---|---|---|---|
| Baseline | 8,192 | 2 | 16 | 2 | 7,339 |
| After 25 steps | 8,192 | 3 | 15 | 2 | 7,364 |
| Baseline | 12,288 | 3 | 9 | 6 | 9,924 |
| After 50 steps (30 at 8k, then 20 at 12k) | 12,288 | 4 | 9 | 4 | 9,121 |

"Out of budget": unsolved and used 90% or more of the budget. "Stopped early": unsolved, not out of budget, game not
over (the model ended its turn without a parseable tool call, or similar). One game is 5 points of win rate, so these
differences are within noise. Word-by-word tables: `compare_evals.md`.

Training games by phase (different words per phase): solved 37% (steps 1-10) to 20% (41-50); guesses that fit all
feedback so far 59% to 48%; invalid attempts per game 0.59 to 1.52 (the last two partly from longer games at 12k).
No sign of learning. See `trend.md`.

## Runs

| Folder | What |
|---|---|
| `attempt1-fp32-oom/`, `attempt2-eval-oom/`, `attempt3-zero-updates/` | Failed attempts and their fixes |
| `run-8k-stopped-step30/` | 8k run, stopped after step 31; checkpoints 15 and 30 kept locally |
| `baseline-qwen3-4b-think-12k/` | 12k baseline |
| `grpo-qwen3-4b-think-12k-from30/` | Resumed from step 30 at 12k, steps 31-50; final adapter (checkpoints 45/50 local only) |

Compute: about 45 units for the attempts and both runs (8k run ~17, 12k run 19.5).
