# WordleRL: Wordle GRPO (Qwen3 + TRL)

Trains a Qwen3 model to play Wordle with TRL's `GRPOTrainer(environment_factory=...)`.
Game logic is TextArena's `Wordle-v0`, with the standard Wordle word lists and our own scoring.

## Layout

| Path | What it is |
|---|---|
| `wordle_rl/env.py` | `WordleToolEnv`: the TRL environment (one `guess(word)` tool) around TextArena's Wordle |
| `wordle_rl/scoring.py` | Reward components and default weights |
| `wordle_rl/prompt.py` | Rules shown to the model |
| `wordle_rl/words.py` | Loads the word lists |
| `wordle_rl/progress.py` | Progress lines (step/total, elapsed, ETA, IST) for log files |
| `train_grpo.py` | Training script (`--dry-run` checks config without a GPU; `--eval-only` = baseline) |
| `scripts/colab/` | Colab launcher (laptop) and remote job (VM) |
| `plans/` | Colab run plans |
| `requirements.lock` | Exact package versions of the laptop venv, used to build the VM's venv |
| `scripts/check_setup.py` | Prints the tool schema and the exact Qwen3 prompt, thinking on and off |
| `data/raw_*.txt` | Original downloaded lists (2,315 answers; 10,657 extra accepted guesses) |
| `data/answers_train.txt`, `answers_eval.txt` | 2,000 training secrets / 315 held-out secrets (seed 0, `scripts/make_splits.py`). Training uses only the first 100 training words after a seeded shuffle (`--train-size`, decided 2026-10-29). The baseline and every evaluation use the same fixed random 20 of the 315 held-out words (`--eval-size 20`, seed 1234, decided 2026-10-29) |
| `data/allowed_guesses.txt` | All 12,972 accepted guesses (answers + extras) |

## Environment

Virtual env in WSL at `~/.venvs/wordle` (not in the project folder; faster than `/mnt/c`).
Created by `scripts/setup_env.sh`: trl 1.16.0, vllm 0.30.0, transformers 5.20.2, torch 2.13 (CUDA 13), peft 0.21.2, textarena 0.7.4.

Run anything inside it from PowerShell:

```
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/praty/Projects/Code/Wordle/scripts/wsl_run.sh python -m pytest -q tests
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/praty/Projects/Code/Wordle/scripts/wsl_run.sh python train_grpo.py --dry-run
```

## Running on Colab

`scripts/colab/launch.sh` (adapted from LearningExp; rules in the `colab-gpu-runs` skill) ships the committed code
(`git archive HEAD`) to a fresh Colab VM and runs a plan file (`plans/*.json`: a list of named `train_grpo.py` runs).

```
bash scripts/colab/launch.sh --plan plans/pilot_4b.json --gpu L4 --max-hours 1.5 --preflight-only
```

- **Pre-flight** (nothing billed): clean git tree, plan committed, bf16 GPU, Colab login, unique session name. It also checks the bundle the VM will receive: no CRLF, `bash -n`, Python compiles.
- **On the VM** (`scripts/colab/remote_job.py`):
  - builds `/content/venv` from `requirements.lock` (the laptop venv's exact versions) and checks bf16;
  - runs each item, skipping items whose `results/<name>/DONE` is committed, and keeps going past failures;
  - stops at the `--max-hours` deadline;
  - packs results every 5 min, and releases itself at the end.
- **The laptop** downloads status and results every 2 min into `results/<name>/`. It caps the attached client at ~20% of a core and always runs `colab stop` at the end. Launch logs go to `results/colab_launch_*/` (git-ignored).

## Scoring

Each component is a separate TRL reward function (logged as `rewards/<name>/mean`); the total is the weighted sum.
Change weights with `--reward-weights "solved=1,speed=0,letters=0.2"`; weight 0 drops a component.

| Component | Default weight | Value |
|---|---|---|
| solved | 1.0 | 1 if the secret was found, else 0 |
| speed | 0.25 | Only when solved: (6 - guesses used) / 5, so 1.0 on guess 1 and 0 on guess 6 |
| consistency | 0.25 | Share of guesses 2..n that fit all feedback so far (hard-mode rule) |
| invalid | 0.25 | -(invalid attempts) / 3, floored at -1 (not a word, wrong length, repeat, unparseable) |
| unfinished | 0.5 | -1 if the model stopped before the game ended |
| letters | 0 | Best guess's (greens + 0.5 x yellows) / 5 (TextArena's partial credit) |

Invalid guesses do not use up one of the 6 turns; 4 invalid guesses in a row end the game as a loss.

## LoRA

Training is always low-rank LoRA (no full fine-tune option), following TRL's "LoRA Without Regret" guide for GRPO:
rank 1 (`--lora-r`), alpha 32 (`--lora-alpha`), dropout 0, on all linear layers, learning rate 1e-5
(about 10x a full fine-tune's), and 16 games per optimizer step (LoRA handles batches above ~32 worse).
TRL's guide gives rank 1-32 as the range for RL.

## Thinking mode

Qwen3-4B (the original hybrid release) switches with the chat template flag `enable_thinking`,
set here through `GRPOConfig(chat_template_kwargs={"enable_thinking": ...})` and the `--thinking` flag.
Off: the template inserts an empty `<think></think>` so the model answers straight away.
On: the model writes its reasoning in `<think>...</think>` before each tool call, so give it a much larger
`--max-completion-length` (default 8192 with `--thinking`, 1536 without), which covers the whole game.
`Qwen3-4B-Instruct-2507` cannot think and `Qwen3-4B-Thinking-2507` always thinks; the flag has no effect on them.
