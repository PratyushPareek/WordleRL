"""GRPO training of a Qwen3 model on Wordle (TextArena game, standard word lists, custom scoring).

Examples (run inside WSL via scripts/wsl_run.sh):
  # Baseline: untrained model on the 20 held-out evaluation words
  python train_grpo.py --model Qwen/Qwen3-4B --eval-only --eval-batch-size 20

  # Training, thinking on
  python train_grpo.py --model Qwen/Qwen3-4B --thinking

On Colab, runs go through scripts/colab/launch.sh with a plan file (plans/*.json).
Every game is written to <output-dir>/games.jsonl; timings and summaries to <output-dir>/run.json.
"""

import argparse
import json
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from wordle_rl.env import WordleToolEnv
from wordle_rl.progress import ProgressCallback
from wordle_rl.prompt import build_prompt
from wordle_rl.scoring import DEFAULT_WEIGHTS, make_game_recorder, make_reward_funcs
from wordle_rl.words import answers


def parse_weights(text: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    for part in (text or "").split(","):
        if part.strip():
            name, value = part.split("=")
            if name.strip() not in weights:
                raise ValueError(f"unknown reward component {name!r}; known: {list(weights)}")
            weights[name.strip()] = float(value)
    return weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (enable_thinking=True)")
    p.add_argument(
        "--reward-weights",
        default=None,
        help="Override component weights, e.g. 'solved=1,speed=0,letters=0.2'. "
        f"Defaults: {DEFAULT_WEIGHTS}. Weight 0 drops the component.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--train-size", type=int, default=100,
                   help="Unique secret words to train on: the first N of the training list after a seeded shuffle "
                   "(Pratyush, 2026-10-29: 100). One pass = N / (words per step) steps")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--num-generations", type=int, default=8, help="Games per secret word (GRPO group size)")
    # 4 x 4 = 16 games per optimizer step: LoRA tolerates large batches worse than full fine-tuning (keep < 32).
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=None,
                   help="Token budget for the whole game (all turns + feedback). Default 1536, or 8192 with --thinking")
    # Temperature 1.0 for varied games within a group; top-p/top-k at Qwen's recommended values for thinking
    # (Pratyush, 2026-10-29), which trims the tail of unlikely tokens that makes thinking ramble.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=3)
    # Low-rank LoRA on all linear layers (TRL "LoRA Without Regret": RL needs rank 1-32, rank 1 matches full FT).
    p.add_argument("--lora-r", type=int, default=1)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    p.add_argument("--vllm-server-url", default="http://localhost:8000")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--vllm-max-model-length", type=int, default=None,
                   help="vLLM context window. Default max_completion_length + 1024 (prompt is ~400 tokens); "
                   "the model's own 40k default needs too much KV-cache memory")
    p.add_argument("--eval-size", type=int, default=20,
                   help="Held-out secret words per evaluation, baseline included (Pratyush, 2026-10-29: 20; 0 = no eval)")
    p.add_argument("--eval-batch-size", type=int, default=None,
                   help="Games played together per eval batch (default = --per-device-train-batch-size). "
                   "Larger is faster with vLLM, if the KV cache fits")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=15, help="Save a checkpoint (adapter + optimizer) every N steps")
    p.add_argument("--report-to", default="none", help="none, trackio, wandb, ...")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-vllm", action="store_true",
                   help="Generate with transformers instead of vLLM (local CPU tests of the real code paths only)")
    p.add_argument("--dry-run", action="store_true", help="Build config and datasets, then exit (no model, no GPU)")
    p.add_argument("--eval-only", action="store_true",
                   help="Baseline: play the held-out words with the untrained model (no LoRA, no training) "
                   "and write every game to <output-dir>/games.jsonl")
    return p.parse_args()


EVAL_SEED = 1234


class WordleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose evaluation only plays and scores games.

    TRL's evaluation also re-scores every finished game with the training model (importance-sampling
    log-probs, then an eval loss) over the whole eval batch at once: 20 games x ~8k tokens ran out of memory
    on a 40 GB A100 (2026-10-29). Evaluation needs only the games and their rewards, which
    _generate_and_score_completions computes and logs; with importance sampling off and no reference model
    (beta = 0) it runs no forward pass of the training model. Training is unchanged.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        saved = self.vllm_importance_sampling_correction
        self.vllm_importance_sampling_correction = False
        try:
            self._prepare_inputs(inputs)
        finally:
            self.vllm_importance_sampling_correction = saved
        return torch.zeros((), device=self.accelerator.device), None, None


def make_dataset(words) -> Dataset:
    prompt = build_prompt()
    return Dataset.from_dict({"prompt": [prompt] * len(words), "secret": list(words)})


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.reward_weights)
    names = [n for n, w in weights.items() if w != 0]
    max_completion_length = args.max_completion_length or (8192 if args.thinking else 1536)
    kind = "baseline" if args.eval_only else "grpo"
    output_dir = args.output_dir or f"outputs/{args.model.split('/')[-1]}-wordle-{kind}{'-think' if args.thinking else ''}"

    train_words = list(answers("train"))
    random.Random(args.seed).shuffle(train_words)
    train_words = train_words[: args.train_size]
    # answers_eval.txt is alphabetical, so take a fixed random sample (own seed: the same words for the
    # baseline and every evaluation during training, whatever --seed is).
    eval_words = list(answers("eval"))
    random.Random(EVAL_SEED).shuffle(eval_words)
    eval_words = eval_words[: args.eval_size]
    if args.eval_only and not eval_words:
        raise SystemExit("--eval-only needs --eval-size > 0")

    print(f"model {args.model} thinking={args.thinking} max_completion_length={max_completion_length}")
    print(f"reward weights: { {n: weights[n] for n in names} }")
    print(f"train secrets {len(train_words)}, eval secrets {len(eval_words)}")

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        chat_template_kwargs={"enable_thinking": args.thinking},
        # 6 valid guesses + a few invalid attempts; stops runaway tool loops.
        max_tool_calling_iterations=10,
        reward_weights=[weights[n] for n in names],
        use_vllm=not args.no_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length or max_completion_length + 1024,
        # TRL loads the model in float32 unless told otherwise (15 GB for 4B instead of 8 GB): load it in the
        # training precision. This also left too little GPU memory for vLLM in the first A100 run (2026-10-29).
        model_init_kwargs={"dtype": ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
                           if torch.cuda.is_available() else "float32"},
        # T4 (Colab free tier) has no bf16; fall back to fp16 there.
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=1,
        # TRL's rich completions table is unreadable in log files; every game goes to games.jsonl instead.
        log_completions=False,
        eval_strategy="steps" if eval_words and not args.eval_only else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.eval_batch_size or args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=args.report_to,
        seed=args.seed,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} on all linear layers, learning rate {args.learning_rate}")
    if args.dry_run:
        words_per_step = config.generation_batch_size // config.num_generations
        print(f"dry run OK: generation batch {config.generation_batch_size} games/step, "
              f"{words_per_step} secret words/step, one pass over {len(train_words)} training words = "
              f"{len(train_words) // words_per_step} steps; {args.max_steps} steps = "
              f"{args.max_steps * words_per_step / len(train_words):.1f} passes")
        return

    # Every game (training and evaluation) is appended to games.jsonl by a zero-weight reward function.
    os.makedirs(output_dir, exist_ok=True)
    games_path = os.path.join(output_dir, "games.jsonl")
    if os.path.exists(games_path):
        os.remove(games_path)
    recorder = make_game_recorder(games_path)
    reward_funcs = make_reward_funcs(names) + [recorder]
    config.reward_weights = config.reward_weights + [0.0]

    start = time.time()
    trainer = WordleGRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        train_dataset=make_dataset(train_words),
        eval_dataset=make_dataset(eval_words) if eval_words else None,
        args=config,
        environment_factory=WordleToolEnv,
        peft_config=None if args.eval_only else peft_config,
        callbacks=[ProgressCallback()],
    )
    recorder.trainer = trainer
    setup_sec = time.time() - start
    run = {"args": vars(args), "max_completion_length": max_completion_length, "setup_sec": round(setup_sec)}
    if args.eval_only:
        t = time.time()
        metrics = trainer.evaluate()
        run["eval_sec"] = round(time.time() - t)
        run["eval_metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print("baseline metrics:", {k: round(v, 4) for k, v in run["eval_metrics"].items()})
    else:
        t = time.time()
        trainer.train()
        run["train_sec"] = round(time.time() - t)
        run["log_history"] = trainer.state.log_history
        trainer.save_model(output_dir)
    run["games"] = summarize_games(games_path, max_completion_length)
    with open(os.path.join(output_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(f"setup {setup_sec:.0f}s, total {time.time() - start:.0f}s; wrote {output_dir}/run.json")


def summarize_games(path: str, max_completion_length: int) -> dict:
    """Per mode (train / eval): win rate, guesses, invalid guesses, early stops, length-limit hits."""
    games = [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    out = {}
    for mode in sorted({g["mode"] for g in games}, key=str):
        gs = [g for g in games if g["mode"] == mode]
        n = len(gs)
        won = [g for g in gs if g["solved"]]
        tokens = [g["completion_tokens"] or 0 for g in gs]
        out[str(mode)] = s = {
            "games": n,
            "solved_rate": len(won) / n,
            "mean_guesses_when_solved": sum(len(g["guesses"]) for g in won) / len(won) if won else None,
            "games_with_invalid_rate": sum(g["invalid"] > 0 for g in gs) / n,
            "stopped_early_rate": sum(not g["done"] for g in gs) / n,
            # A game out of budget stops at the last turn that still fits, so count >= 90% of the budget.
            "hit_length_limit_rate": sum(t >= 0.9 * max_completion_length for t in tokens) / n,
            "mean_completion_tokens": sum(tokens) / n,
        }
        print(f"[{mode}] games {n}: solved {s['solved_rate']:.1%}, guesses when solved "
              f"{s['mean_guesses_when_solved'] or float('nan'):.2f}, with an invalid guess "
              f"{s['games_with_invalid_rate']:.1%}, stopped early {s['stopped_early_rate']:.1%}, "
              f"hit the length limit {s['hit_length_limit_rate']:.1%}, mean tokens {s['mean_completion_tokens']:.0f}")
    return out


if __name__ == "__main__":
    main()
