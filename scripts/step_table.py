"""Per-step, per-word summary of training games from a run's games.jsonl.

  python scripts/step_table.py results/grpo-qwen3-4b-think/games.jsonl [--budget 8192]

Training step = recorded step + 1: games are recorded when they are scored, before that step's update.
"""

import argparse
import json
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("games")
    p.add_argument("--budget", type=int, default=8192)
    a = p.parse_args()

    groups = defaultdict(list)
    for line in open(a.games):
        g = json.loads(line)
        if g["mode"] == "train":
            groups[(g["step"] + 1, g["secret"])].append(g)

    print("| Step | Word | Games | Solved | Guesses when solved | Games with an invalid guess | Hit token limit | Stopped early | Mean tokens |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (step, word), gs in sorted(groups.items()):
        won = [g for g in gs if g["solved"]]
        gw = f"{sum(len(g['guesses']) for g in won) / len(won):.1f}" if won else "–"
        print(f"| {step} | {word} | {len(gs)} | {len(won)} | {gw} | {sum(g['invalid'] > 0 for g in gs)} | "
              f"{sum((g['completion_tokens'] or 0) >= 0.9 * a.budget for g in gs)} | "
              f"{sum(not g['done'] for g in gs)} | {sum(g['completion_tokens'] or 0 for g in gs) / len(gs):.0f} |")


if __name__ == "__main__":
    main()
