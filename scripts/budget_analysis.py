"""How far games get before running out of tokens: guesses made, tokens per guess, by phase.

  python scripts/budget_analysis.py results/baseline-qwen3-4b-think/games.jsonl results/grpo-qwen3-4b-think/games.jsonl
"""

import json
import sys
from collections import Counter

BUDGET = 8192


def phases(paths):
    out = {}
    for path in paths:
        for g in map(json.loads, open(path)):
            if g["mode"] == "eval" and g["step"] == 0:
                key = "Baseline (held-out)"
            elif g["mode"] == "eval":
                key = f"Eval step {g['step']} (held-out)"
            else:
                s = g["step"] + 1
                key = "Training steps 1-10" if s <= 10 else "Training steps 11-20" if s <= 20 else "Training steps 21+"
            out.setdefault(key, []).append(g)
    return out


def main() -> None:
    data = phases(sys.argv[1:])
    print("| Phase | Games | Out of budget | Guesses made when out of budget: 0 / 1 / 2 / 3 / 4 / 5 | Mean guesses when out | Tokens per guess (out of budget) | Tokens per guess (solved) |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(data, key=lambda k: (not k.startswith("Base"), not k.startswith("Train"), k)):
        gs = data[key]
        out = [g for g in gs if (g["completion_tokens"] or 0) >= 0.9 * BUDGET and not g["solved"]]
        won = [g for g in gs if g["solved"]]
        dist = Counter(len(g["guesses"]) for g in out)
        tpg = lambda xs: sum(g["completion_tokens"] / max(len(g["guesses"]) + g["invalid"], 1) for g in xs) / len(xs) if xs else float("nan")
        print(f"| {key} | {len(gs)} | {len(out)} ({len(out) / len(gs):.0%}) | "
              f"{' / '.join(str(dist.get(k, 0)) for k in range(6))} | "
              f"{sum(len(g['guesses']) for g in out) / len(out):.1f} | {tpg(out):.0f} | {tpg(won):.0f} |")


if __name__ == "__main__":
    main()
