"""Word-by-word comparison of evaluations on the same held-out words.

  python scripts/compare_evals.py --budget 12288 "Baseline=results/baseline-x/games.jsonl:0" "Step 50=results/grpo-x/games.jsonl:50"
Each argument is LABEL=PATH:STEP (eval games recorded at that trainer step).
"""

import argparse
import json


def outcome(g, budget):
    if g["solved"]:
        return f"won in {len(g['guesses'])}"
    if (g["completion_tokens"] or 0) >= 0.9 * budget:
        return "out of budget"
    return "lost (6 guesses)" if g["done"] else "stopped early"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    runs = []
    for spec in a.runs:
        label, rest = spec.split("=", 1)
        path, step = rest.rsplit(":", 1)
        games = {g["secret"]: g for g in map(json.loads, open(path)) if g["mode"] == "eval" and g["step"] == int(step)}
        runs.append((label, games))
    words = sorted(set().union(*(g.keys() for _, g in runs)))
    print("| Word | " + " | ".join(l for l, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for w in words:
        print(f"| {w} | " + " | ".join(outcome(g[w], a.budget) if w in g else "–" for _, g in runs) + " |")
    print("| **Solved** | " + " | ".join(f"**{sum(x['solved'] for x in g.values())} of {len(g)}**" for _, g in runs) + " |")
    print("| Out of budget | " + " | ".join(str(sum(outcome(x, a.budget) == 'out of budget' for x in g.values())) for _, g in runs) + " |")
    print("| Stopped early (below budget) | " + " | ".join(str(sum(outcome(x, a.budget) == 'stopped early' for x in g.values())) for _, g in runs) + " |")
    print("| Games with an invalid guess | " + " | ".join(str(sum(x['invalid'] > 0 for x in g.values())) for _, g in runs) + " |")
    print("| Mean tokens | " + " | ".join(f"{sum(x['completion_tokens'] for x in g.values()) / len(g):,.0f}" for _, g in runs) + " |")


if __name__ == "__main__":
    main()
