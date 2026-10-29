"""Training trend by phase, using measures that depend less on which words a step drew.

  python scripts/trend.py <games.jsonl> [<games.jsonl> ...]
"""

import json
import sys


def main() -> None:
    phases = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    print("| Steps | Run | Games | Solved | Guesses that fit all feedback so far | Invalid attempts per game | Tokens per guess attempt | Mean tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for path in sys.argv[1:]:
      run = path.split("/")[-2]
      games = [g for g in map(json.loads, open(path)) if g["mode"] == "train"]
      for lo, hi in phases:
        gs = [g for g in games if lo <= g["step"] + 1 <= hi]
        if not gs:
            continue
        budget = run
        n = len(gs)
        cons = [g["score_consistency"] for g in gs if len(g["guesses"]) >= 2]
        tpg = sum(g["completion_tokens"] / max(len(g["guesses"]) + g["invalid"], 1) for g in gs) / n
        print(f"| {lo}-{min(hi, max(g['step'] + 1 for g in gs))} | {budget} | {n} | {sum(g['solved'] for g in gs) / n:.0%} | "
              f"{sum(cons) / len(cons):.0%} | {sum(g['invalid'] for g in gs) / n:.2f} | {tpg:.0f} | "
              f"{sum(g['completion_tokens'] for g in gs) / n:.0f} |")


if __name__ == "__main__":
    main()
