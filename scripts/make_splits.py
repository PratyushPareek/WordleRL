"""Build the word lists used for training and evaluation from the standard Wordle lists.

Inputs (downloaded from cfreshman's gists, the commonly used copy of the original NYT lists):
  data/raw_answers.txt          2,315 possible secret words
  data/raw_allowed_guesses.txt  10,657 extra words accepted as guesses (never secrets)

Outputs:
  data/answers_train.txt   secrets used during training
  data/answers_eval.txt    held-out secrets, never seen in training
  data/allowed_guesses.txt every word accepted as a guess (answers + extra guesses)
"""

import argparse
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WORD = re.compile(r"^[a-z]{5}$")


def read_words(path: Path) -> list[str]:
    words = [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]
    bad = [w for w in words if not WORD.match(w)]
    if bad:
        raise ValueError(f"{path.name}: {len(bad)} malformed entries, e.g. {bad[:5]}")
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-size", type=int, default=315)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    answers = sorted(set(read_words(DATA / "raw_answers.txt")))
    extra = set(read_words(DATA / "raw_allowed_guesses.txt"))
    allowed = sorted(set(answers) | extra)

    shuffled = answers[:]
    random.Random(args.seed).shuffle(shuffled)
    eval_words = sorted(shuffled[: args.eval_size])
    train_words = sorted(shuffled[args.eval_size :])

    (DATA / "answers_train.txt").write_text("\n".join(train_words) + "\n")
    (DATA / "answers_eval.txt").write_text("\n".join(eval_words) + "\n")
    (DATA / "allowed_guesses.txt").write_text("\n".join(allowed) + "\n")
    print(f"answers: {len(answers)} -> train {len(train_words)}, eval {len(eval_words)} (seed {args.seed})")
    print(f"allowed guesses: {len(allowed)}")


if __name__ == "__main__":
    main()
