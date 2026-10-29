"""Reward components for GRPO.

Each component is a separate TRL reward function, so TRL logs it on its own
(`rewards/<name>/mean`) and its weight is set with `GRPOConfig.reward_weights`.
Every function reads the finished episode stored on the environment instance.

Components (all in [-1, 1] before weighting):
  solved        1 if the secret was found, else 0.
  speed         Only when solved: guesses saved out of the 5 that could be saved
                (1.0 = solved on guess 1, 0.0 = solved on guess 6). 0 when not solved.
  consistency   Share of guesses 2..n that fit all feedback seen so far (Wordle
                "hard mode" rule: greens kept, yellows reused elsewhere, greys avoided).
                Rewards using the feedback rather than guessing blindly. 0 if fewer than 2 guesses.
  invalid       -(invalid attempts) / max_invalid, floored at -1. Invalid = not a word,
                wrong length, repeated word, or unparseable.
  unfinished    -1 if the model stopped before the game ended (no win, fewer than 6
                valid guesses), e.g. it stopped calling the tool or ran out of tokens.
  letters       Best single guess's (greens + 0.5 * yellows) / 5, TextArena's partial credit.
                Off by default; TRL reports plain win/lose rewards trained more cleanly.
"""

from collections import Counter

NUM_GUESSES = 6

# Default weights; override with --reward-weights on the training script.
DEFAULT_WEIGHTS = {
    "solved": 1.0,
    "speed": 0.25,
    "consistency": 0.25,
    "invalid": 0.25,
    "unfinished": 0.5,
    "letters": 0.0,
}


def feedback_for(guess: str, secret: str) -> list[str]:
    """Wordle colouring with duplicate letters handled the standard way (greens first)."""
    result = ["X"] * len(guess)
    remaining = Counter()
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            result[i] = "G"
        else:
            remaining[s] += 1
    for i, g in enumerate(guess):
        if result[i] != "G" and remaining[g] > 0:
            result[i] = "Y"
            remaining[g] -= 1
    return result


def is_consistent(word: str, history: list[tuple[str, list[str]]]) -> bool:
    """True if `word` could still be the secret given every (guess, feedback) seen so far."""
    return all(feedback_for(guess, word) == fb for guess, fb in history)


def score_components(episode: dict, max_invalid: int = 3) -> dict[str, float]:
    """episode: {"history": [(guess, feedback), ...], "solved": bool, "invalid": int, "done": bool}"""
    history = episode["history"]
    solved = episode["solved"]
    n = len(history)

    later = [is_consistent(history[i][0], history[:i]) for i in range(1, n)]
    best_letters = max(
        ((fb.count("G") + 0.5 * fb.count("Y")) / len(fb) for _, fb in history),
        default=0.0,
    )
    return {
        "solved": 1.0 if solved else 0.0,
        "speed": (NUM_GUESSES - n) / (NUM_GUESSES - 1) if solved else 0.0,
        "consistency": sum(later) / len(later) if later else 0.0,
        "invalid": -min(episode["invalid"], max_invalid) / max_invalid,
        "unfinished": 0.0 if episode["done"] else -1.0,
        "letters": best_letters,
    }


def make_reward_funcs(names: list[str]):
    """One TRL reward function per component, named after it (TRL logs by __name__)."""

    def make(name):
        def reward(environments, **kwargs) -> list[float]:
            # `_episode` is private so TRL does not expose it to the model as a tool.
            return [score_components(env._episode())[name] for env in environments]

        reward.__name__ = name
        return reward

    return [make(name) for name in names]


def make_game_recorder(path: str):
    """A zero-weight TRL reward function that appends every finished game to a JSONL file.
    Set `record_games.trainer` after building the trainer to tag rows with train/eval and the step."""
    import json

    def record_games(environments, completion_ids=None, **kwargs) -> list[float]:
        trainer = getattr(record_games, "trainer", None)
        mode = ("train" if trainer.model.training else "eval") if trainer is not None else None
        step = trainer.state.global_step if trainer is not None else None
        with open(path, "a") as f:
            for i, env in enumerate(environments):
                ep = env._episode()
                row = {
                    "mode": mode,
                    "step": step,
                    "secret": ep["secret"],
                    "guesses": [w for w, _ in ep["history"]],
                    "feedback": ["".join(fb) for _, fb in ep["history"]],
                    "solved": ep["solved"],
                    "invalid": ep["invalid"],
                    "done": ep["done"],
                    "completion_tokens": len(completion_ids[i]) if completion_ids is not None else None,
                    **{f"score_{k}": v for k, v in score_components(ep).items()},
                }
                f.write(json.dumps(row) + "\n")
        return [0.0] * len(environments)

    return record_games
