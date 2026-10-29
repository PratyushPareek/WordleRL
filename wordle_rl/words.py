"""Standard Wordle word lists (built by scripts/make_splits.py)."""

from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> list[str]:
    return [w for w in (DATA / name).read_text().split() if w]


@lru_cache
def allowed_guesses() -> frozenset[str]:
    return frozenset(_read("allowed_guesses.txt"))


@lru_cache
def answers(split: str) -> tuple[str, ...]:
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    return tuple(_read(f"answers_{split}.txt"))
