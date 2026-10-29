"""Wordle environment for TRL's GRPOTrainer(environment_factory=...).

Game logic (feedback colouring, repeat/length/format checks, turn limit) is TextArena's
Wordle-v0. Changed here:
  - secrets come from the dataset row (`secret` column), so train/eval words stay separate;
  - guesses are checked against the standard Wordle guess list instead of NLTK;
  - the model calls a `guess(word)` tool; brackets for TextArena are added here;
  - TextArena's own reward is ignored; scoring.py scores the finished episode.
"""

import textarena as ta
from textarena.envs.Wordle.env import WordleEnv as TextArenaWordle

from wordle_rl.words import allowed_guesses

NUM_GUESSES = 6
# Invalid guesses in a row before TextArena ends the game as a loss (its default is 1).
ERROR_ALLOWANCE = 3


class StandardWordle(TextArenaWordle):
    """TextArena Wordle with a caller-chosen secret and the standard guess list."""

    def __init__(self, allowed: frozenset[str], num_guesses: int = NUM_GUESSES, error_allowance: int = ERROR_ALLOWANCE):
        ta.Env.__init__(self)  # skip TextArena's NLTK word list and dictionary loading
        self.word_length = 5
        self.num_guesses = num_guesses
        self.error_allowance = error_allowance
        self.allowed = allowed

    def _check_word(self, word: str) -> bool:
        return word in self.allowed

    def reset(self, secret: str, seed: int | None = None):
        self.state = ta.SinglePlayerState(num_players=1, seed=seed, error_allowance=self.error_allowance)
        game_state = {
            "secret_word": secret,
            "guess_history": [],
            "word_length": self.word_length,
            "num_guesses": self.num_guesses,
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)
        self.get_observation()  # drop TextArena's own intro prompt; ours is in prompt.py


class WordleToolEnv:
    """One instance per rollout; TRL reuses instances and calls reset() before each game."""

    def __init__(self):
        self._game = StandardWordle(allowed_guesses())
        self._invalid = 0

    def reset(self, secret: str, **kwargs) -> str | None:
        self._game.reset(secret.lower())
        self._invalid = 0
        return None  # the rules are already in the prompt

    def guess(self, word: str) -> str:
        """
        Submit one guess in the current Wordle game.

        Args:
            word: A 5-letter English word, for example "crane".

        Returns:
            Feedback for each letter (G = right spot, Y = wrong spot, X = not in the word), or the reason the guess was rejected.
        """
        state = self._game.state
        if state.done:
            raise ValueError("The game is over. Do not call guess again.")
        word = str(word).strip().strip("[]'\"` ").lower()
        guesses_before = len(state.game_state["guess_history"])
        self._game.step(f"[{word}]")
        if len(state.game_state["guess_history"]) == guesses_before:
            self._invalid += 1

        _, messages = self._game.get_observation()
        lines = [msg for _, msg, kind in messages if kind != ta.ObservationType.PLAYER_ACTION]
        if state.done:
            lines.append(f"Game over. {state.game_info[0]['reason']}")
        return "\n".join(lines)

    def _episode(self) -> dict:
        """Finished-game summary for scoring.py (private: TRL exposes public methods as tools)."""
        state = self._game.state
        history = [(w, list(fb)) for w, fb in state.game_state["guess_history"]]
        return {
            "history": history,
            "solved": bool(history) and all(f == "G" for f in history[-1][1]),
            "invalid": self._invalid,
            "done": state.done,
            "secret": state.game_state["secret_word"],
        }
