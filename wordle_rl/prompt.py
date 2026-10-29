"""The instructions the model sees at the start of every game."""

RULES = """Let's play Wordle. Find the secret 5-letter English word in at most 6 guesses.

Make each guess by calling the `guess` tool with one 5-letter word. After each guess you get one letter of feedback per position:
- G: right letter, right position
- Y: the letter is in the word, but in a different position
- X: the letter is not in the word (or the word has no more copies of it)

Guesses must be real English words and must not repeat an earlier guess. Use all the feedback so far: keep green letters in place, move yellow letters to new positions, and avoid grey letters. Make exactly one guess per turn, and stop once the game is over."""


def build_prompt() -> list[dict]:
    # Thinking is switched by chat_template_kwargs={"enable_thinking": ...}, not by the prompt text.
    return [{"role": "user", "content": RULES}]
