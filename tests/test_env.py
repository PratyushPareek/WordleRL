import pytest

from wordle_rl.env import WordleToolEnv
from wordle_rl.scoring import score_components


def play(secret, guesses):
    env = WordleToolEnv()
    env.reset(secret=secret, prompt=[])
    replies = [env.guess(g) for g in guesses]
    return env, replies


def test_win_in_two():
    env, replies = play("slate", ["crane", "slate"])
    assert "X X G X G" in replies[0]
    assert "Game over" in replies[1] and "Congratulations" in replies[1]
    ep = env._episode()
    assert ep["solved"] and ep["done"] and ep["invalid"] == 0
    assert score_components(ep)["speed"] == 0.8
    with pytest.raises(ValueError):
        env.guess("crane")


def test_invalid_guesses_do_not_use_turns():
    env, replies = play("slate", ["xxxxx", "cran", "crane", "crane"])
    assert "not an English word" in replies[0]
    assert "exactly 5 letters" in replies[1]
    assert "already guessed" in replies[3]
    ep = env._episode()
    assert len(ep["history"]) == 1 and ep["invalid"] == 3 and not ep["done"]


def test_brackets_and_case_are_accepted():
    env, _ = play("slate", ["[CRANE]"])
    assert env._episode()["history"][0][0] == "crane"


def test_too_many_invalid_in_a_row_ends_game():
    env, replies = play("slate", ["aaaaa", "bbbbb", "ccccc", "ddddd"])
    ep = env._episode()
    assert ep["done"] and not ep["solved"] and ep["invalid"] == 4
    assert "Game over" in replies[-1]


def test_loss_after_six_guesses():
    env, replies = play("slate", ["crane", "pound", "fight", "jumbo", "wreck", "glyph"])
    ep = env._episode()
    assert ep["done"] and not ep["solved"]
    assert "secret word was" in replies[-1]
    s = score_components(ep)
    assert s["solved"] == 0 and s["unfinished"] == 0


def test_reset_reuses_instance():
    env, _ = play("slate", ["crane", "slate"])
    env.reset(secret="pound", prompt=[])
    ep = env._episode()
    assert ep["history"] == [] and ep["invalid"] == 0 and not ep["done"] and ep["secret"] == "pound"


def test_only_guess_is_exposed_as_tool():
    import inspect

    public = [n for n, _ in inspect.getmembers(WordleToolEnv, inspect.isfunction) if not n.startswith("_")]
    assert sorted(public) == ["guess", "reset"]
