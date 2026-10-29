from wordle_rl.scoring import feedback_for, is_consistent, score_components


def test_feedback_duplicates():
    # Only one E in the secret: the green E is counted first, the other E is grey.
    assert feedback_for("geese", "those") == list("XXXGG")
    assert feedback_for("speed", "abide") == list("XXYXY")
    assert feedback_for("crane", "crane") == list("GGGGG")


def test_consistency():
    history = [("crane", feedback_for("crane", "slate"))]  # X X G X G
    assert is_consistent("slate", history)
    assert not is_consistent("crane", history)
    assert not is_consistent("pound", history)


def test_components_solved_fast_and_clean():
    history = [(w, feedback_for(w, "slate")) for w in ("crane", "slate")]
    s = score_components({"history": history, "solved": True, "invalid": 0, "done": True})
    assert s == {"solved": 1.0, "speed": 0.8, "consistency": 1.0, "invalid": 0.0, "unfinished": 0.0, "letters": 1.0}


def test_components_abandoned_with_invalid():
    history = [("crane", feedback_for("crane", "slate"))]
    s = score_components({"history": history, "solved": False, "invalid": 5, "done": False})
    assert s["solved"] == 0.0 and s["speed"] == 0.0
    assert s["invalid"] == -1.0
    assert s["unfinished"] == -1.0
    assert s["letters"] == 0.4
