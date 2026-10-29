from wordle_rl.scoring import feedback_for, is_consistent, score_components, stop_reason


def test_stop_reason():
    h = [("crane", feedback_for("crane", "slate"))]
    ep = {"history": h, "solved": False, "invalid": 0, "done": False}
    assert stop_reason(ep, 8000, None, 8192, 10) == "out_of_budget"
    assert stop_reason(ep, 1000, {"role": "assistant", "content": "I think it is slate"}, 8192, 10) == "no_tool_call"
    assert stop_reason(ep, 7000, {"role": "assistant", "tool_calls": [{"type": "function"}]}, 8192, 10) == "out_of_budget"
    assert stop_reason({**ep, "invalid": 9}, 1000, None, 8192, 10) == "tool_call_cap"
    assert stop_reason({**ep, "done": True}, 1000, None, 8192, 10) == "lost_invalid_streak"
    assert stop_reason({**ep, "done": True, "history": h * 6}, 1000, None, 8192, 10) == "lost_6_guesses"
    assert stop_reason({**ep, "solved": True, "done": True}, 1000, None, 8192, 10) == "solved"


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
