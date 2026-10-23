"""Quick checks on synthetic data (never shown in the app). Run: python test_model.py"""

import numpy as np
import pandas as pd

from data_loader import check_game
from model import build_matrix, fit_ridge


def test_ridge_recovers_known_player_effects():
    # 10 players with known true values; random lineups, outcome = sum of the five values.
    rng = np.random.default_rng(1)
    truth = {str(p): v for p, v in zip(range(10), [8, 5, 3, 1, 0, 0, -1, -3, -5, -8])}
    rows = []
    for g in range(400):
        five = rng.choice(list(truth), size=5, replace=False)
        net = sum(truth[p] for p in five)  # per 100 possessions
        rows.append({"game_id": str(g), "opponent": "AAA", "is_home": False,
                     "player_ids": "-".join(five), "off_poss": 50, "def_poss": 50,
                     # 50 possessions, so the point margin is net / 2
                     "points_for": 100 + net / 4, "points_against": 100 - net / 4})
    X, y, w, columns = build_matrix(pd.DataFrame(rows))
    coefs = dict(zip(columns, fit_ridge(X, y, w, lam=1)))
    estimates = [coefs[f"player:{p}"] for p in truth]
    # Effects are only identified up to a constant shared with the intercept, so compare centered values.
    assert np.allclose(np.array(estimates) - np.mean(estimates), list(truth.values()), atol=0.1)


def test_check_game_flags_score_mismatch():
    game = {"HomeTeamAbbreviation": "AAA", "AwayTeamAbbreviation": "BBB",
            "HomePoints": 100, "AwayPoints": 90, "HomePossessions": 50, "AwayPossessions": 50}
    rows = [{"team": "AAA", "player_ids": "1-2-3-4-5", "player_names": "a, b, c, d, e",
             "points_for": 100, "off_poss": 50},
            {"team": "BBB", "player_ids": "6-7-8-9-9", "player_names": "f, g, h, i, i",
             "points_for": 80, "off_poss": 50}]
    problems = check_game(game, rows)
    assert any("final score" in p for p in problems)
    assert any("5 distinct" in p for p in problems)
    assert not any(p.startswith("AAA") for p in problems)


if __name__ == "__main__":
    test_ridge_recovers_known_player_effects()
    test_check_game_flags_score_mismatch()
    print("all checks passed")
