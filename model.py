"""Fit a regularized adjusted plus-minus (RAPM-style) baseline.

Run:  python model.py        (after python data_loader.py)

Each row of data/lineups.csv is one five-man lineup in one game. For that row:
  y      = net points per 100 possessions  = 100 * (points_for - points_against) / possessions
  weight = possessions                     = (off_poss + def_poss) / 2
  X      = +1 for each of the 5 players, +1 for the opponent team, +1 if at home

Ridge regression solves  (X'WX + lambda*I) b = X'Wy  and shrinks every player
toward 0 (an average player). lambda is picked by holding out 20% of games.

Writes data/rapm.csv (one row per player) and data/model.json (settings + fit).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import season_config as season

DATA_DIR = Path(__file__).parent / "data"
LAMBDAS = [500, 1000, 2000, 4000, 8000, 16000]
LOW_SAMPLE_POSSESSIONS = 1000


def build_matrix(lineups):
    """Return X (rows x columns), y, weights and the column names."""
    lineups = lineups.reset_index(drop=True)
    player_lists = lineups.player_ids.astype(str).str.split("-")
    players = sorted({p for ids in player_lists for p in ids})
    teams = sorted(lineups.opponent.unique())
    columns = ["intercept", "home"] + [f"player:{p}" for p in players] + [f"opponent:{t}" for t in teams]
    index = {name: i for i, name in enumerate(columns)}

    X = np.zeros((len(lineups), len(columns)))
    X[:, index["intercept"]] = 1
    X[:, index["home"]] = lineups.is_home.astype(float)
    for row, ids in enumerate(player_lists):
        for p in ids:
            X[row, index[f"player:{p}"]] = 1
        X[row, index[f"opponent:{lineups.opponent[row]}"]] = 1

    possessions = (lineups.off_poss + lineups.def_poss) / 2
    y = 100 * (lineups.points_for - lineups.points_against) / possessions
    return X, y.to_numpy(), possessions.to_numpy(), columns


def fit_ridge(X, y, w, lam):
    """Weighted ridge regression. The intercept (column 0) is not shrunk."""
    penalty = lam * np.eye(X.shape[1])
    penalty[0, 0] = 0
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw + penalty, Xw.T @ y)


def pick_lambda(X, y, w, game_ids, seed=0):
    """Hold out 20% of games; return the lambda with the lowest weighted error, and all errors."""
    games = np.unique(game_ids)
    test_games = np.random.default_rng(seed).choice(games, size=len(games) // 5, replace=False)
    test = np.isin(game_ids, test_games)
    errors = {}
    for lam in LAMBDAS:
        b = fit_ridge(X[~test], y[~test], w[~test], lam)
        errors[lam] = float(np.average((y[test] - X[test] @ b) ** 2, weights=w[test]))
    return min(errors, key=errors.get), errors


def player_table(lineups, columns, coefs):
    """One row per player: name, possessions on court, estimate."""
    names, possessions = {}, {}
    for ids, names_str, off, deff in zip(lineups.player_ids.astype(str), lineups.player_names,
                                         lineups.off_poss, lineups.def_poss):
        for p, name in zip(ids.split("-"), names_str.split(", ")):
            names[p] = name
            possessions[p] = possessions.get(p, 0) + (off + deff) / 2

    rows = [
        {"player_id": c.split(":")[1], "name": names[c.split(":")[1]],
         "possessions": round(possessions[c.split(":")[1]]), "rapm": round(b, 2)}
        for c, b in zip(columns, coefs) if c.startswith("player:")
    ]
    table = pd.DataFrame(rows).sort_values("rapm", ascending=False)
    table["low_sample"] = table.possessions < LOW_SAMPLE_POSSESSIONS
    return table


def main():
    lineups = pd.read_csv(DATA_DIR / "lineups.csv", dtype={"game_id": str})
    season.require_manifest([season.require_game(gid) for gid in lineups.game_id.unique()])
    X, y, w, columns = build_matrix(lineups)
    lam, errors = pick_lambda(X, y, w, lineups.game_id.to_numpy())
    coefs = fit_ridge(X, y, w, lam)

    table = player_table(lineups, columns, coefs)
    table.to_csv(DATA_DIR / "rapm.csv", index=False)
    model = {
        "lambda": lam,
        "holdout_weighted_mse_by_lambda": errors,
        "intercept": round(float(coefs[0]), 2),
        "home_court": round(float(coefs[columns.index("home")]), 2),
        "players": len(table),
        "rows": len(lineups),
        "low_sample_possessions": LOW_SAMPLE_POSSESSIONS,
    }
    (DATA_DIR / "model.json").write_text(json.dumps(model, indent=2))
    print(json.dumps(model, indent=2))
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
