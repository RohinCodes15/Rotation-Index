"""Download and cache 2024-25 NBA regular-season lineup data, then check it.

Source: the public pbpstats.com API (no key). The pbpstats Python library itself
needs stats.nba.com / cdn.nba.com, which did not respond from this machine.

Run:  python data_loader.py
Writes:
  data/raw/games.json             list of every game
  data/raw/lineups/<GameId>.json  one file per game (cached, never re-downloaded)
  data/lineups.csv                one row per (game, team, five-man lineup)
  data/status.json                what loaded and what failed the checks
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

SEASON = "2024-25"
SEASON_TYPE = "Regular Season"
API = "https://api.pbpstats.com"

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
GAME_DIR = RAW_DIR / "lineups"


def get_json(url, params):
    """GET a URL, retrying a few times because the free API sometimes times out."""
    for attempt in range(4):
        try:
            # The API answers the default python-requests user agent slowly or with errors.
            response = requests.get(url, params=params, timeout=60, headers={"User-Agent": "rotation-index/0.1"})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == 3:
                raise
            print(f"  retrying after error: {error}")
            time.sleep(5 * (attempt + 1))


def load_games():
    path = RAW_DIR / "games.json"
    if not path.exists():
        print("Downloading game list...")
        data = get_json(f"{API}/get-games/nba", {"Season": SEASON, "SeasonType": SEASON_TYPE})
        path.write_text(json.dumps(data))
    return json.loads(path.read_text())["results"]


def load_game_lineups(game_id):
    path = GAME_DIR / f"{game_id}.json"
    if not path.exists():
        data = get_json(f"{API}/get-game-stats", {"GameId": game_id, "Type": "Lineup"})
        path.write_text(json.dumps(data))
        time.sleep(0.3)  # be polite to a free API
    return json.loads(path.read_text())


def lineup_rows(game, lineup_data):
    """Turn one game's full-game lineup stats into flat rows."""
    rows = []
    sides = {
        "Home": (game["HomeTeamAbbreviation"], game["AwayTeamAbbreviation"]),
        "Away": (game["AwayTeamAbbreviation"], game["HomeTeamAbbreviation"]),
    }
    for side, (team, opponent) in sides.items():
        for lineup in lineup_data["stats"][side]["FullGame"]:
            rows.append({
                "game_id": game["GameId"],
                "date": game["Date"],
                "team": team,
                "opponent": opponent,
                "is_home": side == "Home",
                "player_ids": lineup["EntityId"],
                "player_names": lineup["Name"],
                "off_poss": lineup.get("OffPoss", 0),
                "def_poss": lineup.get("DefPoss", 0),
                "points_for": lineup.get("Points", 0),
                # PlusMinus = points for - points against
                "points_against": lineup.get("Points", 0) - lineup.get("PlusMinus", 0),
            })
    return rows


def check_game(game, rows):
    """Return a list of problems with one game's lineup data (empty list = fine)."""
    problems = []
    for side in ("Home", "Away"):
        team = game[f"{side}TeamAbbreviation"]
        team_rows = [r for r in rows if r["team"] == team]
        if not team_rows:
            problems.append(f"{team}: no lineups")
            continue
        for r in team_rows:
            ids = r["player_ids"].split("-")
            if len(set(ids)) != 5 or len(r["player_names"].split(", ")) != 5:
                problems.append(f"{team}: lineup without 5 distinct players ({r['player_ids']})")
        points = sum(r["points_for"] for r in team_rows)
        if points != game[f"{side}Points"]:
            problems.append(f"{team}: lineup points {points} != final score {game[f'{side}Points']}")
        poss = sum(r["off_poss"] for r in team_rows)
        if abs(poss - game[f"{side}Possessions"]) > 2:
            problems.append(f"{team}: lineup possessions {poss} != game possessions {game[f'{side}Possessions']}")
    return problems


def main():
    GAME_DIR.mkdir(parents=True, exist_ok=True)
    games = load_games()
    print(f"{len(games)} games in the {SEASON} {SEASON_TYPE}")

    all_rows, bad_games, failed_downloads = [], {}, {}
    for i, game in enumerate(games, start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(games)} games")
        try:
            rows = lineup_rows(game, load_game_lineups(game["GameId"]))
        except (requests.RequestException, KeyError, ValueError) as error:
            failed_downloads[game["GameId"]] = str(error)
            continue
        problems = check_game(game, rows)
        if problems:
            bad_games[game["GameId"]] = problems  # excluded from the model, listed on the dashboard
        else:
            all_rows.extend(rows)

    lineups = pd.DataFrame(all_rows)
    zero_poss = int(((lineups.off_poss + lineups.def_poss) == 0).sum()) if len(lineups) else 0
    if len(lineups):
        lineups = lineups[(lineups.off_poss + lineups.def_poss) > 0]
    lineups.to_csv(DATA_DIR / "lineups.csv", index=False)

    status = {
        "season": SEASON,
        "season_type": SEASON_TYPE,
        "source": "api.pbpstats.com (get-games, get-game-stats Type=Lineup)",
        "loaded_at": time.strftime("%Y-%m-%d %H:%M"),
        "games_in_season": len(games),
        "games_used": int(lineups.game_id.nunique()) if len(lineups) else 0,
        "failed_downloads": failed_downloads,
        "games_failing_checks": bad_games,
        "zero_possession_lineups_dropped": zero_poss,
        "lineup_rows": len(lineups),
        "offensive_possessions": int(lineups.off_poss.sum()) if len(lineups) else 0,
    }
    (DATA_DIR / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in status.items()}, indent=2))


if __name__ == "__main__":
    main()
