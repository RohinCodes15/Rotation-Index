from pathlib import Path
from pbpstats.client import Client

data_folder = Path("data/pbpstats")

for folder in ("game_details", "overrides", "pbp", "schedule"):
    (data_folder / folder).mkdir(parents=True, exist_ok=True)

settings = {
    "dir": str(data_folder),
    "Games": {
        "source": "web",
        "data_provider": "data_nba",
    },
}

client = Client(settings)
season = client.Season("nba", "2024-25", "Regular Season")
games = season.games.final_games

print(f"Final games found: {len(games)}")