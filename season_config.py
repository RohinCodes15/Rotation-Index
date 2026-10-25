"""Configured season and authoritative, locally pinned regular-season manifest.

Source: https://api.pbpstats.com/get-games/nba?Season=2025-26&SeasonType=Regular%20Season
Retrieved 2026-10-24. Production caches are separate and are never migrated here.
"""
import json
import hashlib
from pathlib import Path

SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
GAME_PREFIX = "00225"
MANIFEST_PATH = Path(__file__).parent / "manifests/2025-26-regular-season.json"
MANIFEST_SHA256 = "962fe0f930e5e82c8599ef80e804d9c4c57732e0f7869d1a62104e28d18bdf1f"
_manifest_bytes = MANIFEST_PATH.read_bytes()
if hashlib.sha256(_manifest_bytes).hexdigest() != MANIFEST_SHA256:
    raise ValueError("configured season manifest differs from the pinned authoritative download")
_entries = json.loads(_manifest_bytes)["results"]
GAMES = {g["GameId"]: g for g in _entries}
if (not GAMES or len(GAMES) != len(_entries)
        or any(not isinstance(gid, str) or len(gid) != 10 or not gid.isdigit()
               or not gid.startswith(GAME_PREFIX) for gid in GAMES)):
    raise ValueError("invalid configured season manifest")


def require_game(game_id):
    if not isinstance(game_id, str) or game_id not in GAMES:
        raise ValueError(f"{game_id!r} is not in the authoritative {SEASON} manifest")
    return GAMES[game_id]


def require_metadata(game):
    """Bind metadata to the manifest, not just to a season-looking filename."""
    if not isinstance(game, dict) or game != require_game(game.get("GameId")):
        raise ValueError(f"game metadata does not match the {SEASON} manifest")


def require_manifest(games):
    if not isinstance(games, list):
        raise ValueError("game manifest must be a list")
    for game in games:
        require_metadata(game)
    ids = [g["GameId"] for g in games]
    if len(ids) != len(GAMES) or set(ids) != set(GAMES):
        raise ValueError(f"expected {len(GAMES):,} distinct {SEASON} manifest games")
    return sorted(ids)
