"""Tiny synthetic cache files for gate tests; never a claimed NBA season.

Only use inside temporary test directories. This simulates an audit attestation
so tests can exercise file freshness without downloading or fabricating raw NBA data.
"""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import possessions_loader as loader
import season_config as season

from possessions_loader import _audit_lineage, version


def attach_lineage_fixture(status, root):
    raw_dir, normalized_dir = root / "pbpstats/pbp", root / "games"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for gid, record in status["game_results"].items():
        if record["status"] != "validated":
            continue
        raw = raw_dir / f"live_{gid}.json"
        normalized_path = normalized_dir / f"{gid}.json"
        normalized = {"game": season.require_game(gid), "pbpstats_version": version("pbpstats"),
                      "schema_version": 1, "possessions": [],
                      "applied_corrections": record.get("applied_corrections", [])}
        # Preserve files between display requests so freshness tests are meaningful.
        if not raw.exists():
            raw.write_text(json.dumps({"game": {"gameId": gid, "actions": []}}))
        if not normalized_path.exists():
            normalized_path.write_text(json.dumps(normalized))
        record["lineage"] = _audit_lineage(
            raw, normalized_path, result=normalized,
            raw_before=hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
    return status


def historical_season_fixture():
    """Explicit historical configuration for tests backed by the unchanged 2024 cache.

    Production always uses the pinned 2025 manifest; historical parser/math
    regressions run under their own test-only season configuration.
    """
    stack = ExitStack()
    path = Path(__file__).parent / "data/cache/possessions/games.json"
    games = json.loads(path.read_text())["results"]
    stack.enter_context(patch.object(season, "GAMES", {g["GameId"]: g for g in games}))
    stack.enter_context(patch.object(season, "SEASON", "2024-25"))
    stack.enter_context(patch.object(loader, "SEASON", "2024-25"))
    return stack
