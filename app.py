import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from possession_model import (
    EXPECTED_REGULAR_SEASON_GAMES,
    IncompleteSeasonError,
    check_game_inputs,
    require_complete_season,
)

app = Flask(__name__)
DATA_DIR = Path(__file__).parent / "data"
POSSESSION_COLUMNS = ["player_id", "offense_possessions", "defense_possessions",
                      "offense_per_100", "defense_per_100", "net_per_100"]
# In-process only, keyed by the name-source files' state. Nothing is written.
_name_cache = {}
# In-process only: per-file digests and gate results, keyed by file stat signatures
# so a full season is not re-hashed on every request. Any file change refreshes them.
_file_cache = {}
_gate_cache = {}


def read_json(name):
    path = DATA_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def cached_player_names(player_ids):
    """Map player IDs to names from local caches only; display, never modeling.

    Prefer full names from the lineup cache (EntityId paired with Name), then
    fill gaps with the raw live feeds' playerNameI. Unreadable files are skipped;
    unnamed players fall back to their ID in the template.
    """
    lineup_files = sorted((DATA_DIR / "raw/lineups").glob("*.json"))
    live_files = sorted((DATA_DIR / "cache/possessions/pbpstats/pbp").glob("live_*.json"))
    wanted = frozenset(int(p) for p in player_ids)
    key = (wanted, tuple((str(f), f.stat().st_mtime_ns) for f in lineup_files + live_files))
    if key in _name_cache:
        return _name_cache[key]
    names = {}
    for path in lineup_files:
        try:
            stats = json.loads(path.read_text())["stats"]
            lineups = [l for side in ("Home", "Away") for l in stats[side]["FullGame"]]
            for lineup in lineups:
                ids, full_names = lineup["EntityId"].split("-"), lineup["Name"].split(", ")
                if len(ids) == len(full_names) == 5:
                    for player_id, name in zip(ids, full_names):
                        if int(player_id) in wanted and name.strip():
                            names.setdefault(int(player_id), name.strip())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    for path in live_files:
        if wanted <= names.keys():
            break
        try:
            for action in json.loads(path.read_text())["game"]["actions"]:
                player_id, name = action.get("personId"), action.get("playerNameI")
                if type(player_id) is int and player_id in wanted and isinstance(name, str) and name.strip():
                    names.setdefault(player_id, name.strip())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    _name_cache.clear()
    _name_cache[key] = names
    return names


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _id_ranges(game_ids):
    """Collapse sorted numeric game IDs into 'first–last' runs for display."""
    ranges = []
    for game_id in sorted(game_ids):
        if ranges and int(game_id) == int(ranges[-1][1]) + 1:
            ranges[-1][1] = game_id
        else:
            ranges.append([game_id, game_id])
    return [first if first == last else f"{first}–{last}" for first, last in ranges]


def input_freshness(status, described):
    """Per-game view of whether normalized model inputs match the audited cache."""
    expected_ids = status.get("expected_game_ids")
    results = status.get("game_results") if isinstance(status.get("game_results"), dict) else {}
    if not isinstance(expected_ids, list) or not expected_ids:
        return {"checked": False, "counts": {}, "problems": [], "fresh": False, "signature": ()}
    root = DATA_DIR / "cache/possessions"
    counts = dict.fromkeys(("fresh", "stale", "missing", "missing_raw", "unverifiable"), 0)
    problems, signature = [], []
    for game_id in sorted(set(expected_ids)):
        # Same comparison the model gate runs; the cache only skips re-hashing unchanged files.
        check = check_game_inputs(game_id, results.get(game_id), root / "pbpstats/pbp", root / "games",
                                  cache=_file_cache)
        counts[check.category] += 1
        signature.extend(check.signatures)
        if check.category not in ("fresh", "missing_raw"):
            problems.append(described(game_id, {"category": check.category, "reason": check.reason}))
    return {"checked": True, "counts": counts, "problems": problems, "signature": tuple(signature),
            "fresh": counts["fresh"] == len(set(expected_ids))}


def possession_audit(status, verification):
    """Plain-language view of the cache audit and model-input freshness; local files only.

    `ready` requires both the model's own season gate and this page's per-game
    freshness check, so estimates never show when either one fails.
    """
    try:
        listed = read_json("cache/possessions/games.json") or {}
        games = {g["GameId"]: {"date": g.get("Date"),
                               "matchup": f"{g.get('AwayTeamAbbreviation')} at {g.get('HomeTeamAbbreviation')}"}
                 for g in listed.get("results", []) if isinstance(g, dict) and "GameId" in g}
    except (ValueError, AttributeError, TypeError):
        games = {}

    def described(game_id, fields):
        return {**fields, "game_id": game_id, **games.get(game_id, {"date": None, "matchup": None})}

    expected = _count(status.get("games_expected")) or EXPECTED_REGULAR_SEASON_GAMES
    counts = {}
    for category in ("cached", "validated", "failed", "missing"):
        ids = status.get(f"{category}_game_ids")
        counts[category] = _count(status.get(f"games_{category}"))
        if counts[category] is None and isinstance(ids, list):
            counts[category] = len(ids)
    results = status.get("game_results") if isinstance(status.get("game_results"), dict) else {}
    errors = status.get("validation_errors") if isinstance(status.get("validation_errors"), dict) else {}
    failed_ids = set(errors) | {gid for gid, r in results.items() if isinstance(r, dict) and r.get("status") == "failed"}
    if isinstance(status.get("failed_game_ids"), list):
        failed_ids |= set(status["failed_game_ids"])
    failures = [described(gid, {"error": errors.get(gid) or (results.get(gid) or {}).get("error")
                                         or "No error message recorded."})
                for gid in sorted(failed_ids)]
    corrections = [described(gid, correction)
                   for gid, result in sorted(results.items()) if isinstance(result, dict)
                   for correction in result.get("applied_corrections") or [] if isinstance(correction, dict)]
    if not results and verification:
        corrections = [described(c.get("game_id"), c) for c in verification.get("applied_corrections") or []
                       if isinstance(c, dict)]
    missing_ids = status.get("missing_game_ids") if isinstance(status.get("missing_game_ids"), list) else []
    differences = status.get("possession_count_differences")

    inputs = input_freshness(status, described)
    status_path = DATA_DIR / "cache/possessions/status.json"
    try:
        stat = status_path.stat()
        gate_key = (str(status_path), stat.st_mtime_ns, stat.st_size, stat.st_ino, inputs["signature"])
    except OSError:
        gate_key = None
    if gate_key is None or gate_key not in _gate_cache:
        try:
            require_complete_season(status_path)
            gate_passed = True
        except (IncompleteSeasonError, OSError, ValueError):
            gate_passed = False
        if gate_key is not None:
            _gate_cache.clear()
            _gate_cache[gate_key] = gate_passed
    else:
        gate_passed = _gate_cache[gate_key]
    ready = gate_passed and inputs["fresh"]
    raw_complete = (status.get("mode") == "offline_audit" and status.get("audit_complete") is True
                    and counts["validated"] == expected and not counts["failed"] and not counts["missing"]
                    and not failures)
    reasons = []
    if not ready:
        if status.get("mode") != "offline_audit":
            reasons.append("This status was not written by the offline audit, so it cannot certify the season. "
                           "Run the offline audit to check every listed game against the local cache.")
        elif status.get("audit_complete") is not True:
            reasons.append("The last offline audit did not finish checking the game list.")
        if counts["missing"]:
            missing = counts["missing"]
            reasons.append(f"{missing:,} {'game has' if missing == 1 else 'games have'} no cached play-by-play yet.")
        failed = max(counts["failed"] or 0, len(failures))
        if failed:
            reasons.append(f"{failed:,} cached {'game' if failed == 1 else 'games'} failed validation; see the failures below.")
        ic = inputs["counts"]
        if not inputs["checked"]:
            reasons.append("The audit report has no game list to check model inputs against.")
        for key, singular, plural in (
            ("stale", "game's model inputs no longer match the audit (stale).",
             "games' model inputs no longer match the audit (stale)."),
            ("missing", "validated game has no normalized model-input file.",
             "validated games have no normalized model-input file."),
            ("unverifiable", "game's model inputs could not be verified against the audit.",
             "games' model inputs could not be verified against the audit."),
        ):
            if ic.get(key):
                reasons.append(f"{ic[key]:,} {singular if ic[key] == 1 else plural} See model inputs below.")
        if not reasons:
            reasons.append("The audit report's game lists and per-game results do not agree, so it cannot certify the season.")
    validated = counts["validated"] or 0

    def share(n):
        return round(100 * min(max(n or 0, 0), expected) / expected, 2)

    return {
        "ready": ready, "raw_complete": raw_complete, "inputs": inputs,
        "reasons": reasons, "expected": expected, "counts": counts,
        "progress": {"validated": share(validated), "failed": share(counts["failed"]),
                     "missing": share(counts["missing"])},
        "failures": failures, "corrections": corrections,
        "missing_ranges": _id_ranges(g for g in missing_ids if isinstance(g, str) and g.isdigit()),
        "differences": [described(gid, {"by_team": d}) for gid, d in sorted(differences.items()) if isinstance(d, dict)]
                       if isinstance(differences, dict) else [],
        "mode": status.get("mode"), "message": status.get("error"),
    }


def possession_outputs(possession_status, ready):
    """Read saved possession-model outputs; None unless they match the audited cache.

    Returns (metadata, table, problem, failed_checks). Display only: never fits or writes.
    """
    if not ready:
        return None, None, None, []
    table_path = DATA_DIR / "possession_players.csv"
    try:
        metadata = read_json("possession_model.json")
        table = pd.read_csv(table_path) if table_path.exists() else None
    except (ValueError, pd.errors.ParserError):
        return None, None, "mismatch", ["saved outputs could not be read"]
    if metadata is None or table is None:
        return None, None, "missing", []
    if not isinstance(metadata, dict):
        return None, None, "mismatch", ["model metadata is not a JSON object"]
    status = possession_status
    split = [metadata.get("training_game_ids"), metadata.get("holdout_game_ids")]
    split_ok = all(isinstance(ids, list) for ids in split)

    def table_ok():
        if list(table.columns) != POSSESSION_COLUMNS or len(table) != metadata.get("players"):
            return False
        if not table[POSSESSION_COLUMNS].notna().all().all() or not table.player_id.is_unique:
            return False
        counts = table[["offense_possessions", "defense_possessions"]]
        return bool((counts >= 0).all().all() and ((table.net_per_100 - table.offense_per_100
                                                     - table.defense_per_100).abs() < 1e-9).all())

    checks = {
        "model type and schema version": metadata.get("model_type") == "possession_ridge"
                                         and metadata.get("schema_version") == 1,
        "season and season type": (metadata.get("season"), metadata.get("season_type"))
                                  == (status.get("season"), status.get("season_type")),
        "data source": metadata.get("source") == status.get("source"),
        "validated and used game counts": metadata.get("games_validated") == status.get("games_validated")
                                          and metadata.get("games_used") == EXPECTED_REGULAR_SEASON_GAMES,
        "audited game IDs": split_ok and not set(split[0]) & set(split[1])
                            and len(split[0]) + len(split[1]) == EXPECTED_REGULAR_SEASON_GAMES
                            and set(split[0]) | set(split[1]) == set(status.get("validated_game_ids") or []),
        "model-eligible possessions": metadata.get("eligible_possessions") == status.get("model_eligible_possessions"),
        "defense sign and net definition": metadata.get("defense_sign") == "positive means points prevented"
                                           and metadata.get("net_definition") == "offense_per_100 + defense_per_100",
        "player table": table_ok(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return None, None, "mismatch", failed
    return metadata, table, None, []


@app.route("/")
def home():
    status = read_json("status.json")
    possession_status = read_json("cache/possessions/status.json")
    verification = read_json("cache/possessions/verification.json")
    model = read_json("model.json") if status and possession_status is None else None
    players = None
    min_poss = request.args.get("min_poss", default=1000, type=int)
    if status and possession_status is None and model and (DATA_DIR / "rapm.csv").exists():
        table = pd.read_csv(DATA_DIR / "rapm.csv")
        players = table[table.possessions >= min_poss].to_dict("records")
    possession_model = possession_players = possession_problem = audit = None
    failed_checks = []
    if possession_status is not None:
        audit = possession_audit(possession_status if isinstance(possession_status, dict) else {}, verification)
        possession_model, table, possession_problem, failed_checks = possession_outputs(possession_status, audit["ready"])
        if table is not None:
            on_court = table.offense_possessions + table.defense_possessions
            possession_players = (table[on_court >= min_poss]
                                  .sort_values(["net_per_100", "player_id"], ascending=[False, True])
                                  .to_dict("records"))
            names = cached_player_names(table.player_id)
            for player in possession_players:
                player["name"] = names.get(int(player["player_id"]))
    return render_template(
        "index.html",
        status=status,
        possession_status=possession_status,
        verification=verification,
        model=model,
        players=players,
        possession_model=possession_model,
        possession_players=possession_players,
        possession_problem=possession_problem,
        failed_checks=failed_checks,
        audit=audit,
        min_poss=min_poss,
    )


if __name__ == "__main__":
    app.run(debug=True)
