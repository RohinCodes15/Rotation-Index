"""Validated possession-level inputs for the Rotation Index RAPM model.

The public ridge fitting pipeline requires a complete season cache before
selecting a penalty, fitting all data, or writing possession-model outputs.
"""

import json
import hashlib
import csv
import io
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
GAME_CACHE = BASE / "data/cache/possessions/games"
RAW_PBP_CACHE = BASE / "data/cache/possessions/pbpstats/pbp"
STATUS_FILE = BASE / "data/cache/possessions/status.json"
EXPECTED_REGULAR_SEASON_GAMES = 1230
# Sum-of-squared-errors ridge penalties, shared by offense and defense blocks.
RIDGE_PENALTY_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


class IncompleteSeasonError(ValueError):
    """Raised when a season-level model is requested from a partial cache."""


@dataclass(frozen=True)
class PossessionRow:
    """One model-eligible offensive possession with both five-player lineups."""

    game_id: str
    period: int
    number: int
    offense_team_id: str
    defense_team_id: str
    offense_players: tuple[int, ...]
    defense_players: tuple[int, ...]
    points: int


@dataclass(frozen=True)
class PossessionDesignData:
    """Compact inputs; offense and defense use separate coefficient blocks.

    Each (n, 5) index array addresses the same sorted player_ids vocabulary.
    points_per_possession is unscaled points on each single offensive possession;
    game_ids preserves row order for later game-level train/holdout splits.
    """

    player_ids: np.ndarray
    offense_indices: np.ndarray
    defense_indices: np.ndarray
    points_per_possession: np.ndarray
    game_ids: np.ndarray


def build_design_data(rows):
    """Index PossessionRows without constructing an n-by-player design matrix.

    Does not fit estimates or require a complete season: partial real caches can
    be inspected here, while require_complete_season remains the fitting gate.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("cannot build design data without possession rows")
    lineups = []
    for row in rows:
        offense = _validated_lineup(
            list(row.offense_players), side="offense", game_id=row.game_id,
            period=row.period, number=row.number,
        )
        defense = _validated_lineup(
            list(row.defense_players), side="defense", game_id=row.game_id,
            period=row.period, number=row.number,
        )
        if set(offense).intersection(defense):
            raise ValueError("player appears on both teams")
        if type(row.points) is not int or row.points < 0:
            raise ValueError("invalid point outcome")
        if not isinstance(row.game_id, str) or not row.game_id:
            raise ValueError("missing game ID")
        lineups.append((offense, defense))
    player_ids = np.array(sorted({p for sides in lineups for side in sides for p in side}), dtype=np.int64)
    player_index = {int(player): index for index, player in enumerate(player_ids)}
    return PossessionDesignData(
        player_ids=player_ids,
        offense_indices=np.array([[player_index[p] for p in offense] for offense, _ in lineups], dtype=np.int32),
        defense_indices=np.array([[player_index[p] for p in defense] for _, defense in lineups], dtype=np.int32),
        points_per_possession=np.array([row.points for row in rows], dtype=np.float64),
        game_ids=np.array([row.game_id for row in rows]),
    )


def _validated_lineup(values, *, side, game_id, period, number):
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(
            f"{game_id} Q{period} possession {number}: {side} lineup must have five players"
        )
    if any(type(player_id) is not int or player_id <= 0 for player_id in values):
        raise ValueError(
            f"{game_id} Q{period} possession {number}: {side} lineup has an invalid player ID"
        )
    if len(set(values)) != 5:
        raise ValueError(
            f"{game_id} Q{period} possession {number}: {side} lineup has duplicate players"
        )
    return tuple(sorted(values))


def build_possession_rows(game_data):
    """Extract real, countable, unambiguous possession rows from one normalized game."""
    if game_data.get("schema_version") != 1:
        raise ValueError("unsupported normalized possession schema")
    game = game_data.get("game")
    possessions = game_data.get("possessions")
    if not isinstance(game, dict) or not isinstance(possessions, list):
        raise ValueError("normalized game is missing game metadata or possessions")

    game_id = str(game.get("GameId", ""))
    if not game_id:
        raise ValueError("normalized game is missing GameId")

    rows = []
    seen = set()
    for possession in possessions:
        if not isinstance(possession, dict):
            raise ValueError(f"{game_id}: malformed possession record")
        if str(possession.get("game_id", "")) != game_id:
            raise ValueError(f"{game_id}: possession belongs to a different game")
        period = possession.get("period")
        number = possession.get("number")
        if type(period) is not int or period < 1 or type(number) is not int or number < 1:
            raise ValueError(f"{game_id}: invalid possession period or number")
        key = (period, number)
        if key in seen:
            raise ValueError(f"{game_id} Q{period}: duplicate possession number {number}")
        seen.add(key)

        eligible = possession.get("model_eligible")
        counted = possession.get("counted_possession")
        if type(eligible) is not bool or type(counted) is not bool:
            raise ValueError(f"{game_id} Q{period} possession {number}: missing eligibility flags")
        if eligible and not counted:
            raise ValueError(
                f"{game_id} Q{period} possession {number}: uncounted possession marked eligible"
            )
        if not eligible or not counted:
            continue

        offense_team_id = str(possession.get("offense_team_id", ""))
        defense_team_id = str(possession.get("defense_team_id", ""))
        if not offense_team_id or not defense_team_id or offense_team_id == defense_team_id:
            raise ValueError(f"{game_id} Q{period} possession {number}: invalid team IDs")
        points = possession.get("points")
        if type(points) is not int or points < 0:
            raise ValueError(f"{game_id} Q{period} possession {number}: invalid point outcome")

        offense_players = _validated_lineup(
            possession.get("offense_players"), side="offense", game_id=game_id,
            period=period, number=number,
        )
        defense_players = _validated_lineup(
            possession.get("defense_players"), side="defense", game_id=game_id,
            period=period, number=number,
        )
        if set(offense_players).intersection(defense_players):
            raise ValueError(f"{game_id} Q{period} possession {number}: player appears on both teams")

        rows.append(
            PossessionRow(
                game_id=game_id,
                period=period,
                number=number,
                offense_team_id=offense_team_id,
                defense_team_id=defense_team_id,
                offense_players=offense_players,
                defense_players=defense_players,
                points=points,
            )
        )
    return rows


def load_cached_possession_rows(game_cache=GAME_CACHE):
    """Read only normalized local game files; never contacts a data source."""
    game_cache = Path(game_cache)
    rows = []
    seen_games = set()
    for path in sorted(game_cache.glob("*.json")):
        game_data = json.loads(path.read_text())
        game_id = str(game_data.get("game", {}).get("GameId", ""))
        if path.stem != game_id:
            raise ValueError(f"cache filename {path.name} does not match game ID {game_id}")
        if game_id in seen_games:
            raise ValueError(f"duplicate normalized cache for game {game_id}")
        seen_games.add(game_id)
        rows.extend(build_possession_rows(game_data))
    return rows


@dataclass(frozen=True)
class CachedPlayerNames:
    """Unambiguous feed labels and unresolved ID/name conflicts, keyed by ID."""

    names: dict[int, str]
    conflicts: dict[int, dict[str, tuple[str, ...]]]


def load_cached_player_names(raw_cache=RAW_PBP_CACHE):
    """Read live_*.json only; never download, correct, or write cached data.

    Use the feed's explicit personId/playerNameI and personId/playerName pairs.
    Prefer playerNameI (e.g. 'J. Tatum'); use playerName only if no initial-name
    label exists for that ID anywhere in the cache. These are feed labels, not
    inferred full names. Compare each field separately so initials and surnames
    are not mistaken for conflicts. If either field has multiple distinct names
    for an ID, omit it from names and report all observed labels in conflicts.
    Names shared by different IDs are allowed. Team events (personId=0) and
    absent names do not create a lookup entry. Descriptions are never parsed.
    """
    paths = sorted(Path(raw_cache).glob("live_*.json"))
    if not paths:
        raise FileNotFoundError(f"no cached raw play-by-play files in {raw_cache}")
    observed = {}
    for path in paths:
        actions = json.loads(path.read_text())["game"]["actions"]
        if not isinstance(actions, list):
            raise ValueError(f"{path.name}: missing play-by-play actions")
        for event in actions:
            player_id = event.get("personId")
            if type(player_id) is not int or player_id <= 0:
                continue
            for field in ("playerNameI", "playerName"):
                name = event.get(field)
                if isinstance(name, str) and name.strip():
                    observed.setdefault(player_id, {}).setdefault(field, set()).add(name.strip())
    names, conflicts = {}, {}
    for player_id, fields in sorted(observed.items()):
        if any(len(values) > 1 for values in fields.values()):
            conflicts[player_id] = {field: tuple(sorted(values)) for field, values in fields.items()}
        else:
            values = fields.get("playerNameI", fields.get("playerName"))
            names[player_id] = next(iter(values))
    return CachedPlayerNames(names, conflicts)


def require_complete_season(status_path=STATUS_FILE, *, game_cache=None):
    """Require a finished, consistent offline audit of all 1,230 season games.

    Summary flags alone are insufficient: expected/cached/validated ID sets and
    every per-game outcome must agree, with no missing games or validation errors.
    """
    status_path = Path(status_path)
    try:
        status = json.loads(status_path.read_text())
    except (OSError, ValueError, UnicodeError) as error:
        raise IncompleteSeasonError("season cache audit is missing, unreadable, or corrupt") from error
    if not isinstance(status, dict):
        raise IncompleteSeasonError("season cache audit must be an object")
    expected = status.get("games_expected")
    validated = status.get("games_validated")
    id_sets = []
    for category in ("expected", "cached", "validated"):
        ids = status.get(f"{category}_game_ids")
        if (not isinstance(ids, list) or len(ids) != EXPECTED_REGULAR_SEASON_GAMES
                or any(not isinstance(gid, str) or len(gid) != 10
                       or not gid.startswith("00224") or not gid.isdigit() for gid in ids)):
            break
        id_sets.append(set(ids))
    ids_valid = (len(id_sets) == 3
                 and len(id_sets[0]) == EXPECTED_REGULAR_SEASON_GAMES
                 and id_sets[0] == id_sets[1] == id_sets[2])
    outcomes = status.get("game_results")
    outcomes_valid = (ids_valid and isinstance(outcomes, dict)
                      and set(outcomes) == id_sets[0]
                      and all(isinstance(result, dict) and result.get("status") == "validated"
                              and not result.get("error") for result in outcomes.values()))
    if not (
        status.get("complete") is True
        and status.get("audit_complete") is True
        and status.get("mode") == "offline_audit"
        and expected == EXPECTED_REGULAR_SEASON_GAMES
        and validated == EXPECTED_REGULAR_SEASON_GAMES
        and status.get("games_cached") == EXPECTED_REGULAR_SEASON_GAMES
        and status.get("games_failed") == 0
        and status.get("games_missing") == 0
        and status.get("failed_game_ids") == []
        and status.get("missing_game_ids") == []
        and status.get("validation_errors") == {}
        and not status.get("error")
        and outcomes_valid
    ):
        raise IncompleteSeasonError(
            f"season cache incomplete: {validated} of {expected} games validated; "
            "a complete cache audit with no failed or missing games is required; "
            "season player estimates are unavailable"
        )
    _require_current_lineage(status, status_path, game_cache=game_cache)
    return status


def _require_current_lineage(status, status_path, *, game_cache=None):
    """Check audit attestations against current files; never repair or write them.

    Uses check_game_inputs, the same per-game comparison the dashboard displays,
    without a file cache so every gate call re-reads current bytes.
    """
    normalized_root = Path(game_cache) if game_cache is not None else status_path.parent / "games"
    raw_root = status_path.parent / "pbpstats/pbp"
    for gid in status["expected_game_ids"]:
        check = check_game_inputs(gid, status["game_results"][gid], raw_root, normalized_root)
        if check.category != "fresh":
            raise IncompleteSeasonError(
                f"game {gid}: normalized inputs are missing, stale, or unverifiable "
                f"({check.category}: {check.reason}); "
                "rerun the offline audit after refreshing normalized inputs"
            )


LINEAGE_METHOD = "canonical_normalization_replay_v1"
# Unverified audit lineage statuses: (category, plain-language reason).
LINEAGE_AT_AUDIT = {
    "missing_raw": ("missing_raw", "No raw play-by-play was cached when the audit ran."),
    "missing_normalized": ("missing", "The raw game validated, but no normalized model-input file existed when the audit ran."),
    "mismatch": ("stale", "When the audit ran, the saved normalized file did not match a fresh rebuild from the raw play-by-play and corrections."),
    "invalid_normalized": ("unverifiable", "The normalized file was not valid JSON when the audit ran."),
    "unreadable_normalized": ("unverifiable", "The normalized file could not be read when the audit ran."),
    "unreadable_raw": ("unverifiable", "The raw play-by-play file could not be read when the audit ran."),
}


@dataclass(frozen=True)
class GameInputCheck:
    """One game's inputs versus its audit lineage.

    category is fresh, stale, missing, missing_raw (waiting on raw data), or
    unverifiable; only fresh passes the gate. signatures identify the file
    states read, so callers can memoize results on them.
    """

    category: str
    reason: str | None
    signatures: tuple = ()


def _input_file_facts(path, *, normalized=False, cache=None):
    """Stat signature, SHA-256, and (for normalized games) canonical fingerprints.

    With a cache dict, results are reused while the file's (mtime_ns, size,
    inode) signature is unchanged; without one, current bytes are always read.
    """
    from possessions_loader import _json_fingerprint

    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"state": "missing", "signature": (str(path), None)}
    except OSError:
        return {"state": "unreadable", "signature": (str(path), None)}
    signature = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)
    if cache is not None and signature in cache:
        return cache[signature]
    try:
        data = path.read_bytes()
    except OSError:
        return {"state": "unreadable", "signature": signature}
    facts = {"state": "present", "signature": signature, "sha256": hashlib.sha256(data).hexdigest()}
    if normalized:
        try:
            saved = json.loads(data)
            build_possession_rows(saved)
            if (not isinstance(saved["applied_corrections"], list)
                    or not isinstance(saved["pbpstats_version"], str)
                    or not saved["pbpstats_version"]):
                raise ValueError("invalid corrections or parser version")
            facts.update(content=_json_fingerprint(saved),
                         corrections=_json_fingerprint(saved["applied_corrections"]),
                         game=_json_fingerprint(saved["game"]), game_id=saved["game"]["GameId"],
                         pbpstats_version=saved["pbpstats_version"])
            facts["components"] = {
                label: _json_fingerprint(saved[field])
                for label, field in (("possessions", "possessions"), ("game_metadata", "game"),
                                     ("parser_version", "pbpstats_version"),
                                     ("normalized_schema_version", "schema_version"))
            }
        except (ValueError, TypeError, KeyError, UnicodeError, AttributeError):
            facts["state"] = "invalid"
    if cache is not None:
        cache[signature] = facts
    return facts


def check_game_inputs(game_id, result, raw_root, normalized_root, *, cache=None):
    """Compare one game's current raw/normalized files with its verified audit lineage.

    Deterministic and read-only. Fresh requires: verified replay lineage; raw
    bytes equal to the lineage and attested hashes; normalized bytes equal to the
    lineage hash; canonical content and correction fingerprints equal to the
    recorded, replayed, and attested values (corrections fail closed); attested
    game metadata, an in-file GameId equal to game_id, and the attested parser
    version. Any other outcome names what is missing, stale, or unverifiable.
    """
    lineage = result.get("lineage") if isinstance(result, dict) else None
    if not isinstance(lineage, dict):
        return GameInputCheck("unverifiable", "The audit report has no input record for this game (older audit format).")
    if lineage.get("status") in LINEAGE_AT_AUDIT:
        return GameInputCheck(*LINEAGE_AT_AUDIT[lineage["status"]])
    verified = lineage.get("verified_inputs")
    if not (lineage.get("verified") is True and lineage.get("status") == "verified"
            and lineage.get("method") == LINEAGE_METHOD and isinstance(verified, dict)):
        reason = lineage.get("reason") or "the audit did not verify them"
        return GameInputCheck("unverifiable", f"The audit could not verify these inputs: {reason}.")
    raw = _input_file_facts(Path(raw_root) / f"live_{game_id}.json", cache=cache)
    saved = _input_file_facts(Path(normalized_root) / f"{game_id}.json", normalized=True, cache=cache)
    signatures = (raw["signature"], saved["signature"])
    for facts, label in ((raw, "raw play-by-play file"), (saved, "normalized model-input file")):
        if facts["state"] == "missing":
            return GameInputCheck("missing", f"The {label} has been removed since the audit.", signatures)
        if facts["state"] == "unreadable":
            return GameInputCheck("unverifiable", f"The {label} can no longer be read.", signatures)
    if saved["state"] == "invalid":
        return GameInputCheck("unverifiable", "The normalized file is no longer valid JSON with the expected fields.",
                              signatures)
    content_fields = (lineage.get("normalized_content_sha256"), lineage.get("replayed_normalized_sha256"),
                      verified.get("normalized_content_sha256"))
    correction_fields = (lineage.get("recorded_corrections_sha256"), lineage.get("replayed_corrections_sha256"),
                         verified.get("corrections_sha256"))
    content_matches = all(saved["content"] == value for value in content_fields)
    changed = []
    if not raw["sha256"] == lineage.get("raw_sha256") == verified.get("raw_sha256"):
        changed.append("the raw play-by-play file")
    if saved["sha256"] != lineage.get("normalized_sha256"):
        changed.append("the normalized file's formatting" if content_matches else "the normalized file")
    if not content_matches:
        changed.append("the normalized possession data")
    if not all(saved["corrections"] == value for value in correction_fields):
        changed.append("the recorded corrections")
    if saved["game"] != verified.get("game_metadata_sha256") or saved["game_id"] != game_id:
        changed.append("the game metadata")
    if saved["pbpstats_version"] != verified.get("pbpstats_version"):
        changed.append("the parser version record")
    for component, fingerprint in saved["components"].items():
        if not all(fingerprint == value for value in (
                lineage.get(f"recorded_{component}_sha256"),
                lineage.get(f"replayed_{component}_sha256"),
                verified.get(f"{component}_sha256"))):
            changed.append(f"the {component} fingerprint")
    if changed:
        return GameInputCheck("stale", "Changed since the audit: " + ", ".join(changed) + ".", signatures)
    return GameInputCheck("fresh", None, signatures)


@dataclass(frozen=True)
class PossessionRidgeFit:
    """Centered effects in points per possession; positive defense prevents points."""

    player_ids: np.ndarray
    intercept: float
    offense_effects: np.ndarray
    defense_effects: np.ndarray


def _center_effects(intercept, offense, defense):
    """Subtract block means and compensate for five players on each side."""
    offense_mean, defense_mean = offense.mean(), defense.mean()
    return (
        float(intercept + 5 * offense_mean - 5 * defense_mean),
        offense - offense_mean,
        defense - defense_mean,
    )


def _solve_compact_ridge(design, penalty, *, tolerance=1e-10, max_iterations=None):
    """Numerical kernel, also used for offline tests; callers use fit_possession_ridge.

    Minimize sum((y - intercept - sum(offense) + sum(defense))**2)
    + penalty * (sum(offense**2) + sum(defense**2)). Eliminate the
    unpenalized intercept by centering the operator, then use conjugate gradients.
    Memory is O(possessions + players), with no possession-by-player matrix.
    """
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("ridge penalty must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    y = np.asarray(design.points_per_possession, dtype=np.float64)
    count = len(design.player_ids)
    if y.ndim != 1 or not len(y) or not count or not np.all(np.isfinite(y)):
        raise ValueError("nonempty finite outcomes and player IDs required")
    for indices in (design.offense_indices, design.defense_indices):
        if (indices.shape != (len(y), 5) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= count)):
            raise ValueError("expected valid (possessions, 5) player indices")
    if max_iterations is None:
        max_iterations = 10 * (2 * count)
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    def forward(beta):
        return (beta[:count][design.offense_indices].sum(axis=1)
                - beta[count:][design.defense_indices].sum(axis=1))

    def transpose(values):
        weights = np.repeat(values, 5)
        return np.concatenate((
            np.bincount(design.offense_indices.ravel(), weights=weights, minlength=count),
            -np.bincount(design.defense_indices.ravel(), weights=weights, minlength=count),
        ))

    def normal(beta):
        predicted = forward(beta)
        return transpose(predicted - predicted.mean()) + penalty * beta

    rhs = transpose(y - y.mean())
    beta = np.zeros(2 * count)
    residual = rhs.copy()
    direction = residual.copy()
    squared_norm = float(residual @ residual)
    threshold = tolerance * max(1.0, np.linalg.norm(rhs))
    for _ in range(max_iterations):
        if np.sqrt(squared_norm) <= threshold:
            break
        product = normal(direction)
        curvature = float(direction @ product)
        if not np.isfinite(curvature) or curvature <= 0:
            raise RuntimeError("ridge solver encountered invalid curvature")
        step = squared_norm / curvature
        beta += step * direction
        residual -= step * product
        next_norm = float(residual @ residual)
        direction = residual + (next_norm / squared_norm) * direction
        squared_norm = next_norm
    if np.linalg.norm(rhs - normal(beta)) > threshold:
        raise RuntimeError("ridge solver did not converge")
    intercept = float(y.mean() - forward(beta).mean())
    intercept, offense, defense = _center_effects(intercept, beta[:count], beta[count:])
    return PossessionRidgeFit(design.player_ids.copy(), intercept, offense, defense)


def fit_possession_ridge(design=None, penalties=RIDGE_PENALTY_GRID, *,
                         status_path=STATUS_FILE, game_cache=None,
                         output_dir=BASE / "data", tolerance=1e-10,
                         max_iterations=None):
    """Gate, select by game holdout, refit all possessions, and write summaries.

    With no design, read the normalized local cache. Both status and design must
    cover 1,230 games. No output directories/files are touched until validation,
    selection, full-data fitting, and serialization have all succeeded.

    Outputs are possession_players.csv and possession_model.json, separate from
    legacy model files. Return the full-data PossessionRidgeFit. No downloads.
    """
    status = require_complete_season(status_path, game_cache=game_cache)
    if design is None:
        normalized_root = Path(game_cache) if game_cache is not None else Path(status_path).parent / "games"
        design = build_design_data(load_cached_possession_rows(normalized_root))
    selection = select_ridge_penalty(
        design, penalties, status_path=status_path, game_cache=game_cache,
        tolerance=tolerance, max_iterations=max_iterations,
    )
    fit = _solve_compact_ridge(
        design, selection.selected_penalty,
        tolerance=tolerance, max_iterations=max_iterations,
    )
    summary = summarize_player_impacts(fit, design)
    table = io.StringIO(newline="")
    fields = ("player_id", "offense_possessions", "defense_possessions",
              "offense_per_100", "defense_per_100", "net_per_100")
    writer = csv.writer(table)
    writer.writerow(fields)
    writer.writerows(zip(
        summary.player_ids.tolist(), summary.offense_possessions.tolist(),
        summary.defense_possessions.tolist(), summary.offense_per_100.tolist(),
        summary.defense_per_100.tolist(), summary.net_per_100.tolist(),
    ))
    metadata = {
        "model_type": "possession_ridge", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season": status.get("season"), "season_type": status.get("season_type"),
        "source": status.get("source"), "games_validated": status["games_validated"],
        "games_used": len(set(design.game_ids.tolist())),
        "eligible_possessions": len(design.game_ids), "players": len(design.player_ids),
        "selected_penalty": selection.selected_penalty,
        "candidate_penalties": list(selection.holdout_errors),
        "holdout_errors": selection.holdout_errors,
        "holdout_error_units": "squared points per possession",
        "holdout_metric": "mean squared error, equally weighted possessions",
        "holdout_split": "20% of unique game IDs ranked by SHA-256(rotation-index-holdout-v1:<game ID>)",
        "training_game_ids": list(selection.training_game_ids),
        "holdout_game_ids": list(selection.holdout_game_ids),
        "final_fit": "all eligible possessions; selected SSE penalty unchanged",
        "objective": "sum squared residuals + penalty * (sum offense squared + sum defense squared)",
        "intercept_points_per_possession": fit.intercept,
        "intercept_penalized": False,
        "centering": "separate unweighted player means of zero; intercept adjusted to preserve predictions",
        "effect_units": "points per 100 possessions",
        "defense_sign": "positive means points prevented",
        "net_definition": "offense_per_100 + defense_per_100",
        "limitations": ["Single-season conditional associations, not causal effects.",
                        "Only model-eligible possessions included; exclusions can select the sample.",
                        "Holdout errors tune the penalty and are not an independent final evaluation."],
    }
    payloads = {"possession_players.csv": table.getvalue(),
                "possession_model.json": json.dumps(metadata, indent=2, allow_nan=False) + "\n"}
    # Recheck immediately before publishing so changes during fitting cannot
    # silently overwrite valid outputs using an earlier audit snapshot.
    if require_complete_season(status_path, game_cache=game_cache) != status:
        raise IncompleteSeasonError("cache audit changed during fitting; estimates were not written")
    # Stage both complete payloads before replacing either destination.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name, content in payloads.items():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=output_dir, delete=False) as temp:
                staged.append((Path(temp.name), output_dir / name))
                temp.write(content)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return fit


@dataclass(frozen=True)
class PlayerImpactSummary:
    """Per-player impacts per 100 possessions, aligned with player_ids.

    Positive defense means points prevented, so net is offense + defense.
    Possession counts are the player's own offensive and defensive possessions.
    """

    player_ids: np.ndarray
    offense_possessions: np.ndarray
    defense_possessions: np.ndarray
    offense_per_100: np.ndarray
    defense_per_100: np.ndarray
    net_per_100: np.ndarray


def summarize_player_impacts(fit, design):
    """Scale fitted per-possession effects to per 100 and attach possession counts.

    Pure in-memory transform: no fitting, gating, file output, or publication.
    The fit and design must share the same player vocabulary in the same order.
    """
    count = len(design.player_ids)
    if not np.array_equal(fit.player_ids, design.player_ids):
        raise ValueError("fit and design player IDs differ")
    if fit.offense_effects.shape != (count,) or fit.defense_effects.shape != (count,):
        raise ValueError("fitted effects must have one value per player")
    if not (np.all(np.isfinite(fit.offense_effects)) and np.all(np.isfinite(fit.defense_effects))):
        raise ValueError("fitted effects must be finite")
    for indices in (design.offense_indices, design.defense_indices):
        if (indices.ndim != 2 or indices.shape[1] != 5 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= count)):
            raise ValueError("expected valid (possessions, 5) player indices")
    offense = 100.0 * fit.offense_effects
    defense = 100.0 * fit.defense_effects
    return PlayerImpactSummary(
        player_ids=design.player_ids.copy(),
        offense_possessions=np.bincount(design.offense_indices.ravel(), minlength=count),
        defense_possessions=np.bincount(design.defense_indices.ravel(), minlength=count),
        offense_per_100=offense,
        defense_per_100=defense,
        net_per_100=offense + defense,
    )


@dataclass(frozen=True)
class RidgePenaltySelection:
    """Holdout MSE is in squared points per possession, not per 100 possessions."""

    selected_penalty: float
    holdout_errors: dict[float, float]
    training_game_ids: tuple[str, ...]
    holdout_game_ids: tuple[str, ...]


def _game_holdout(game_ids):
    """Fixed 20% game split, invariant to row order and possession counts.

    Rank unique IDs by SHA-256 of 'rotation-index-holdout-v1:<game ID>'.
    Hold out ceil(0.2 * games), i.e. 246 of a complete 1,230-game season.
    """
    game_ids = np.asarray(game_ids)
    if game_ids.ndim != 1 or not len(game_ids) or any(
        not isinstance(g, str) or not g for g in game_ids
    ):
        raise ValueError("nonempty one-dimensional string game IDs required")
    unique = sorted(set(game_ids.tolist()))
    if len(unique) < 2:
        raise ValueError("game holdout requires at least two distinct games")
    ranked = sorted(unique, key=lambda g: (
        hashlib.sha256(f"rotation-index-holdout-v1:{g}".encode()).digest(), g,
    ))
    holdout = tuple(sorted(ranked[:(len(unique) + 4) // 5]))
    training = tuple(sorted(set(unique) - set(holdout)))
    return np.isin(game_ids, holdout), training, holdout


def _select_ridge_penalty(design, penalties, *, tolerance, max_iterations):
    """Numerical selection kernel; public callers must pass the season gate."""
    penalties = tuple(sorted(set(float(p) for p in penalties)))
    if not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError("candidate penalties must be nonempty, finite, and positive")
    holdout_mask, training_ids, holdout_ids = _game_holdout(design.game_ids)
    y = np.asarray(design.points_per_possession)
    if y.shape != holdout_mask.shape or not np.all(np.isfinite(y)):
        raise ValueError("finite outcomes must match game IDs")
    training = PossessionDesignData(
        player_ids=design.player_ids,
        offense_indices=design.offense_indices[~holdout_mask],
        defense_indices=design.defense_indices[~holdout_mask],
        points_per_possession=y[~holdout_mask],
        game_ids=design.game_ids[~holdout_mask],
    )
    # Retain the full ID vocabulary: players appearing only in held-out games
    # have zero fitted coefficients under ridge, never holdout-derived effects.
    errors = {}
    for penalty in penalties:
        fit = _solve_compact_ridge(
            training, penalty, tolerance=tolerance, max_iterations=max_iterations,
        )
        predictions = (
            fit.intercept
            + fit.offense_effects[design.offense_indices[holdout_mask]].sum(axis=1)
            - fit.defense_effects[design.defense_indices[holdout_mask]].sum(axis=1)
        )
        error = float(np.mean((y[holdout_mask] - predictions) ** 2))
        if not np.isfinite(error):
            raise RuntimeError("nonfinite ridge holdout error")
        errors[penalty] = error
    # Deterministic tie rule: choose the smaller penalty for exactly equal MSE.
    selected = min(errors, key=lambda p: (errors[p], p))
    return RidgePenaltySelection(selected, errors, training_ids, holdout_ids)


def select_ridge_penalty(design, penalties=RIDGE_PENALTY_GRID, *,
                         status_path=STATUS_FILE, game_cache=None, tolerance=1e-10,
                         max_iterations=None):
    """Gate season data, tune on a deterministic 80/20 game split, return report.

    Default grid: 1, 10, 100, 1000, 10000 for the solver's SSE objective.
    Score each candidate by possession-weighted MSE over held-out games (each
    possession has equal weight). These are tuning errors, not an independent
    final evaluation. The selected penalty is not rescaled or refit here.
    No files or player estimates are published.
    """
    status = require_complete_season(status_path, game_cache=game_cache)
    if set(design.game_ids.tolist()) != set(status["validated_game_ids"]):
        raise IncompleteSeasonError("penalty selection requires design data from all 1,230 audited games")
    return _select_ridge_penalty(
        design, penalties, tolerance=tolerance, max_iterations=max_iterations,
    )

