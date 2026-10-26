"""Fail-closed, resumable 2025-26 possession cache. See POSSESSION_DATA.md."""

import json
import hashlib
import sys
import shutil
import stat
import time
from threading import RLock
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import requests
import season_config as season
from pbpstats.client import Client
from pbpstats.resources.enhanced_pbp import (
    FieldGoal,
    FreeThrow,
    Foul,
    InvalidNumberOfStartersException,
    Rebound,
    Substitution,
    Turnover,
)
from pbpstats.resources.enhanced_pbp.live.start_of_period import LiveStartOfPeriod

BASE = Path(__file__).resolve().parent
ROOT = BASE / "data/cache/possessions"
RAW = ROOT / "pbpstats"
NORMALIZED = ROOT / "games"
SEASON = season.SEASON
LINEAGE_SCHEMA_VERSION = 1
NORMALIZED_SCHEMA_VERSION = 1
LINEAGE_COMPONENTS = (("possessions", "possessions"), ("game_metadata", "game"),
                      ("parser_version", "pbpstats_version"), ("normalized_schema_version", "schema_version"))
LIVE_PBP_URL = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/playbyplay/playbyplay_{}.json"
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 1.0  # waits 1, 2, 4, 8 s between attempts: at most 15 s per file
_STARTERS_LOCK = RLock()

# Individually verified NBA live-feed corrections. Each entry is guarded by the
# game/action identity and the exact feed value it replaces; never infer a sub.
VERIFIED_CORRECTIONS = (
    {
        "game_id": "0022400061",
        "action_number": 509,
        "period": 4,
        "expected_clock": "PT04M20.00S",
        "team_id": 1610612738,
        "expected_player1_id": 201143,  # Al Horford, incorrect feed value
        "corrected_player1_id": 1628436,  # Luke Kornet
        "corrected_player_name": "Luke Kornet",
        "reason": (
            "The official NBA game book records Kornet entering for Horford at "
            "8:35, then records Queta playing 4:20 and Kornet 4:15 in Q4. "
            "At 4:20, Queta replaced Kornet."
        ),
        "evidence_url": "https://statsdmz.nba.com/pdfs/20241022/20241022_NYKBOS_book.pdf",
    },
)

_LIVE_BOXSCORE = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/boxscore/boxscore_{}.json"

# Official OT1 box-score evidence (fixtures/official_evidence_2025-26.json: the player's
# OT1 MIN is 5:00) for a starter no period-5 action names. Each entry completes only
# that exact game/period/team, and only when pbpstats found exactly `parser_four`.
VERIFIED_PERIOD_STARTERS = {
    ("0022500458", 5, 1610612763): {"player": 1630583,  # Santi Aldama, MEM
                                    "parser_four": (1628991, 1629630, 1642377, 1642907)},
    ("0022500458", 5, 1610612755): {"player": 1629656,  # Quentin Grimes, PHI
                                    "parser_four": (202331, 203954, 1630178, 1642845)},
    ("0022500637", 5, 1610612738): {"player": 1630573,  # Sam Hauser, BOS
                                    "parser_four": (1627759, 1629014, 1630202, 1630568)},
    ("0022501029", 5, 1610612742): {"player": 1629023,  # P.J. Washington, DAL
                                    "parser_four": (1629655, 1630230, 1642843, 1642948)},
}

# Official home team where the pinned manifest's sides are reversed (evidence:
# fixtures/official_evidence_2025-26.json). Guarded by the exact manifest teams it
# corrects; the manifest itself stays untouched.
VERIFIED_HOME_TEAMS = {
    "0022500147": {
        "manifest_home_team_id": "1610612742",  # DAL
        "home_team_id": "1610612765",  # DET: homeTeam, gameCode 20251101/DALDET, Arena CDMX
        "evidence_url": _LIVE_BOXSCORE.format("0022500147"),
    },
}


def save(path, data, *, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2 if pretty else None, sort_keys=pretty,
                              separators=None if pretty else (",", ":")))
    tmp.replace(path)


def lineup(event, teams):
    players = {str(k): sorted(v) for k, v in event.current_players.items()}
    if set(players) != teams:
        raise ValueError(f"event {event.event_num}: wrong/missing teams")
    flat = []
    for ids in players.values():
        if len(ids) != 5 or len(set(ids)) != 5 or any(i <= 0 for i in ids):
            raise ValueError(f"event {event.event_num}: invalid five-player lineup")
        flat.extend(ids)
    if len(set(flat)) != 10:
        raise ValueError(f"event {event.event_num}: player on both teams")
    return players


def apply_verified_corrections(game_id, events):
    """Apply only explicit, feed-version-guarded corrections before lineup access."""
    applied = []
    for correction in VERIFIED_CORRECTIONS:
        if correction["game_id"] != game_id:
            continue
        matches = [e for e in events if e.event_num == correction["action_number"]]
        if len(matches) != 1:
            raise ValueError(
                f"correction target {game_id}/{correction['action_number']} "
                f"matched {len(matches)} events"
            )
        event = matches[0]
        if not (
            isinstance(event, Substitution)
            and event.period == correction["period"]
            and event.clock == correction["expected_clock"]
            and event.team_id == correction["team_id"]
            and event.action_type == "substitution"
            and event.sub_type == "out"
            and event.player1_id == correction["expected_player1_id"]
        ):
            raise ValueError(
                f"correction guard failed for {game_id}/{event.event_num}: "
                "expected the verified Q4 substitution-out event and feed player ID"
            )
        original_player_id = event.player1_id
        event.player1_id = correction["corrected_player1_id"]
        applied.append(
            {
                "game_id": game_id,
                "action_number": event.event_num,
                "period": event.period,
                "clock": event.clock,
                "display_clock": "4:20",
                "team_id": event.team_id,
                "action_type": event.action_type,
                "sub_type": event.sub_type,
                "expected_feed_player1_id": original_player_id,
                "corrected_player1_id": event.player1_id,
                "corrected_player_name": correction["corrected_player_name"],
                "reason": correction["reason"],
                "evidence_url": correction["evidence_url"],
            }
        )
    return applied


def _opening_tip_players(raw_events):
    """{period: {player: team}} for players named in each period's opening jump ball.

    pbpstats' live parser drops jumpBallWon/LostPersonId, so a tip participant with
    no other event that period is missing from its starters. A participant counts
    only if every non-jump-ball event names one team for them and they were not
    substituted in between the period start and the tip.
    """
    teams = {}
    for event in raw_events:
        if event.get("actionType") != "jumpball" and event.get("personId") and event.get("teamId"):
            teams.setdefault(event["personId"], set()).add(event["teamId"])
    tips, subbed_in = {}, set()
    for event in raw_events:
        period = event.get("period")
        if event.get("actionType") == "period" and event.get("subType") == "start":
            subbed_in = set()
        elif event.get("actionType") == "substitution" and event.get("subType") == "in":
            subbed_in.add(event.get("personId"))
        elif (event.get("actionType") == "jumpball" and event.get("descriptor") == "startperiod"
              and period not in tips):
            tips[period] = {
                pid: next(iter(teams[pid]))
                for pid in (event.get("jumpBallWonPersonId"), event.get("jumpBallLostPersonId"))
                if pid and len(teams.get(pid, ())) == 1 and pid not in subbed_in
            }
    return tips


def _starters_with_opening_tip(tips, game_id, verified=None):
    """pbpstats period starters, completed only by explicit opening-tip evidence or a
    verified {(period, team): {"player", "parser_four"}} entry for this game."""
    original = LiveStartOfPeriod.get_period_starters

    def get_period_starters(self, file_directory=None, ignore_missing_starters=False):
        try:
            return original(self, file_directory, ignore_missing_starters)
        except InvalidNumberOfStartersException:
            if self.game_id != game_id:
                raise
            starters = original(self, file_directory, ignore_missing_starters=True)
            for pid, team in tips.get(self.period, {}).items():
                if team in starters and pid not in starters[team]:
                    starters[team] = [*starters[team], pid]
            for (period, team), entry in (verified or {}).items():
                if period == self.period and sorted(starters.get(team, ())) == sorted(entry["parser_four"]):
                    starters[team] = [*starters[team], entry["player"]]
            players = [pid for team in starters.values() for pid in team]
            if (len(starters) != 2 or any(len(set(p)) != 5 or len(p) != 5 for p in starters.values())
                    or len(set(players)) != 10 or any(type(pid) is not int or pid <= 0 for pid in players)):
                raise
            return starters

    return get_period_starters


def normalize(game_info):
    season.require_metadata(game_info)
    gid = game_info["GameId"]
    source = json.loads((RAW / "pbp" / f"live_{gid}.json").read_text())["game"]
    if source["gameId"] != gid:
        raise ValueError("wrong game id or season in raw feed")
    client = Client({"dir": str(RAW), "Possessions": {"source": "file", "data_provider": "live"}})
    # The parser hook is global; serialize its use and bind its evidence to this game.
    evidence = _opening_tip_players(source["actions"])
    rostered = {(e.get("personId"), e.get("teamId")) for e in source["actions"]}
    verified = {(period, team): entry for (game_id, period, team), entry in VERIFIED_PERIOD_STARTERS.items()
                if game_id == gid and (entry["player"], team) in rostered}  # the feed must place them on this team
    with _STARTERS_LOCK:
        original_starters = LiveStartOfPeriod.get_period_starters
        LiveStartOfPeriod.get_period_starters = _starters_with_opening_tip(evidence, gid, verified)
        try:
            game = client.Game(gid)
        finally:
            LiveStartOfPeriod.get_period_starters = original_starters
    teams = {str(game_info[f"{side}TeamId"]) for side in ("Home", "Away")}
    expected = {str(game_info[f"{side}TeamId"]): game_info[f"{side}Points"] for side in ("Home", "Away")}
    raw_events = source["actions"]
    if source["gameId"] != gid or not raw_events:
        raise ValueError("missing events or wrong game id")
    if {e["period"] for e in raw_events} != set(range(1, max(e["period"] for e in raw_events) + 1)):
        raise ValueError("missing period")
    if max(e["period"] for e in raw_events) < 4:
        raise ValueError("incomplete game")
    final = {}
    for side in ("Home", "Away"):
        scores = [e[f"score{side}"] for e in raw_events if str(e.get(f"score{side}", "")).strip()]
        final[side] = int(scores[-1]) if scores else None
    home = VERIFIED_HOME_TEAMS.get(gid)
    if home:
        if (home["manifest_home_team_id"], home["home_team_id"]) != (game_info["HomeTeamId"], game_info["AwayTeamId"]):
            raise ValueError("verified home-team correction does not match the game list")
        if final != {"Home": game_info["AwayPoints"], "Away": game_info["HomePoints"]}:
            raise ValueError("final play-by-play score disagrees with the verified home team")
    elif final == {"Home": game_info["AwayPoints"], "Away": game_info["HomePoints"]}:
        # Same team totals, opposite sides: contradictory sources, never swapped here.
        raise ValueError(
            f"home/away disagrees with game list: feed home score {final['Home']} is manifest away "
            f"{game_info['AwayTeamAbbreviation']}, feed away score {final['Away']} is manifest home "
            f"{game_info['HomeTeamAbbreviation']}; official home-team evidence required")
    else:
        for side in ("Home", "Away"):
            if final[side] != game_info[f"{side}Points"]:
                raise ValueError(f"{side}: final play-by-play score disagrees with game list")
    rows, score, counts, seen = [], Counter(), Counter(), []
    possessions = game.possessions.items
    all_events = [event for possession in possessions for event in possession.events]
    applied_corrections = apply_verified_corrections(gid, all_events)
    for p in possessions:
        offense = str(p.offense_team_id)
        if offense not in teams:
            raise ValueError(f"invalid possession offense {offense}")
        defense = next(iter(teams - {offense}))
        events, configurations, reasons = [], set(), set()
        points, other_points, orebs = 0, Counter(), 0
        for e in p.events:
            seen.append(e.event_num)
            relevant = isinstance(e, (FieldGoal, FreeThrow, Foul, Rebound, Turnover))
            # Live feeds list individual sub-outs before sub-ins: intermediate
            # administrative events can correctly have fewer than five players.
            on_court = lineup(e, teams) if relevant else {str(k): sorted(v) for k, v in e.current_players.items()}
            record = {"event_id": e.event_num, "period": e.period, "clock": e.clock,
                      "type": type(e).__name__, "description": e.description,
                      "lineups": on_court}
            record["lineup_validated_for_play"] = relevant
            attributed = on_court
            if isinstance(e, FreeThrow):
                record["technical"] = bool(e.is_technical_ft)
                if not e.is_technical_ft:
                    foul = e.foul_that_led_to_ft
                    if foul is None or foul.period != e.period:
                        reasons.add("unmatched_free_throw_foul")
                    else:
                        attributed = lineup(foul, teams)
                        record["attribution_event_id"] = foul.event_num
                        record["attribution_lineups"] = attributed
                else:
                    # Keep these for score reconciliation, outside ordinary possession response.
                    reasons.add("technical_free_throw")
            if relevant:
                configurations.add(json.dumps(attributed, sort_keys=True))
            if isinstance(e, Rebound) and e.is_real_rebound and e.oreb:
                orebs += 1
            if isinstance(e, (FieldGoal, FreeThrow)):
                value = e.shot_value if e.is_made else 0
                scoring_team = str(e.team_id)
                if scoring_team not in teams:
                    raise ValueError("unknown scoring team")
                record.update(scoring_team_id=scoring_team, points=value)
                score[scoring_team] += value
                if scoring_team == offense and not (isinstance(e, FreeThrow) and e.is_technical_ft):
                    points += value
                else:
                    other_points[scoring_team] += value
                    if value and scoring_team != offense:
                        reasons.add("opponent_scores_during_possession")
            events.append(record)
        counted = bool(p.events[-1].count_as_possession)
        if counted:
            counts[offense] += 1
        else:
            reasons.add("not_counted_by_pbpstats")
        if len(configurations) != 1:
            reasons.add("changing_or_missing_lineup")
        if any(e.period != p.period for e in p.events):
            raise ValueError("possession crosses periods")
        stable = json.loads(next(iter(configurations))) if len(configurations) == 1 else None
        rows.append({"game_id": gid, "period": p.period, "number": p.number,
                     "start_time": p.start_time, "end_time": p.end_time,
                     "offense_team_id": offense, "defense_team_id": defense,
                     "counted_possession": counted, "points": points,
                     "exceptional_points_by_team": dict(other_points),
                     "offensive_rebounds": orebs,
                     "offense_players": stable[offense] if stable else None,
                     "defense_players": stable[defense] if stable else None,
                     "model_eligible": not reasons, "flags": sorted(reasons), "events": events})
    if seen != [e["actionNumber"] for e in raw_events] or len(seen) != len(set(seen)):
        raise ValueError("events missing, duplicated, reordered, or unassigned")
    if any(score[t] != expected[t] for t in teams):
        raise ValueError(f"scoring events {dict(score)} != final scores {expected}")
    if any(counts[t] == 0 for t in teams):
        raise ValueError("missing team possessions")
    # API possession totals can differ by provider; report exact differences, never hide them.
    differences = {str(game_info[f"{s}TeamId"]): counts[str(game_info[f"{s}TeamId"])] - game_info[f"{s}Possessions"] for s in ("Home", "Away")}
    return {"schema_version": NORMALIZED_SCHEMA_VERSION, "pbpstats_version": version("pbpstats"),
            "game": game_info, "score_by_team": dict(score),
            "counted_possessions_by_team": dict(counts),
            "possession_count_difference_vs_api": differences,
            "applied_corrections": applied_corrections, "possessions": rows}


def listed_game_ids(games):
    """Require exactly the configured authoritative regular-season manifest IDs."""
    return season.require_manifest(games)


class DownloadError(Exception):
    """A download failure whose message is a stable reason code (no timestamps or paths)."""


def fetch_json(url, **kwargs):
    """GET JSON, retrying only transient failures with bounded exponential backoff."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=30, **kwargs)
        except requests.Timeout:
            reason = "timeout"
        except requests.ConnectionError:
            reason = "connection_error"
        else:
            if response.status_code in RETRY_STATUS_CODES:
                reason = f"http_{response.status_code}"
            elif response.status_code != 200:
                raise DownloadError(f"http_{response.status_code}")
            else:
                try:
                    return response.json()
                except ValueError:
                    raise DownloadError("invalid_json") from None
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise DownloadError(f"{reason}_after_{MAX_ATTEMPTS}_attempts")


def live_pbp_problem(game_id, data):
    """Reason a downloaded feed is not a whole game's play-by-play, else None.

    A structural gate before saving only; normalize() still runs every check
    when --rebuild-all/--offline later read the file.
    """
    if game_id not in season.GAMES:
        return "wrong_season_or_unlisted_game"
    game = data.get("game") if isinstance(data, dict) else None
    if not isinstance(game, dict) or game.get("gameId") != game_id:
        return "wrong_or_missing_game_id"
    actions = game.get("actions")
    if not isinstance(actions, list) or not actions:
        return "missing_actions"
    periods = [a.get("period") if isinstance(a, dict) else None for a in actions]
    if (any(type(p) is not int or p < 1 for p in periods) or max(periods) < 4
            or len(set(periods)) != max(periods)):
        return "incomplete_game"
    last = actions[-1]
    if (last.get("actionType") != "game" or last.get("subType") != "end"
            or last.get("clock") != "PT00M00.00S" or last["period"] != max(periods)):
        return "incomplete_game"
    return None


def storage_preflight():
    """Read-only estimate: largest cached files per type, with 2x safety headroom."""
    ids = sorted(season.GAMES)

    def sizes(paths):
        result = []
        for path in paths:
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"not a regular cached file: {path}")
            result.append(info.st_size)
        return result

    raw = sizes(RAW / "pbp" / f"live_{game_id}.json" for game_id in ids)
    normalized = sizes(NORMALIZED / f"{game_id}.json" for game_id in ids)
    missing_raw, missing_normalized = len(ids) - len(raw), len(ids) - len(normalized)
    # Any season's cached files are valid size samples (never model inputs), so a new season can start.
    raw = sizes(sorted((RAW / "pbp").glob("live_*.json")))
    normalized = sizes(sorted(NORMALIZED.glob("*.json")))
    raw_max, normalized_max = max(raw, default=0), max(normalized, default=0)
    known = not ((missing_raw and not raw_max) or (missing_normalized and not normalized_max))
    estimate = missing_raw * raw_max + missing_normalized * normalized_max if known else None
    required = 2 * estimate if known else None
    available = shutil.disk_usage(ROOT).free
    return {"available_bytes": available, "missing_games": missing_raw,
            "missing_normalized_games": missing_normalized,
            "raw_sample_games": sum(size > 0 for size in raw),
            "normalized_sample_games": sum(size > 0 for size in normalized),
            "raw_max_bytes": raw_max, "normalized_max_bytes": normalized_max,
            "estimated_required_bytes": estimate, "safety_required_bytes": required,
            "passes_safety_margin": known and available >= required,
            "reason": None if known else "Missing non-empty cached samples for a required file type."}


def download_missing_games():
    """Download every listed game's missing raw play-by-play; nothing else.

    Stops before any game request unless storage_preflight() passes.
    Cached raw files are skipped, never re-downloaded or replaced. A feed is saved
    (atomically) only after it parses and passes live_pbp_problem(). One game's
    failure is recorded and the rest continue. Never normalizes, audits, or runs
    the model: status.json and normalized files are not touched. Writes only raw
    files and download_report.json. The game list is the pinned season manifest.
    """
    ids = sorted(season.GAMES)
    storage = storage_preflight()
    if not storage["passes_safety_margin"]:  # fail closed: no game is requested
        if storage["safety_required_bytes"] is None:
            raise DownloadError("storage_unknown: " + storage["reason"])
        raise DownloadError(f"insufficient_storage: {storage['safety_required_bytes']:,} bytes required "
                            f"with headroom, {storage['available_bytes']:,} available")
    report = {"season": SEASON, "games_listed": len(ids), "already_cached": 0,
              "downloaded_game_ids": [], "failed_game_ids": [], "failures": {}}
    for game_id in ids:
        target = RAW / "pbp" / f"live_{game_id}.json"
        if target.exists():
            try:
                problem = live_pbp_problem(game_id, json.loads(target.read_text()))
            except (OSError, ValueError, UnicodeError):
                problem = "invalid_cached_raw"
            if problem:
                report["failed_game_ids"].append(game_id)
                report["failures"][game_id] = f"cached_{problem}"
            else:
                report["already_cached"] += 1
            continue
        try:
            data = fetch_json(LIVE_PBP_URL.format(game_id))
            problem = live_pbp_problem(game_id, data)
            if problem:
                raise DownloadError(problem)
            save(target, data)
        except DownloadError as error:
            reason = str(error)
        except requests.RequestException as error:
            reason = f"request_error_{type(error).__name__}"
        except OSError as error:
            reason = f"write_error_{type(error).__name__}"
        else:
            report["downloaded_game_ids"].append(game_id)
            time.sleep(0.15)
            continue
        report["failed_game_ids"].append(game_id)
        report["failures"][game_id] = reason
    save(ROOT / "download_report.json", report, pretty=True)
    return report


def _json_fingerprint(value):
    """SHA-256 of canonical JSON; whitespace and dictionary order do not matter."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _audit_lineage(raw_path, normalized_path, *, result=None, raw_before=None):
    """Verify current input/output consistency by replay, never claim history.

    Byte hashes identify files actually observed. Only a successful normalization
    whose canonical output equals the saved normalized file gets verified_inputs.
    Unreadable/missing files and failed replays never receive inferred lineage.
    """
    lineage = {"schema_version": LINEAGE_SCHEMA_VERSION,
               "method": "canonical_normalization_replay_v1", "verified": False,
               "raw_sha256": None, "normalized_sha256": None}
    components = LINEAGE_COMPONENTS
    contents = {}
    for label, path in (("raw", raw_path), ("normalized", normalized_path)):
        try:
            contents[label] = path.read_bytes()
            lineage[f"{label}_sha256"] = hashlib.sha256(contents[label]).hexdigest()
        except FileNotFoundError:
            lineage[f"{label}_file_status"] = "missing"
        except OSError as error:
            lineage[f"{label}_file_status"] = "unreadable"
            lineage[f"{label}_error"] = type(error).__name__
    normalized_error = None
    if "normalized" in contents:
        try:
            saved = json.loads(contents["normalized"])
            lineage["normalized_content_sha256"] = _json_fingerprint(saved)
            if isinstance(saved, dict):
                if isinstance(saved.get("applied_corrections"), list):
                    lineage["recorded_corrections_sha256"] = _json_fingerprint(saved["applied_corrections"])
                for label, field in components:
                    if field in saved:
                        lineage[f"recorded_{label}_sha256"] = _json_fingerprint(saved[field])
        except (ValueError, TypeError, UnicodeError) as error:
            normalized_error = type(error).__name__
    if "raw" not in contents:
        lineage["status"] = "missing_raw" if lineage["raw_file_status"] == "missing" else "unreadable_raw"
        return lineage
    if result is None or raw_before is None:
        lineage["status"] = "unverifiable"
        lineage["reason"] = "raw normalization did not pass validation"
        return lineage
    if raw_before != lineage["raw_sha256"]:
        lineage["status"] = "unverifiable"
        lineage["reason"] = "raw file changed during normalization"
        return lineage
    try:
        season.require_metadata(result["game"])
        raw_game = json.loads(contents["raw"])["game"]
        if raw_game.get("gameId") != result["game"]["GameId"]:
            raise ValueError("raw/normalized game identity mismatch")
        for row in result["possessions"]:
            if row.get("game_id") != result["game"]["GameId"]:
                raise ValueError("possession belongs to a different game")
        for _, field in components:
            result[field]
        if (result['schema_version'] != NORMALIZED_SCHEMA_VERSION
                or not isinstance(result['possessions'], list)
                or not isinstance(result['game'], dict)
                or not result['game'].get('GameId')
                or not isinstance(result['pbpstats_version'], str) or not result['pbpstats_version']
                or not isinstance(result['applied_corrections'], list)):
            raise ValueError('missing or invalid replay components')
        for correction in result['applied_corrections']:
            if (correction['game_id'] != result['game']['GameId']
                    or any(type(correction[key]) is not int or correction[key] <= 0
                           for key in ('action_number', 'expected_feed_player1_id', 'corrected_player1_id'))):
                raise ValueError('invalid correction identity')
    except (KeyError, TypeError, ValueError):
        lineage.update(status='unverifiable', reason='replay lacks required components or correction identity')
        return lineage
    lineage["replayed_normalized_sha256"] = _json_fingerprint(result)
    lineage["replayed_corrections_sha256"] = _json_fingerprint(result["applied_corrections"])
    # Expose component changes even when the saved file cannot be verified.
    for label, field in components:
        if field in result:
            lineage[f"replayed_{label}_sha256"] = _json_fingerprint(result[field])
    if "normalized" not in contents:
        lineage["status"] = ("missing_normalized" if lineage["normalized_file_status"] == "missing"
                             else "unreadable_normalized")
        return lineage
    if normalized_error is not None:
        lineage["status"] = "invalid_normalized"
        lineage["reason"] = normalized_error
        return lineage
    if lineage["normalized_content_sha256"] != lineage["replayed_normalized_sha256"]:
        lineage["status"] = "mismatch"
        lineage["reason"] = "saved normalized content differs from current raw/correction replay"
        return lineage
    lineage["verified"] = True
    lineage["status"] = "verified"
    lineage["verified_inputs"] = {
        "raw_sha256": lineage["raw_sha256"],
        "normalized_content_sha256": lineage["normalized_content_sha256"],
        "corrections_sha256": lineage["replayed_corrections_sha256"],
        "game_metadata_sha256": _json_fingerprint(result["game"]),
        "pbpstats_version": result["pbpstats_version"],
    }
    for label, field in components:
        if field in result:
            lineage["verified_inputs"][f"{label}_sha256"] = lineage[f"replayed_{label}_sha256"]
    return lineage


def audit_cached_season(*, write_status=True):
    """Audit every listed game locally; optionally write the deterministic status report.

    Missing and invalid game files are recorded independently; neither prevents
    auditing later cached games. normalize() uses pbpstats' file-only provider.
    No raw or normalized game files are created, changed, or deleted here.
    Set write_status=False for a fully read-only audit.
    """
    status = {
        "season": SEASON, "season_type": "Regular Season", "mode": "offline_audit",
        "source": "cached NBA live play-by-play via pbpstats; pinned regular-season manifest",
        "complete": False, "audit_complete": False, "games_expected": len(season.GAMES),
        "games_cached": 0, "games_validated": 0, "games_failed": 0, "games_missing": 0,
        "expected_game_ids": [], "cached_game_ids": [], "validated_game_ids": [],
        "failed_game_ids": [], "missing_game_ids": [], "validation_errors": {},
        "possession_segments": 0, "counted_possessions": 0,
        "model_eligible_possessions": 0, "flags": {},
        "possession_count_differences": {}, "game_results": {}, "lineage_counts": {},
    }
    try:
        games = list(season.GAMES.values())
        status["expected_game_ids"] = listed_game_ids(games)
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        if write_status:
            save(ROOT / "status.json", status, pretty=True)
        return status

    flag_counts = Counter()
    for info in sorted(games, key=lambda g: g["GameId"]):
        gid = info["GameId"]
        target = RAW / "pbp" / f"live_{gid}.json"
        normalized_path = NORMALIZED / f"{gid}.json"
        if not target.exists():
            status["missing_game_ids"].append(gid)
            status["game_results"][gid] = {
                "status": "missing", "lineage": _audit_lineage(target, normalized_path),
            }
            continue
        status["cached_game_ids"].append(gid)
        try:
            raw_before = hashlib.sha256(target.read_bytes()).hexdigest()
            result = normalize(info)
        except Exception as error:
            # Keep errors stable across cache locations (no timestamps or temp paths).
            message = f"{type(error).__name__}: {error}".replace(str(RAW), "<raw_cache>")
            status["failed_game_ids"].append(gid)
            status["validation_errors"][gid] = message
            status["game_results"][gid] = {
                "status": "failed", "error": message,
                "lineage": _audit_lineage(target, normalized_path),
            }
            continue
        rows = result["possessions"]
        flags = Counter(f for row in rows for f in row["flags"])
        segments = len(rows)
        counted = sum(row["counted_possession"] for row in rows)
        eligible = sum(row["model_eligible"] for row in rows)
        status["validated_game_ids"].append(gid)
        status["possession_segments"] += segments
        status["counted_possessions"] += counted
        status["model_eligible_possessions"] += eligible
        flag_counts.update(flags)
        differences = result["possession_count_difference_vs_api"]
        if any(differences.values()):
            status["possession_count_differences"][gid] = differences
        status["game_results"][gid] = {
            "status": "validated", "possession_segments": segments,
            "counted_possessions": counted, "model_eligible_possessions": eligible,
            "counted_possessions_by_team": result["counted_possessions_by_team"],
            "score_by_team": result["score_by_team"], "flags": dict(sorted(flags.items())),
            "possession_count_difference_vs_api": differences,
            "applied_corrections": result["applied_corrections"],
            "lineage": _audit_lineage(target, normalized_path, result=result, raw_before=raw_before),
        }
    for category in ("cached", "validated", "failed", "missing"):
        status[f"games_{category}"] = len(status[f"{category}_game_ids"])
    status["flags"] = dict(sorted(flag_counts.items()))
    status["lineage_counts"] = dict(sorted(Counter(
        r["lineage"]["status"] for r in status["game_results"].values()
    ).items()))
    status["audit_complete"] = True
    status["complete"] = status["games_validated"] == status["games_expected"]
    if not status["complete"]:
        status["error"] = (
            f"Offline audit incomplete season: {status['games_validated']} validated, "
            f"{status['games_failed']} failed, {status['games_missing']} missing "
            f"of {status['games_expected']} expected games"
        )
    if write_status:
        save(ROOT / "status.json", status, pretty=True)
    return status


def rebuild_cached_game(game_id, *, dry_run=False):
    """Rebuild one normalized game from its cached raw play-by-play; local files only.

    normalize() validates the raw file, applies only fully guarded corrections and
    enforces five-player lineups; _audit_lineage() supplies the audit's fingerprints.
    Dry runs and failures write nothing. Otherwise the result is staged beside the
    target and replaces it only after the staged copy verifies. Raw files,
    status.json and estimate outputs are never touched; rerun --offline afterwards
    so the audit (and so the dashboard and model gate) attests the new file.
    """
    info = season.require_game(game_id)
    raw_path = RAW / "pbp" / f"live_{game_id}.json"
    target = NORMALIZED / f"{game_id}.json"
    report = {"game_id": game_id, "dry_run": dry_run, "written": False}
    if not raw_path.exists():
        return {**report, "status": "missing_raw",
                "reason": "Waiting on raw play-by-play; nothing was created."}
    try:
        raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        result = normalize(info)
    except Exception as error:
        message = f"{type(error).__name__}: {error}".replace(str(RAW), "<raw_cache>")
        return {**report, "status": "failed", "error": message}
    rows = result["possessions"]
    report.update(
        possession_segments=len(rows),
        counted_possessions=sum(r["counted_possession"] for r in rows),
        model_eligible_possessions=sum(r["model_eligible"] for r in rows),
        score_by_team=result["score_by_team"],
        applied_corrections=result["applied_corrections"],
    )
    current = _audit_lineage(raw_path, target, result=result, raw_before=raw_before)
    if current["status"] == "verified":
        # Same canonical content: keep the existing bytes so audited byte hashes stay valid.
        return {**report, "status": "unchanged", "lineage": current}
    if current["status"] == "unverifiable":  # raw file changed during normalization
        return {**report, "status": "failed", "error": current["reason"]}
    if dry_run:
        action = "would_create" if current["status"] == "missing_normalized" else "would_replace"
        return {**report, "status": action, "previous_lineage_status": current["status"]}
    staged = target.with_name(target.name + ".rebuild.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(json.dumps(result, separators=(",", ":")))  # save()'s compact format
        lineage = _audit_lineage(raw_path, staged, result=result, raw_before=raw_before)
        if lineage["status"] != "verified":
            raise ValueError(f"staged file did not verify ({lineage['status']})")
        staged.replace(target)
    except Exception as error:
        staged.unlink(missing_ok=True)
        return {**report, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    return {**report, "status": "rebuilt", "written": True,
            "previous_lineage_status": current["status"], "lineage": lineage}


def rebuild_cached_season(*, dry_run=False):
    """Run rebuild_cached_game() for every listed game in game-ID order; local files only.

    Each game stages and replaces its own file, so one failure cannot touch another
    game's output. status.json and estimate outputs are never written; rerun --offline.
    """
    results = []
    for game_id in sorted(season.GAMES):
        try:
            results.append(rebuild_cached_game(game_id, dry_run=dry_run))
        except Exception as error:  # isolate: record and continue with the next game
            results.append({"game_id": game_id, "dry_run": dry_run, "written": False,
                            "status": "failed", "error": f"{type(error).__name__}: {error}"})
    counts = Counter(r["status"] for r in results)
    return {"dry_run": dry_run, "games": len(results), "counts": dict(sorted(counts.items())),
            "results": results}


def main(offline=False):
    """Offline audits all existing caches; online only downloads missing raw files."""
    if offline:
        status = audit_cached_season()
        print(json.dumps({k: status[k] for k in (
            "games_expected", "games_cached", "games_validated", "games_failed",
            "games_missing", "possession_segments", "counted_possessions",
            "model_eligible_possessions", "complete",
        )}, indent=2), flush=True)
        if not status["complete"]:
            print(status["error"], flush=True)
            raise SystemExit(1)
        return status
    try:
        report = download_missing_games()
    except Exception as error:  # game list or storage check failed: no game was requested
        print(f"STOPPED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    print(f"Downloaded {len(report['downloaded_game_ids'])}, already cached {report['already_cached']}, "
          f"failed {len(report['failed_game_ids'])} of {report['games_listed']} listed games. "
          "Nothing was normalized or audited: run --rebuild-all, then --offline.", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(1 if report["failed_game_ids"] else 0)


def cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="audit every listed game using cached files only; never download")
    parser.add_argument("--rebuild", metavar="GAME_ID", help="rebuild one normalized game from its cached raw file; never download")
    parser.add_argument("--rebuild-all", action="store_true", help="rebuild every listed game whose raw file is cached; never download")
    parser.add_argument("--dry-run", action="store_true", help="with --rebuild/--rebuild-all: report what would change and write nothing")
    parser.add_argument("--storage-preflight", action="store_true",
                        help="read-only disk-space estimate from local caches; never download or process games")
    args = parser.parse_args(argv)
    if args.storage_preflight:
        if args.offline or args.rebuild or args.rebuild_all or args.dry_run:
            parser.error("--storage-preflight cannot be combined with processing options")
        try:
            report = storage_preflight()
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(2, f"Storage preflight failed: {error}\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        raise SystemExit(0 if report["passes_safety_margin"] else 1)
    if args.dry_run and not (args.rebuild or args.rebuild_all):
        parser.error("--dry-run requires --rebuild GAME_ID or --rebuild-all")
    if args.rebuild and args.rebuild_all:
        parser.error("use --rebuild or --rebuild-all, not both")
    if args.rebuild_all:
        batch = rebuild_cached_season(dry_run=args.dry_run)
        # Human summary on stderr keeps stdout pure JSON for machines.
        print(("DRY RUN: " if args.dry_run else "") + ", ".join(f"{k}={v}" for k, v in batch["counts"].items())
              + f" of {batch['games']} listed games; rerun --offline to re-audit", file=sys.stderr, flush=True)
        print(json.dumps(batch, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1 if batch["counts"].get("failed") else 0)
    if args.rebuild:
        report = rebuild_cached_game(args.rebuild, dry_run=args.dry_run)
        print(json.dumps(report, indent=2), flush=True)
        success = report["status"] in ("unchanged", "rebuilt", "would_create", "would_replace")
        raise SystemExit(0 if success else 1)
    main(offline=args.offline)


if __name__ == "__main__":
    cli()
