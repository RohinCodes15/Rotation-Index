"""Period starters completed only by explicit opening-tip evidence. Offline; temp copies only."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import possessions_loader as pl
import season_config as season
from pbpstats.resources.enhanced_pbp import InvalidNumberOfStartersException
from pbpstats.resources.enhanced_pbp.live.start_of_period import LiveStartOfPeriod

REAL_PBP = Path(__file__).parent / "data/cache/possessions/pbpstats/pbp"
HOME, AWAY = 1, 2
BLOCKERS = ("0022500147", "0022500458", "0022500521", "0022500550",
            "0022500637", "0022501029", "0022501044")
EVIDENCE = json.loads((Path(__file__).parent / "fixtures/official_evidence_2025-26.json").read_text())["games"]


def tip(won, lost, period=5, descriptor="startperiod"):
    return {"period": period, "actionType": "jumpball", "subType": "recovered", "descriptor": descriptor,
            "personId": won, "teamId": HOME, "jumpBallWonPersonId": won, "jumpBallLostPersonId": lost}


def act(pid, team, period=5, action="2pt", sub=None):
    return {"period": period, "actionType": action, "subType": sub, "personId": pid, "teamId": team}


START = {"period": 5, "actionType": "period", "subType": "start", "personId": 0}


class OpeningTipPlayersTests(unittest.TestCase):
    def test_participants_get_their_only_team(self):
        events = [act(10, HOME, 4), act(20, AWAY, 4), START, tip(10, 20)]
        self.assertEqual(pl._opening_tip_players(events), {5: {10: HOME, 20: AWAY}})

    def test_ambiguous_unknown_late_or_non_opening_participants_are_ignored(self):
        events = [act(10, HOME, 4), act(10, AWAY, 4),  # two teams: ambiguous
                  START, act(30, HOME, 5, "substitution", "in"), tip(30, 40),  # 30 subbed in first; 40 kept (team from a later event)
                  act(30, HOME, 5), act(40, AWAY, 5, "2pt"), tip(10, 30, period=6, descriptor="heldball")]
        self.assertEqual(pl._opening_tip_players(events), {5: {40: AWAY}})


class StartersWithOpeningTipTests(unittest.TestCase):
    def run_fallback(self, partial, tips, game_id="0022500550"):
        def original(self, file_directory=None, ignore_missing_starters=False):
            if not ignore_missing_starters:
                raise InvalidNumberOfStartersException("four starters")
            return {team: list(players) for team, players in partial.items()}

        with patch.object(LiveStartOfPeriod, "get_period_starters", original):
            fallback = pl._starters_with_opening_tip(tips, "0022500550")
        return fallback(SimpleNamespace(period=5, game_id=game_id))

    def test_tip_participant_completes_a_four_player_team(self):
        result = self.run_fallback({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]}, {5: {5: HOME, 6: AWAY}})
        self.assertEqual(result, {HOME: [1, 2, 3, 4, 5], AWAY: [6, 7, 8, 9, 10]})

    def test_without_five_explicit_starters_the_original_error_is_raised(self):
        for partial, tips in (
                ({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]}, {}),  # no tip evidence
                ({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]}, {5: {1: HOME}}),  # tip names a known starter
                ({HOME: [1, 2, 3], AWAY: [6, 7, 8, 9, 10]}, {5: {4: HOME}}),  # still only four
                ({HOME: [1, 2, 3, 4, 5], AWAY: [6, 7, 8, 9, 10]}, {5: {11: HOME}}),  # would be six
                ({HOME: [1, 2, 3, 4]}, {5: {5: HOME}}),  # other team absent
                ({HOME: [1, 2, 3, 4], AWAY: [5, 6, 7, 8, 9]}, {5: {5: HOME}}),  # shared player
                ({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]}, {5: {0: HOME}}),  # invalid player
                ({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]}, {6: {5: HOME}})):  # evidence from another period
            with self.subTest(partial=partial, tips=tips), \
                    self.assertRaisesRegex(InvalidNumberOfStartersException, "four starters"):
                self.run_fallback(partial, tips)

    def test_tip_evidence_cannot_cross_game_identity(self):
        with self.assertRaisesRegex(InvalidNumberOfStartersException, "four starters"):
            self.run_fallback({HOME: [1, 2, 3, 4], AWAY: [6, 7, 8, 9, 10]},
                              {5: {5: HOME}}, game_id="0022400061")

    def test_historical_correction_does_not_match_any_current_blocker(self):
        for gid in BLOCKERS:
            event = SimpleNamespace(event_num=509, period=4, clock="PT04M20.00S", team_id=1610612738,
                                    action_type="substitution", sub_type="out", player1_id=201143)
            before = vars(event).copy()
            self.assertEqual(pl.apply_verified_corrections(gid, [event]), [])
            self.assertEqual(vars(event), before)


@unittest.skipUnless((REAL_PBP / "live_0022500550.json").exists(), "2025-26 raw cache not downloaded")
class RealOvertimeGamesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        (Path(temp.name) / "pbp").mkdir()
        for gid in BLOCKERS:
            shutil.copyfile(REAL_PBP / f"live_{gid}.json", Path(temp.name) / "pbp" / f"live_{gid}.json")
        self.enterContext(patch.object(pl, "RAW", Path(temp.name)))
        self.original = LiveStartOfPeriod.get_period_starters

    def test_original_parser_reproduces_all_seven_failures(self):
        with patch.object(pl, "_starters_with_opening_tip", return_value=self.original), \
                patch.object(pl, "VERIFIED_HOME_TEAMS", {}):
            for gid in BLOCKERS:
                error = ValueError if gid == "0022500147" else InvalidNumberOfStartersException
                with self.subTest(gid=gid), self.assertRaises(error):
                    pl.normalize(season.GAMES[gid])

    def test_opening_tip_participant_completes_overtime_starters(self):
        # Bona (PHI) and Washington (DAL) are named only by the period-5 opening tip.
        for gid, team, player in (("0022500550", 1610612755, 1641737), ("0022501044", 1610612742, 1629023)):
            with self.subTest(gid=gid):
                result = pl.normalize(season.GAMES[gid])
                opening = next(r for r in result["possessions"] if r["period"] == 5 and r["model_eligible"])
                lineup = opening["offense_players"] if opening["offense_team_id"] == str(team) else opening["defense_players"]
                self.assertIn(player, lineup)
                for row in result["possessions"]:
                    if row["model_eligible"]:
                        self.assertEqual(len(set(row["offense_players"])), 5)
                        self.assertEqual(len(set(row["defense_players"])), 5)
                self.assertEqual(result["applied_corrections"], [])
                self.assertEqual(result["score_by_team"], {
                    season.GAMES[gid][f"{side}TeamId"]: season.GAMES[gid][f"{side}Points"]
                    for side in ("Home", "Away")})
                self.assertIs(LiveStartOfPeriod.get_period_starters, self.original)

    def test_missing_tip_evidence_keeps_resolved_game_failed(self):
        for gid, missing in (("0022500550", 1641737), ("0022501044", 1629023)):
            path = pl.RAW / "pbp" / f"live_{gid}.json"
            feed = json.loads(path.read_text())
            for event in feed["game"]["actions"]:
                if event.get("period") == 5 and event.get("descriptor") == "startperiod":
                    for field in ("jumpBallWonPersonId", "jumpBallLostPersonId"):
                        if event.get(field) == missing:
                            del event[field]
            path.write_text(json.dumps(feed))
            with self.subTest(gid=gid), self.assertRaises(InvalidNumberOfStartersException):
                pl.normalize(season.GAMES[gid])
            self.assertIs(LiveStartOfPeriod.get_period_starters, self.original)

    def test_verified_ot_starters_match_the_resolved_fixture_entries(self):
        starters = {(g["game_id"], g["period"], g["team_id"]): g for g in EVIDENCE if g["fact"] == "period_starter"}
        resolved = {k for k, g in starters.items() if g["status"] == "resolved"}
        self.assertEqual(set(pl.VERIFIED_PERIOD_STARTERS), resolved)
        for key, entry in pl.VERIFIED_PERIOD_STARTERS.items():
            with self.subTest(key=key):
                g = starters[key]
                self.assertEqual(entry["player"], g["person_id"])
                self.assertNotIn(entry["player"], entry["parser_four"])
                self.assertEqual(len(set(entry["parser_four"])), 4)
                if g["period_starters"]:  # full official five: four plus the player, exactly
                    self.assertEqual(sorted([*entry["parser_four"], entry["player"]]), sorted(g["period_starters"]))
                if "visible_rows_OT1" in g["supporting"]:
                    rows = {pid: minutes for _, pid, minutes in g["supporting"]["visible_rows_OT1"] if pid}
                    self.assertEqual(rows[entry["player"]], "5:00")  # never a "4:60" display
                    # Every parser starter is on the official OT1 table: named there, or one
                    # of the rows whose player id the fixture did not record.
                    unnamed_rows = sum(pid is None for _, pid, _ in g["supporting"]["visible_rows_OT1"])
                    self.assertLessEqual(len(set(entry["parser_four"]) - set(rows)), unnamed_rows)
                else:  # no box-score minutes: only an explicit official five is accepted
                    self.assertEqual(len(g["period_starters"]), 5)

    def test_bkn_gamebook_starters_match_the_local_feed(self):
        g = next(g for g in EVIDENCE if g["game_id"] == "0022500521" and g["fact"] == "period_starter")
        self.assertEqual(g["source_url"], "https://statsdmz.nba.com/pdfs/20260107/20260107_ORLBKN_book.pdf")
        ids = g["supporting"]["player_ids_from_local_feed"]
        self.assertEqual(sorted(ids.values()), sorted(g["period_starters"]))
        feed = json.loads((pl.RAW / "pbp" / "live_0022500521.json").read_text())["game"]["actions"]
        on_bkn = {e["personId"] for e in feed if e.get("teamId") == 1610612751}
        self.assertLessEqual(set(g["period_starters"]), on_bkn)  # every id is a local BKN player
        self.assertEqual(ids["Noah Clowney"], 1641730)

    def test_verified_ot_starter_completes_exactly_that_team(self):
        for (gid, period, team), entry in pl.VERIFIED_PERIOD_STARTERS.items():
            with self.subTest(gid=gid, team=team):
                result = pl.normalize(season.GAMES[gid])
                opening = next(r for r in result["possessions"] if r["period"] == period and r["model_eligible"])
                side = "offense_players" if opening["offense_team_id"] == str(team) else "defense_players"
                self.assertEqual(sorted(opening[side]), sorted([*entry["parser_four"], entry["player"]]))
                for row in result["possessions"]:
                    if row["model_eligible"]:
                        self.assertEqual(len(set(row["offense_players"])), 5)
                        self.assertEqual(len(set(row["defense_players"])), 5)
                self.assertEqual(result["applied_corrections"], [])
                self.assertIs(LiveStartOfPeriod.get_period_starters, self.original)

    def test_ot_games_stay_failed_without_exact_evidence(self):
        table = pl.VERIFIED_PERIOD_STARTERS
        for (gid, period, team), entry in table.items():
            others = {k: v for k, v in table.items() if k != (gid, period, team)}
            other_team = next(t for t in (int(season.GAMES[gid]["HomeTeamId"]), int(season.GAMES[gid]["AwayTeamId"]))
                              if t != team)
            cases = {
                "missing": others,
                "period": {**others, (gid, 4, team): entry},
                "game": {**others, ("0022500001", period, team): entry},
                "team": {**others, (gid, period, other_team): entry},
                "player": {**others, (gid, period, team): {**entry, "player": 1628983}},  # not on this team
                "parser_four": {**others, (gid, period, team): {**entry, "parser_four": (*entry["parser_four"][:3], 1)}},
            }
            for name, patched in cases.items():
                with self.subTest(gid=gid, team=team, case=name), \
                        patch.object(pl, "VERIFIED_PERIOD_STARTERS", patched), \
                        self.assertRaises(InvalidNumberOfStartersException):
                    pl.normalize(season.GAMES[gid])
                self.assertIs(LiveStartOfPeriod.get_period_starters, self.original)

    def test_no_former_blocker_remains(self):
        failed = []
        for gid in BLOCKERS:
            try:
                pl.normalize(season.GAMES[gid])
            except (ValueError, InvalidNumberOfStartersException):
                failed.append(gid)
        self.assertEqual(failed, [])

    def test_verified_home_team_validates_the_neutral_site_game(self):
        game = season.GAMES["0022500147"]
        official = next(g for g in EVIDENCE if g["game_id"] == "0022500147")
        self.assertEqual((official["status"], official["team_id"]), ("resolved", 1610612765))
        self.assertEqual(pl.VERIFIED_HOME_TEAMS["0022500147"]["evidence_url"], official["source_url"])
        self.assertEqual(pl.VERIFIED_HOME_TEAMS, {"0022500147": {
            "manifest_home_team_id": "1610612742", "home_team_id": "1610612765",
            "evidence_url": "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/"
                            "boxscore/boxscore_0022500147.json"}})
        result = pl.normalize(game)
        self.assertEqual(result["score_by_team"], {"1610612765": 122, "1610612742": 110})
        self.assertEqual(result["game"], game)  # manifest metadata is never rewritten
        self.assertEqual(game["HomeTeamAbbreviation"], "DAL")

    def test_home_team_correction_is_bound_to_game_and_teams(self):
        game = season.GAMES["0022500147"]
        with patch.object(pl, "VERIFIED_HOME_TEAMS", {}), \
                self.assertRaisesRegex(ValueError, "home/away disagrees with game list: feed home score 122 is "
                                       "manifest away DET, feed away score 110 is manifest home DAL"):
            pl.normalize(game)
        entry = pl.VERIFIED_HOME_TEAMS["0022500147"]
        with patch.object(pl, "VERIFIED_HOME_TEAMS", {"0022500147": {**entry, "home_team_id": "1610612742"}}), \
                self.assertRaisesRegex(ValueError, "does not match the game list"):
            pl.normalize(game)
        with patch.object(season, "require_metadata"), \
                self.assertRaisesRegex(ValueError, "disagrees with the verified home team"):
            pl.normalize({**game, "AwayPoints": 121})
        # An entry for this game never relabels any other game.
        with patch.object(pl, "VERIFIED_HOME_TEAMS", {"0022500550": entry}), \
                patch.object(season, "require_metadata"), \
                self.assertRaisesRegex(ValueError, "does not match the game list"):
            pl.normalize(season.GAMES["0022500550"])

    def test_genuine_score_mismatch_keeps_the_side_error(self):
        game = {**season.GAMES["0022500550"], "HomePoints": 1}
        with patch.object(season, "require_metadata"), \
                self.assertRaisesRegex(ValueError, "^Home: final play-by-play score disagrees"):
            pl.normalize(game)


if __name__ == "__main__":
    unittest.main()
