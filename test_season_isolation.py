"""Tiny synthetic replay fixtures; no claim that they represent real NBA possessions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as dashboard
import data_loader
import model as lineup_model
import possession_model as model
import possessions_loader as loader
import season_config as season

ORIGINAL_NORMALIZE = loader.normalize


class SeasonIsolationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name) / "data"
        self.root = self.data / "cache/possessions"
        self.raw = self.root / "pbpstats/pbp"
        self.normalized = self.root / "games"
        self.raw.mkdir(parents=True)
        self.normalized.mkdir()
        self.games = {gid: copy.deepcopy(season.GAMES[gid]) for gid in sorted(season.GAMES)[:3]}
        self.ids = list(self.games)
        self.enterContext(patch.object(season, "GAMES", self.games))
        for name, value in (("ROOT", self.root), ("RAW", self.raw.parent), ("NORMALIZED", self.normalized)):
            self.enterContext(patch.object(loader, name, value))
        self.enterContext(patch.object(dashboard, "DATA_DIR", self.data))
        for cache in (dashboard._file_cache, dashboard._gate_cache, dashboard._name_cache):
            cache.clear()
            self.addCleanup(cache.clear)
        for target in ("requests.sessions.Session.request", "socket.socket.connect", "socket.create_connection",
                       "possessions_loader.fetch_json"):
            self.enterContext(patch(target, side_effect=AssertionError("network forbidden")))
        self.outputs = {}
        self.enterContext(patch.object(loader, "normalize", side_effect=lambda info: copy.deepcopy(self.outputs[info['GameId']])))
        (self.root / "games.json").write_text(json.dumps({"results": list(self.games.values())}))
        for name in ("possession_model.json", "possession_players.csv", "possession_diagnostics.json"):
            (self.data / name).write_bytes(b"existing output must not be overwritten\n")

    def write_game(self, gid):
        raw = {"game": {"gameId": gid, "actions": [{"period": p} for p in range(1, 5)]}}
        raw["game"]["actions"][-1].update(actionType="game", subType="end", clock="PT00M00.00S")
        normalized = {"schema_version": loader.NORMALIZED_SCHEMA_VERSION,
                      "pbpstats_version": loader.version("pbpstats"), "game": self.games[gid],
                      "applied_corrections": [], "score_by_team": {},
                      "counted_possessions_by_team": {}, "possession_count_difference_vs_api": {},
                      "possessions": [{"game_id": gid, "period": 1, "number": 1, "flags": [],
                                       "counted_possession": True, "model_eligible": True,
                                       "offense_team_id": "1", "defense_team_id": "2", "points": 1,
                                       "offense_players": [1, 2, 3, 4, 5], "defense_players": [6, 7, 8, 9, 10]}]}
        (self.raw / f"live_{gid}.json").write_text(json.dumps(raw))
        (self.normalized / f"{gid}.json").write_text(json.dumps(normalized))
        self.outputs[gid] = normalized
        return raw, normalized

    def snapshot(self):
        return {p.relative_to(self.data): p.read_bytes() for p in self.data.rglob("*") if p.is_file()}

    def assert_blocked_without_output_writes(self):
        before = self.snapshot()
        with patch.object(model, "_solve_compact_ridge", side_effect=AssertionError("must not fit")):
            for destination in (self.data, self.data / "new_outputs"):
                with self.assertRaises(model.IncompleteSeasonError):
                    model.fit_possession_ridge(status_path=self.root / "status.json", output_dir=destination)
        with patch.object(dashboard.pd, "read_csv", side_effect=AssertionError("must not read estimates")):
            response = dashboard.app.test_client().get('/?player=1')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("Possession-level player estimates", html)
        self.assertNotIn('class="player-detail"', html)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.data / "new_outputs").exists())

    def test_valid_2025_game_and_exact_manifest_coverage_are_accepted(self):
        for gid in self.ids:
            raw, normalized = self.write_game(gid)
            self.assertIsNone(loader.live_pbp_problem(gid, raw))
            self.assertEqual(model.build_possession_rows(normalized)[0].game_id, gid)
        status = loader.audit_cached_season()
        self.assertEqual(status["games_expected"], len(self.games))
        self.assertEqual(status["season"], "2025-26")
        with patch.object(data_loader, "RAW_DIR", self.root):
            self.assertEqual(data_loader.load_games(), list(self.games.values()))
        self.assertEqual(model.require_complete_season(self.root / "status.json"), status)
        self.assertTrue(dashboard.possession_audit(status, None)["ready"])
        before = self.snapshot()
        for gid in self.ids:
            self.assertEqual(model.check_game_inputs(gid, status['game_results'][gid], self.raw, self.normalized).category, 'fresh')
        self.assertEqual(self.snapshot(), before)

    def test_verified_2025_fixture_outputs_are_repeatable_and_displayed(self):
        for gid in self.ids:
            self.write_game(gid)
        loader.audit_cached_season()
        status = self.root / "status.json"
        model.fit_possession_ridge(status_path=status, output_dir=self.data)
        metadata_path = self.data / "possession_model.json"
        metadata = json.loads(metadata_path.read_text())
        diagnostics_path = self.data / "possession_diagnostics.json"
        diagnostics = diagnostics_path.read_bytes()
        players_path = self.data / "possession_players.csv"
        players = players_path.read_bytes()
        self.assertEqual(metadata["season"], "2025-26")
        self.assertEqual(metadata["games_used"], len(self.games))
        self.assertEqual(json.loads(diagnostics)["season"], "2025-26")
        model.fit_possession_ridge(status_path=status, output_dir=self.data)
        repeated = json.loads(metadata_path.read_text())
        metadata.pop("created_at")
        repeated.pop("created_at")
        self.assertEqual(metadata, repeated)
        self.assertEqual(players, players_path.read_bytes())
        self.assertEqual(diagnostics, diagnostics_path.read_bytes())
        client = dashboard.app.test_client()
        page = client.get('/?min_poss=0')
        self.assertEqual(page.status_code, 200)
        self.assertIn("Possession-level player estimates", page.get_data(as_text=True))
        self.assertNotIn("Lineup-total player estimates", page.get_data(as_text=True))
        exported = client.get('/diagnostics.json')
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.get_json()["season"], "2025-26")

    def test_2024_game_is_rejected_and_leftovers_are_ignored(self):
        old = "0022400061"
        with patch.object(loader, "Client", side_effect=AssertionError("old game must not reach parser")):
            with self.assertRaises(ValueError):
                ORIGINAL_NORMALIZE({"GameId": old})
        with self.assertRaises(ValueError):
            data_loader.load_game_lineups(old)
        self.assertEqual(loader.live_pbp_problem(old, {"game": {"gameId": old}}), "wrong_season_or_unlisted_game")
        with self.assertRaises(ValueError):
            loader.listed_game_ids([{"GameId": old}] + list(self.games.values())[1:])
        with self.assertRaises(ValueError):
            loader.rebuild_cached_game(old)
        for gid in self.ids:
            self.write_game(gid)
        wrong = copy.deepcopy(self.outputs[self.ids[0]])
        wrong["game"]["GameId"] = old
        with self.assertRaises(ValueError):
            model.build_possession_rows(wrong)
        (self.raw / f"live_{old}.json").write_text('{"old":true}')
        (self.normalized / f"{old}.json").write_text(json.dumps(wrong))
        self.assertEqual({r.game_id for r in model.load_cached_possession_rows(self.normalized)}, set(self.ids))
        status = loader.audit_cached_season()
        self.assertNotIn(old, status["game_results"])
        self.assertEqual(model.check_game_inputs(old, {}, self.raw, self.normalized).category, "unverifiable")
        # Even a fully re-attested wrong-season audit cannot certify this season.
        status["season"] = "2024-25"
        (self.root / "status.json").write_text(json.dumps(status))
        self.assert_blocked_without_output_writes()
        self.assertFalse(dashboard.possession_audit(status, None)["raw_complete"])
        # An old feed hidden behind a current filename is rejected, never replaced.
        target = self.raw / f"live_{self.ids[0]}.json"
        target.write_text(json.dumps({"game": {"gameId": old}}))
        original = target.read_bytes()
        report = loader.download_missing_games()
        self.assertEqual(report["failed_game_ids"], [self.ids[0]])
        self.assertEqual(report["already_cached"], 2)
        self.assertEqual(target.read_bytes(), original)

    def test_incomplete_coverage_or_self_declared_smaller_count_cannot_pass(self):
        self.write_game(self.ids[0])
        status = loader.audit_cached_season()
        self.assertEqual(status["games_missing"], 2)
        self.assertEqual(status["games_expected"], 3)
        self.assert_blocked_without_output_writes()

        # Matching summary counts are insufficient if the audit omits manifest games.
        status.update(complete=True, audit_complete=True, games_expected=1, games_cached=1,
                      games_validated=1, games_missing=0, missing_game_ids=[], error=None)
        status["expected_game_ids"] = status["cached_game_ids"] = status["validated_game_ids"] = [self.ids[0]]
        status["game_results"] = {self.ids[0]: status["game_results"][self.ids[0]]}
        (self.root / "status.json").write_text(json.dumps(status))
        self.assert_blocked_without_output_writes()


    def test_manifest_metadata_is_verified_before_loading_or_auditing(self):
        for entries in (None, [None], [{"GameId": self.ids[0]}],
                        [dict(g, Date="2024-10-22") for g in self.games.values()]):
            with self.subTest(entries=entries):
                (self.root / "games.json").write_text(json.dumps({"results": entries}))
                before = self.snapshot()
                with patch.object(data_loader, "RAW_DIR", self.root):
                    with self.assertRaises(ValueError):
                        data_loader.load_games()
                # The possession audit's list is the pinned manifest; a bad cached list is ignored.
                with patch.object(loader, "normalize", side_effect=AssertionError("must not parse")):
                    status = loader.audit_cached_season(write_status=False)
                self.assertEqual(status["expected_game_ids"], self.ids)
                self.assertEqual(status["games_missing"], len(self.games))
                self.assertEqual(self.snapshot(), before)
        with self.assertRaises(ValueError):
            data_loader.lineup_rows({"GameId": "0022400061"}, {})

    def test_legacy_model_checks_manifest_before_fitting_or_writing(self):
        for ids in (["0022400061"], self.ids[:-1], [None], self.ids):
            with self.subTest(ids=ids):
                frame = dashboard.pd.DataFrame({"game_id": ids})
                before = self.snapshot()
                with patch.object(lineup_model.pd, "read_csv", return_value=frame), \
                        patch.object(lineup_model, "build_matrix", side_effect=RuntimeError("gate passed")) as build:
                    if ids == self.ids:
                        with self.assertRaisesRegex(RuntimeError, "gate passed"):
                            lineup_model.main()
                        build.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):
                            lineup_model.main()
                        build.assert_not_called()
                self.assertEqual(self.snapshot(), before)

    def test_corrupt_audit_hides_estimates_and_diagnostics(self):
        (self.data / "status.json").write_text(json.dumps({
            "season": season.SEASON, "season_type": season.SEASON_TYPE,
            "source": "legacy fixture", "loaded_at": "offline", "games_used": 1,
            "games_in_season": len(self.games), "failed_downloads": {}, "games_failing_checks": {},
            "zero_possession_lineups_dropped": 0, "lineup_rows": 1, "offensive_possessions": 1,
        }))
        (self.data / "model.json").write_text('{"lambda": 1}')
        (self.data / "rapm.csv").write_text("must not be read")
        path = self.root / "status.json"
        for content in (b"{broken", b"null", b"[]", b'"invalid"', b"\xff"):
            with self.subTest(content=content):
                path.write_bytes(content)
                self.assert_blocked_without_output_writes()
                self.assertEqual(dashboard.app.test_client().get('/diagnostics.json').status_code, 404)

    def test_stale_unverifiable_and_wrong_season_inputs_keep_estimates_hidden(self):
        for gid in self.ids:
            self.write_game(gid)
        status = loader.audit_cached_season()
        gid = self.ids[0]
        raw = self.raw / f"live_{gid}.json"
        normalized = self.normalized / f"{gid}.json"
        original_raw, original_normalized = raw.read_bytes(), normalized.read_bytes()
        wrong_normalized = copy.deepcopy(self.outputs[gid])
        wrong_normalized["game"]["GameId"] = "0022400061"
        self.assertTrue(dashboard.possession_audit(status, None)["ready"])  # warm the gate cache
        for category, change in (
                ("stale", lambda: raw.write_bytes(original_raw + b" ")),
                ("missing", raw.unlink),
                ("missing", normalized.unlink),
                ("unverifiable", lambda: normalized.write_text("{broken")),
                ("unverifiable", lambda: normalized.write_text(json.dumps(wrong_normalized))),
                ("unverifiable", lambda: raw.write_text(json.dumps({"game": {"gameId": "0022400061"}})))):
            with self.subTest(category=category):
                try:
                    change()
                    self.assertEqual(model.check_game_inputs(gid, status['game_results'][gid], self.raw, self.normalized).category, category)
                    self.assert_blocked_without_output_writes()
                finally:
                    raw.write_bytes(original_raw)
                    normalized.write_bytes(original_normalized)


if __name__ == "__main__":
    unittest.main()
