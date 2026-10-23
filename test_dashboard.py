"""Offline Flask display checks. Run: python -m unittest test_dashboard."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from markupsafe import escape

import app as dashboard
import possession_model
from test_cache_fixtures import attach_lineage_fixture
from possession_model import build_design_data, build_possession_rows, check_game_inputs, fit_possession_ridge

REAL_GAME = Path(__file__).resolve().parent / "data/cache/possessions/games/0022400061.json"
REAL_GAME_LIST = REAL_GAME.parent.parent / "games.json"


class DashboardDisplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)
        self.patch = patch.object(dashboard, "DATA_DIR", self.data)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = dashboard.app.test_client()
        self.network_mocks = []
        for target in ("requests.sessions.Session.request", "socket.socket.connect", "socket.create_connection"):
            blocker = patch(target, side_effect=AssertionError("network forbidden in dashboard tests"))
            self.network_mocks.append(blocker.start())
            self.addCleanup(blocker.stop)

    def tearDown(self):
        self.assertEqual(sum(mock.call_count for mock in self.network_mocks), 0)

    def write_json(self, name, value):
        path = self.data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def legacy_files(self, aggregate=True):
        if aggregate:
            self.write_json("status.json", {
                "season": "2024-25", "season_type": "Regular Season",
                "source": "offline fixture", "loaded_at": "offline",
                "games_used": 1, "games_in_season": 1230,
                "failed_downloads": {}, "games_failing_checks": {},
                "zero_possession_lineups_dropped": 0, "lineup_rows": 2,
                "offensive_possessions": 180,
            })
        self.write_json("model.json", {"lambda": 10, "home_court": 1, "players": 1})
        (self.data / "rapm.csv").write_text(
            "name,possessions,rapm,low_sample\nLegacy Fixture Player,2000,2.5,False\n"
        )

    def page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        return response.get_data(as_text=True)

    def assert_hidden(self, html):
        self.assertNotIn("Legacy Fixture Player", html)
        self.assertNotIn("Lineup-total player estimates", html)

    def test_stale_model_files_without_aggregate_status(self):
        self.legacy_files(aggregate=False)
        self.assert_hidden(self.page())

    UNAVAILABLE = "Player estimates are unavailable until all 1,230 regular-season games validate."
    # Only IDs present in the existing season list; fixture possessions below
    # are deliberately reused test rows, never claimed to be a real season.
    FIXTURE_IDS = sorted(g["GameId"] for g in json.loads(REAL_GAME_LIST.read_text())["results"])

    def audit_status(self, failed=(), missing=(), corrections=(), eligible=1230, **extra):
        """Write an offline-audit status shaped like possessions_loader.audit_cached_season."""
        ids = self.FIXTURE_IDS
        cached = [g for g in ids if g not in missing]
        validated = [g for g in cached if g not in failed]
        errors = {g: f"ValueError: fixture failure for {g}" for g in failed}
        results = {g: {"status": "validated", "applied_corrections": []} for g in validated}
        results.update({g: {"status": "failed", "error": errors[g]} for g in failed})
        for correction in corrections:
            results[correction["game_id"]]["applied_corrections"].append(correction)
        status = {
            "season": "2024-25", "season_type": "Regular Season", "source": "offline fixture",
            "mode": "offline_audit", "audit_complete": True, "complete": len(validated) == 1230,
            "games_expected": 1230, "games_cached": len(cached), "games_validated": len(validated),
            "games_failed": len(failed), "games_missing": len(missing),
            "expected_game_ids": ids, "cached_game_ids": cached, "validated_game_ids": validated,
            "failed_game_ids": sorted(failed), "missing_game_ids": sorted(missing),
            "validation_errors": errors, "game_results": results,
            "possession_segments": 1400, "counted_possessions": 1300,
            "model_eligible_possessions": eligible, "flags": {"changing_or_missing_lineup": 70},
            "possession_count_differences": {},
        }
        if len(validated) != 1230:
            status["error"] = (f"Offline audit incomplete season: {len(validated)} validated, "
                               f"{len(failed)} failed, {len(missing)} missing of 1230 expected games")
        status.update(extra)
        if len(validated) == 1230:
            attach_lineage_fixture(status, self.data / "cache/possessions")
        self.write_json("cache/possessions/status.json", status)
        return status

    def test_legacy_files_with_possession_status(self):
        self.legacy_files()
        for complete in (False, True):
            with self.subTest(complete=complete):
                # Old loader-format status: even complete=True is not an offline audit.
                self.write_json("cache/possessions/status.json", {
                    "games_validated": 1 if not complete else 1230,
                    "games_expected": 1230, "complete": complete,
                })
                html = self.page()
                self.assert_hidden(html)
                self.assertNotIn("python model.py", html)
                self.assertIn(self.UNAVAILABLE, html)
                self.assertIn("not written by the offline audit", html)
                self.assertIn("Loader progress report, not an offline audit", html)

    def test_empty_possession_status_still_hides_legacy(self):
        self.legacy_files()
        self.write_json("cache/possessions/status.json", {})
        self.assert_hidden(self.page())

    def test_complete_legacy_inputs_show_labeled_estimates(self):
        self.legacy_files()
        html = self.page()
        self.assertIn("Lineup-total player estimates", html)
        self.assertIn("Legacy Fixture Player", html)

    def test_missing_model_metadata_hides_estimates(self):
        self.legacy_files()
        (self.data / "model.json").unlink()
        self.assert_hidden(self.page())

    def test_missing_csv_hides_estimates(self):
        self.legacy_files()
        (self.data / "rapm.csv").unlink()
        self.assert_hidden(self.page())

    def complete_possession_outputs(self, complete=True):
        """Write status plus outputs produced by the real pipeline writer.

        Test-only 1,230-group fixture reusing one real cached game's rows; it is
        not a real season and everything lives in the temporary data folder.
        """
        real = build_design_data(build_possession_rows(json.loads(REAL_GAME.read_text())))
        take = np.arange(1230) % len(real.game_ids)
        design = replace(real, offense_indices=real.offense_indices[take],
                         defense_indices=real.defense_indices[take],
                         points_per_possession=real.points_per_possession[take],
                         game_ids=np.array(self.FIXTURE_IDS))
        self.audit_status()
        fit_possession_ridge(design, status_path=self.data / "cache/possessions/status.json",
                             output_dir=self.data)
        if not complete:
            self.audit_status(failed=self.FIXTURE_IDS[:1], missing=self.FIXTURE_IDS[2:])
        return design

    def assert_possession_hidden(self, html):
        self.assertNotIn("Possession-level player estimates", html)
        self.assertNotIn("Possession model details", html)

    def test_complete_cache_shows_possession_outputs_and_hides_legacy(self):
        self.legacy_files()
        design = self.complete_possession_outputs()
        table = pd.read_csv(self.data / "possession_players.csv")
        html = self.client.get("/?min_poss=0").get_data(as_text=True)
        self.assert_hidden(html)
        self.assertIn("Possession-level player estimates", html)
        self.assertIn("Possession model details", html)
        self.assertNotIn("Possession cache incomplete", html)
        for heading in ("Off. poss.", "Def. poss.", "Offense", "Defense", "Net"):
            self.assertIn(f"<th>{heading}</th>", html)
        self.assertIn("Defense is points prevented", html)
        metadata = json.loads((self.data / "possession_model.json").read_text())
        self.assertIn(f"<td>{metadata['selected_penalty']}</td>", html)
        self.assertIn("984 training games, 246 holdout games", html)
        self.assertIn("1,230 of 1,230 validated", html)
        self.assertIn("1,230 of 1,300 counted", html)
        self.assertIn("changing or missing lineup: 70", html)
        self.assertIn("All 1,230 games validated.", html)
        self.assertIn("These outputs match the current audit", html)
        self.assertNotIn(self.UNAVAILABLE, html)
        # Rows appear in net order with their own counts and signed values.
        best = table.sort_values(["net_per_100", "player_id"], ascending=[False, True]).to_dict("records")[0]
        row = (f"<td>1</td>\n                        <td>Player ID {best['player_id']}</td>\n"
               f"                        <td>{best['offense_possessions']:,}</td>\n"
               f"                        <td>{best['defense_possessions']:,}</td>")
        self.assertIn(row, html)
        for value in (best["offense_per_100"], best["defense_per_100"], best["net_per_100"]):
            self.assertIn("%+.2f" % value, html)
        self.assertEqual(html.count("<tr>\n                        <td>"), len(design.player_ids))

    OUT_OF_DATE = "All games validated, but some model inputs no longer match the audited cache"

    def input_paths(self, game_id):
        root = self.data / "cache/possessions"
        return root / "pbpstats/pbp" / f"live_{game_id}.json", root / "games" / f"{game_id}.json"

    def assert_gate_rejects(self):
        with self.assertRaises(dashboard.IncompleteSeasonError):
            dashboard.require_complete_season(self.data / "cache/possessions/status.json")

    def test_fresh_inputs_show_estimates_and_freshness_counts(self):
        self.legacy_files()
        self.complete_possession_outputs()
        dashboard.require_complete_season(self.data / "cache/possessions/status.json")
        html = self.page()
        self.assertIn("Possession-level player estimates", html)
        self.assert_hidden(html)
        self.assertIn("<tr><th>Fresh (match the audit)</th><td>1,230 of 1,230</td></tr>", html)
        for label in ("Stale (changed since the audit)", "Missing normalized file", "Could not be verified",
                      "Waiting on raw play-by-play"):
            self.assertIn(f"<tr><th>{label}</th><td>0</td></tr>", html)
        self.assertIn("All model inputs match the audited cache.", html)
        self.assertIn("Every game's model inputs match the audit", html)
        self.assertNotIn(self.OUT_OF_DATE, html)

    def test_stale_inputs_hide_estimates_and_explain_what_changed(self):
        self.legacy_files()
        self.complete_possession_outputs()
        game_id = self.FIXTURE_IDS[7]
        raw_path, normalized_path = self.input_paths(game_id)
        original = {path: path.read_bytes() for path in (raw_path, normalized_path)}
        saved_outputs = {name: (self.data / name).read_bytes()
                         for name in ("possession_players.csv", "possession_model.json")}
        normalized = json.loads(original[normalized_path])

        def rewrite(path, value):
            return lambda: path.write_text(json.dumps(value))

        cases = (
            ("raw edited", "stale", "Changed since the audit: the raw play-by-play file.",
             lambda: raw_path.write_bytes(original[raw_path] + b" ")),
            ("normalized reformatted", "stale", "Changed since the audit: the normalized file's formatting.",
             lambda: normalized_path.write_text(json.dumps(normalized, indent=2))),
            ("possession data edited", "stale", "the normalized possession data",
             rewrite(normalized_path, {**normalized, "possessions": [{"game_id": game_id,
                 "period": 1, "number": 1, "model_eligible": False, "counted_possession": False}]})),
            ("corrections edited", "stale", "the recorded corrections",
             rewrite(normalized_path, {**normalized, "applied_corrections": [{"action_number": 1}]})),
            ("parser version edited", "stale", "the parser version record",
             rewrite(normalized_path, {**normalized, "pbpstats_version": "0.0"})),
            ("normalized deleted", "missing", "The normalized model-input file has been removed since the audit.",
             normalized_path.unlink),
            ("normalized corrupted", "unverifiable", "no longer valid JSON",
             lambda: normalized_path.write_text("{not json")),
        )
        for label, category, reason, change in cases:
            with self.subTest(case=label):
                try:
                    self.check_input_change(game_id, category, reason, change, saved_outputs)
                finally:
                    for path, data in original.items():
                        path.write_bytes(data)
        self.assertIn("Possession-level player estimates", self.page())

    def check_input_change(self, game_id, category, reason, change, saved_outputs):
        # Warm the in-process digest cache first, so a stale result proves it refreshed.
        self.assertIn("Possession-level player estimates", self.page())
        change()
        html = self.page()
        self.assert_possession_hidden(html)
        self.assert_hidden(html)
        self.assert_gate_rejects()
        self.assertIn("until every game's model inputs match the audited cache", html)
        self.assertNotIn(self.UNAVAILABLE, html)
        self.assertIn(f'<span class="tag {category}-tag">{category}</span> <strong>{game_id}', html)
        self.assertIn(str(escape(reason)), html)
        labels = {"stale": "Stale (changed since the audit)", "missing": "Missing normalized file",
                  "unverifiable": "Could not be verified"}
        self.assertIn(f"<tr><th>{labels[category]}</th><td>1</td></tr>", html)
        self.assertIn("<tr><th>Fresh (match the audit)</th><td>1,229 of 1,230</td></tr>", html)
        # Cache counts and audit details are preserved; saved outputs are untouched.
        self.assertIn("<tr><th>Validated</th><td>1,230</td></tr>", html)
        self.assertEqual({n: (self.data / n).read_bytes() for n in saved_outputs}, saved_outputs)

    def test_unverified_audit_lineage_hides_estimates(self):
        self.complete_possession_outputs()
        status_path = self.data / "cache/possessions/status.json"
        audited = json.loads(status_path.read_text())
        game_id = self.FIXTURE_IDS[3]
        for label, lineage, category, reason in (
            ("mismatch at audit", {"status": "mismatch", "verified": False}, "stale",
             "did not match a fresh rebuild"),
            ("no normalized file at audit", {"status": "missing_normalized", "verified": False}, "missing",
             "no normalized model-input file existed when the audit ran"),
            ("replay failed", {"status": "unverifiable", "verified": False, "reason": "raw file changed during normalization"},
             "unverifiable", "raw file changed during normalization"),
            ("older audit format", None, "unverifiable", "no input record for this game"),
        ):
            with self.subTest(case=label):
                status = json.loads(json.dumps(audited))
                if lineage is None:
                    del status["game_results"][game_id]["lineage"]
                else:
                    status["game_results"][game_id]["lineage"] = {
                        "method": "canonical_normalization_replay_v1", **lineage}
                status_path.write_text(json.dumps(status))
                html = self.page()
                self.assert_possession_hidden(html)
                self.assert_gate_rejects()
                self.assertIn(f'<span class="tag {category}-tag">{category}</span> <strong>{game_id}', html)
                self.assertIn(reason, html)
                self.assertIn(self.OUT_OF_DATE, html)

    def gate_error(self):
        try:
            dashboard.require_complete_season(self.data / "cache/possessions/status.json")
        except dashboard.IncompleteSeasonError as error:
            return str(error)
        return None

    def test_dashboard_and_model_gate_agree_on_every_input_state(self):
        self.legacy_files()
        self.complete_possession_outputs()
        outputs = {name: (self.data / name).read_bytes() for name in
                   ("possession_players.csv", "possession_model.json", "rapm.csv", "model.json")}
        game_id = self.FIXTURE_IDS[11]
        root = self.data / "cache/possessions"
        raw_path, normalized_path = self.input_paths(game_id)
        status_path = root / "status.json"
        originals = {path: path.read_bytes() for path in (raw_path, normalized_path, status_path)}
        normalized = json.loads(originals[normalized_path])
        audited = json.loads(originals[status_path])
        read_bytes = Path.read_bytes

        def unreadable(path):
            if path == normalized_path:
                raise PermissionError("fixture unreadable normalized file")
            return read_bytes(path)

        blocked_read = patch.object(Path, "read_bytes", unreadable)

        def rewrite(value):
            return lambda: normalized_path.write_text(json.dumps(value))

        def edit_audit_record(edit):
            def change():
                status = json.loads(json.dumps(audited))
                edit(status["game_results"][game_id])
                status_path.write_text(json.dumps(status))
            return change

        cases = (
            ("fresh inputs", "fresh", None),
            ("raw file changed", "stale", lambda: raw_path.write_bytes(originals[raw_path] + b"\n")),
            ("normalized file reformatted", "stale",
             lambda: normalized_path.write_text(json.dumps(normalized, indent=2))),
            ("possessions changed", "unverifiable", rewrite({**normalized, "possessions": [{"points": 3}]})),
            ("valid possession data changed", "stale", rewrite({**normalized, "possessions": [
                {"game_id": game_id, "period": 1, "number": 1,
                 "model_eligible": False, "counted_possession": False}]})),
            ("corrections changed", "stale", rewrite({**normalized, "applied_corrections": [{"action_number": 9}]})),
            ("audit correction record changed", "stale", edit_audit_record(
                lambda r: r["lineage"].update(recorded_corrections_sha256="0" * 64))),
            ("parser version changed", "stale", rewrite({**normalized, "pbpstats_version": "0.0"})),
            ("normalized file deleted", "missing", normalized_path.unlink),
            ("raw file deleted", "missing", raw_path.unlink),
            ("corrupt normalized JSON", "unverifiable", lambda: normalized_path.write_text("{not json")),
            ("unreadable normalized file", "unverifiable", blocked_read.start),
            ("normalized fields missing", "unverifiable", rewrite({"game": {}})),
            ("audit replay unverifiable", "unverifiable", edit_audit_record(
                lambda r: r["lineage"].update(status="unverifiable", verified=False, reason="replay failed"))),
            ("audit lineage absent", "unverifiable", edit_audit_record(lambda r: r.pop("lineage"))),
            ("audit game record absent", "unverifiable", lambda: self.write_json(
                "cache/possessions/status.json", {**audited, "game_results": {
                    gid: record for gid, record in audited["game_results"].items() if gid != game_id}})),
            ("audit attestation absent", "unverifiable", edit_audit_record(
                lambda r: r["lineage"].pop("verified_inputs"))),
            ("unknown lineage method", "unverifiable", edit_audit_record(
                lambda r: r["lineage"].update(method="other_method"))),
            ("mismatch recorded at audit", "stale", edit_audit_record(
                lambda r: r["lineage"].update(status="mismatch", verified=False))),
        )
        for label, category, change in cases:
            with self.subTest(case=label):
                try:
                    self.page()  # warm the dashboard's file cache before the change
                    if change:
                        change()
                    if label == "unreadable normalized file":
                        dashboard._file_cache.clear()
                    html = self.page()
                    error = self.gate_error()
                    record = json.loads(status_path.read_text())["game_results"].get(game_id)
                    uncached = check_game_inputs(game_id, record, root / "pbpstats/pbp", root / "games")
                    cached = check_game_inputs(game_id, record, root / "pbpstats/pbp", root / "games",
                                               cache=dashboard._file_cache)
                    self.assertEqual(cached, uncached)
                    self.assertEqual(uncached.category, category)
                    if category == "fresh":
                        self.assertIsNone(error)
                        self.assertIn("Possession-level player estimates", html)
                        self.assertIn("<tr><th>Fresh (match the audit)</th><td>1,230 of 1,230</td></tr>", html)
                    else:
                        # The gate stops on this game with exactly the dashboard's classification.
                        self.assertIsNotNone(error)
                        if record is not None:
                            self.assertIn(f"game {game_id}:", error)
                            self.assertIn(f"({category}: {uncached.reason})", error)
                        self.assert_possession_hidden(html)
                        self.assertIn(f'<span class="tag {category}-tag">{category}</span> <strong>{game_id}', html)
                        self.assertIn(str(escape(uncached.reason)), html)
                        with patch("possession_model._solve_compact_ridge") as solver:
                            for destination in (self.data, self.data / "new_outputs"):
                                with self.assertRaises(possession_model.IncompleteSeasonError):
                                    fit_possession_ridge(status_path=status_path, output_dir=destination)
                            solver.assert_not_called()
                        self.assertFalse((self.data / "new_outputs").exists())
                    self.assert_hidden(html)
                    self.assertIn("<tr><th>Validated</th><td>1,230</td></tr>", html)
                    self.assertIn("changing or missing lineup: 70", html)
                    self.assertEqual({name: (self.data / name).read_bytes() for name in outputs}, outputs)
                finally:
                    blocked_read.stop()
                    dashboard._file_cache.clear()
                    for path, data in originals.items():
                        path.write_bytes(data)

    def test_dashboard_and_gate_share_one_comparison(self):
        self.complete_possession_outputs()
        dashboard._gate_cache.clear()
        calls = {"dashboard": 0, "gate": 0}

        def spy(name):
            def wrapped(*args, **kwargs):
                calls[name] += 1
                return original(*args, **kwargs)
            return wrapped

        original = possession_model.check_game_inputs
        with patch.object(dashboard, "check_game_inputs", spy("dashboard")), \
                patch.object(possession_model, "check_game_inputs", spy("gate")):
            html = self.page()
        self.assertIn("Possession-level player estimates", html)
        self.assertEqual(calls, {"dashboard": 1230, "gate": 1230})
        self.assertFalse(hasattr(dashboard, "_game_freshness"))

    def test_partial_season_reports_inputs_waiting_on_raw_data(self):
        self.audit_status(missing=self.FIXTURE_IDS[1:])
        status_path = self.data / "cache/possessions/status.json"
        status = json.loads(status_path.read_text())
        attach_lineage_fixture(status, self.data / "cache/possessions")
        for game_id in self.FIXTURE_IDS[1:]:
            status["game_results"][game_id] = {"status": "missing", "lineage": {
                "method": "canonical_normalization_replay_v1", "status": "missing_raw", "verified": False}}
        status_path.write_text(json.dumps(status))
        html = self.page()
        self.assert_gate_rejects()
        self.assert_possession_hidden(html)
        self.assertIn(self.UNAVAILABLE, html)
        self.assertNotIn(self.OUT_OF_DATE, html)
        self.assertIn("<tr><th>Fresh (match the audit)</th><td>1 of 1,230</td></tr>", html)
        self.assertIn("<tr><th>Waiting on raw play-by-play</th><td>1,229</td></tr>", html)
        self.assertNotIn('class="input-problems"', html)
        for label in ("Stale (changed since the audit)", "Could not be verified", "Missing normalized file"):
            self.assertIn(f"<tr><th>{label}</th><td>0</td></tr>", html)

    def test_min_possessions_filters_on_offense_plus_defense(self):
        self.complete_possession_outputs()
        table = pd.read_csv(self.data / "possession_players.csv")
        on_court = table.offense_possessions + table.defense_possessions
        threshold = int(on_court.median()) + 1
        html = self.client.get(f"/?min_poss={threshold}").get_data(as_text=True)
        self.assertEqual(html.count("<tr>\n                        <td>"), int((on_court >= threshold).sum()))

    def test_incomplete_cache_hides_existing_possession_outputs(self):
        self.legacy_files()
        self.complete_possession_outputs(complete=False)
        html = self.page()
        self.assert_possession_hidden(html)
        self.assert_hidden(html)
        self.assertIn(self.UNAVAILABLE, html)
        self.assertIn("1 cached game failed validation", html)

    def test_missing_failed_or_unfinished_audit_hides_saved_estimates(self):
        self.legacy_files()
        self.complete_possession_outputs()
        saved = {name: (self.data / name).read_bytes() for name in
                 ("possession_players.csv", "possession_model.json")}
        for label, change in (
            ("missing game", {"missing": self.FIXTURE_IDS[:1]}),
            ("failed game", {"failed": self.FIXTURE_IDS[:1]}),
            ("unfinished audit", {"audit_complete": False}),
        ):
            with self.subTest(case=label):
                self.audit_status(**change)
                with patch.object(dashboard, "cached_player_names") as names:
                    html = self.page()
                    names.assert_not_called()
                self.assert_possession_hidden(html)
                self.assert_hidden(html)
                self.assertIn(self.UNAVAILABLE, html)
                self.assertEqual({name: (self.data / name).read_bytes() for name in saved}, saved)

    def test_complete_cache_without_outputs_shows_no_estimates(self):
        self.audit_status()
        html = self.page()
        self.assert_possession_hidden(html)
        self.assertIn("no possession-model outputs were found", html)
        self.assertIn("All 1,230 games validated.", html)

    def test_incomplete_audit_shows_progress_counts_failures_and_corrections(self):
        self.write_json("cache/possessions/games.json", {"results": [
            {"GameId": "0022400001", "Date": "2024-11-12", "HomeTeamAbbreviation": "BOS", "AwayTeamAbbreviation": "ATL"},
        ]})
        correction = {"game_id": "0022400002", "action_number": 509, "period": 4, "display_clock": "4:20",
                      "action_type": "substitution", "sub_type": "out", "expected_feed_player1_id": 201143,
                      "corrected_player1_id": 1628436, "corrected_player_name": "Fixture Player",
                      "reason": "Fixture game book reason.", "evidence_url": "https://example.invalid/book.pdf"}
        failed, missing = ["0022400001"], self.FIXTURE_IDS[5:10] + self.FIXTURE_IDS[20:]
        self.audit_status(failed=failed, missing=missing, corrections=[correction])
        html = self.page()
        self.assert_gate_rejects()
        self.assert_possession_hidden(html)
        self.assertIn("<strong>14 of 1,230</strong> regular-season games validated (1.1%)", html)
        for label, value in (("Cached (play-by-play on disk)", "15"), ("Validated", "14"),
                             ("Failed validation", "1"), ("Not cached yet", "1,215")):
            self.assertIn(f"<tr><th>{label}</th><td>{value}</td></tr>", html)
        self.assertIn(self.UNAVAILABLE, html)
        self.assertIn("1,215 games have no cached play-by-play yet.", html)
        self.assertIn("1 cached game failed validation", html)
        # Failure details name the game from the local game list and quote the error.
        self.assertIn("<strong>0022400001 (ATL at BOS, 2024-11-12)</strong>: <code>ValueError: fixture failure for 0022400001</code>", html)
        self.assertIn("corrected to Fixture Player (ID 1628436)", html)
        self.assertIn("the feed listed player 201143", html)
        self.assertIn("Fixture game book reason.", html)
        self.assertIn('href="https://example.invalid/book.pdf"', html)
        self.assertIn(f"{self.FIXTURE_IDS[5]}–{self.FIXTURE_IDS[9]}, {self.FIXTURE_IDS[20]}–{self.FIXTURE_IDS[-1]}", html)
        self.assertIn('class="bar validated" style="width: 1.14%"', html)
        self.assertIn('class="bar failed"', html)

    def test_clean_partial_audit_reports_no_failures_or_corrections(self):
        self.audit_status(missing=self.FIXTURE_IDS[1:])
        html = self.page()
        self.assertIn("No cached game failed validation.", html)
        self.assertIn("No corrections recorded.", html)
        self.assertNotIn('class="bar failed"', html)
        self.assertNotIn("cached games failed validation", html)

    def test_inconsistent_complete_audit_is_not_ready(self):
        # Summary flags say complete, but one per-game result disagrees: the model gate rejects it.
        self.audit_status()
        status = json.loads((self.data / "cache/possessions/status.json").read_text())
        status["game_results"][self.FIXTURE_IDS[0]] = {"status": "failed", "error": "late failure"}
        self.write_json("cache/possessions/status.json", status)
        html = self.page()
        self.assert_possession_hidden(html)
        self.assertIn(self.UNAVAILABLE, html)
        self.assertIn("late failure", html)

    def test_outputs_from_an_earlier_audit_state_are_hidden(self):
        self.complete_possession_outputs()
        for name, change in (
            ("eligible possessions", {"model_eligible_possessions": 1229}),
            ("source", {"source": "re-audited fixture"}),
        ):
            with self.subTest(name):
                self.audit_status(**{"eligible": change.get("model_eligible_possessions", 1230),
                                     **{k: v for k, v in change.items() if k != "model_eligible_possessions"}})
                html = self.page()
                self.assert_possession_hidden(html)
                self.assertIn("do not match the current possession cache", html)
                self.assertIn(name, html.split("Checks that did not match:")[1].split("</p>")[0])

    def write_name_caches(self, ids):
        """Synthetic local name sources: lineup full names for ids[:5], live-feed names for ids[3:7]."""
        lineup = {"EntityId": "-".join(str(i) for i in ids[:5]),
                  "Name": ", ".join(f"Lineup Name {i}" for i in ids[:5])}
        broken = {"EntityId": "-".join(str(i) for i in ids[5:10]), "Name": "Too, Few"}
        self.write_json("raw/lineups/0000000001.json",
                        {"stats": {"Home": {"FullGame": [lineup, broken]}, "Away": {"FullGame": []}}})
        (self.data / "raw/lineups/0000000002.json").write_text("{not json")
        self.write_json("cache/possessions/pbpstats/pbp/live_0000000001.json", {"game": {"actions": [
            {"personId": i, "playerNameI": f"L. Live{i}"} for i in ids[3:7]
        ] + [{"personId": 0, "playerNameI": ""}]}})

    def player_cell(self, html, player_id, text):
        return f'<span title="NBA player ID {player_id}">{text}</span>' in html

    def test_player_names_prefer_lineup_cache_then_live_feed_then_id(self):
        self.complete_possession_outputs()
        ids = sorted(pd.read_csv(self.data / "possession_players.csv").player_id.tolist())
        self.write_name_caches(ids)
        html = self.client.get("/?min_poss=0").get_data(as_text=True)
        for i in ids[:5]:
            self.assertTrue(self.player_cell(html, i, f"Lineup Name {i}"), i)
            self.assertNotIn(f"L. Live{i}", html)
        for i in ids[5:7]:
            self.assertTrue(self.player_cell(html, i, f"L. Live{i}"), i)
        for i in ids[7:]:
            self.assertIn(f"<td>Player ID {i}</td>", html)
        self.assertNotIn("Too", html)

    def test_incomplete_cache_hides_names_with_estimates(self):
        self.complete_possession_outputs(complete=False)
        ids = sorted(pd.read_csv(self.data / "possession_players.csv").player_id.tolist())
        self.write_name_caches(ids)
        with patch.object(dashboard, "cached_player_names") as names:
            html = self.page()
            names.assert_not_called()
        self.assert_possession_hidden(html)
        self.assertNotIn("Lineup Name", html)

    def test_mismatched_or_malformed_outputs_are_hidden(self):
        self.complete_possession_outputs()
        metadata = json.loads((self.data / "possession_model.json").read_text())
        csv_text = (self.data / "possession_players.csv").read_text()
        for name, change in (
            ("other season", lambda: self.write_json("possession_model.json", {**metadata, "season": "2023-24"})),
            ("partial games", lambda: self.write_json("possession_model.json", {**metadata, "games_used": 1})),
            ("defense sign", lambda: self.write_json("possession_model.json", {**metadata, "defense_sign": "points allowed"})),
            ("missing row", lambda: (self.data / "possession_players.csv").write_text(csv_text.rsplit("\n", 2)[0] + "\n")),
            ("renamed column", lambda: (self.data / "possession_players.csv").write_text(csv_text.replace("net_per_100", "net", 1))),
            ("other game IDs", lambda: self.write_json("possession_model.json", {
                **metadata, "holdout_game_ids": metadata["holdout_game_ids"][1:] + ["0022499999"]})),
            ("net not offense + defense", lambda: (self.data / "possession_players.csv").write_text(
                csv_text.replace(csv_text.splitlines()[1].rsplit(",", 1)[1], "999.0", 1))),
            ("unreadable metadata", lambda: (self.data / "possession_model.json").write_text("{broken")),
        ):
            with self.subTest(name):
                change()
                html = self.page()
                self.assert_possession_hidden(html)
                self.assertIn("do not match the current possession cache", html)
                self.write_json("possession_model.json", metadata)
                (self.data / "possession_players.csv").write_text(csv_text)


if __name__ == "__main__":
    unittest.main()
