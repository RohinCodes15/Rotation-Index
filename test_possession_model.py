"""Real-cache checks for possession-model input validation."""

import copy
import json
import csv
import tempfile
import hashlib
import unittest
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_cache_fixtures import attach_lineage_fixture

from possession_model import (
    IncompleteSeasonError,
    PossessionDesignData,
    PossessionRidgeFit,
    build_possession_rows,
    build_design_data,
    load_cached_possession_rows,
    require_complete_season,
    fit_possession_ridge,
    _solve_compact_ridge,
    _center_effects,
    _game_holdout,
    _select_ridge_penalty,
    select_ridge_penalty,
    summarize_player_impacts,
    summarize_holdout_predictions,
    net_effect_correlation,
    RIDGE_PENALTY_GRID,
    load_cached_player_names,
)


BASE = Path(__file__).resolve().parent
REAL_GAME = BASE / "data/cache/possessions/games/0022400061.json"
STATUS = BASE / "data/cache/possessions/status.json"
RAW_GAME = BASE / "data/cache/possessions/pbpstats/pbp/live_0022400061.json"


def complete_audit_fixture():
    """Status-only fixture; IDs come from the existing list, not invented games."""
    ids = sorted(g["GameId"] for g in json.loads((STATUS.parent / "games.json").read_text())["results"])
    return {
        "mode": "offline_audit", "audit_complete": True, "complete": True,
        "games_expected": 1230, "games_cached": 1230, "games_validated": 1230,
        "games_failed": 0, "games_missing": 0,
        "expected_game_ids": ids, "cached_game_ids": ids, "validated_game_ids": ids,
        "failed_game_ids": [], "missing_game_ids": [], "validation_errors": {},
        "game_results": {gid: {"status": "validated"} for gid in ids},
        "season": "test fixture",
    }


class HoldoutDiagnosticsTests(unittest.TestCase):
    def test_net_effect_correlation_and_undefined_cases(self):
        fit = PossessionRidgeFit(np.arange(3), 1., np.array([-1., 0., 1.]), np.zeros(3))
        self.assertAlmostEqual(net_effect_correlation(fit, fit), 1.)
        self.assertAlmostEqual(net_effect_correlation(
            fit, replace(fit, offense_effects=np.zeros(3), defense_effects=-2 * fit.offense_effects)), -1.)
        self.assertAlmostEqual(net_effect_correlation(
            fit, replace(fit, offense_effects=np.array([1., -2., 1.]))), 0.)
        self.assertIsNone(net_effect_correlation(fit, replace(fit, offense_effects=np.zeros(3))))
        single = PossessionRidgeFit(np.array([1]), 1., np.array([0.]), np.array([0.]))
        self.assertIsNone(net_effect_correlation(single, single))
        for invalid in (replace(fit, player_ids=fit.player_ids[::-1]),
                        replace(fit, offense_effects=np.array([1.])),
                        replace(fit, defense_effects=np.array([0., float('nan'), 0.]))):
            with self.assertRaises(ValueError):
                net_effect_correlation(fit, invalid)

    def test_penalty_stability_reuses_existing_fits_deterministically(self):
        rng = np.random.default_rng(7)
        lineups = np.array([rng.permutation(10) for _ in range(20)])
        design = PossessionDesignData(
            player_ids=np.arange(10), offense_indices=lineups[:, :5],
            defense_indices=lineups[:, 5:], points_per_possession=rng.integers(0, 4, 20).astype(float),
            game_ids=np.repeat([f'fixture-{i}' for i in range(10)], 2),
        )
        fitted = []
        def capture(data, penalty, **kwargs):
            fit = _solve_compact_ridge(data, penalty, **kwargs)
            fitted.append(fit)
            return fit
        with patch('possession_model._solve_compact_ridge', side_effect=capture) as solver:
            report = _select_ridge_penalty(design, (10, 1), tolerance=1e-10, max_iterations=None)
        self.assertEqual(solver.call_count, 2)
        nets = [fit.offense_effects + fit.defense_effects for fit in fitted]
        self.assertEqual(report.penalty_stability, {
            'min_penalty': 1., 'max_penalty': 10.,
            'net_effect_correlation': float(np.corrcoef(nets)[0, 1]),
        })
        repeated = _select_ridge_penalty(design, (1, 10, 1), tolerance=1e-10, max_iterations=None)
        self.assertEqual(report, repeated)
        single = _select_ridge_penalty(design, (1,), tolerance=1e-10, max_iterations=None)
        self.assertEqual(single.penalty_stability,
                         {'min_penalty': 1., 'max_penalty': 1., 'net_effect_correlation': None})

    def test_training_mean_and_ridge_share_holdout_without_leakage(self):
        games = np.repeat([f'fixture-{i}' for i in range(10)], 2)
        mask, training, holdout = _game_holdout(games)
        outcomes = np.tile([0., 2.], 10)
        outcomes[mask] = np.tile([5., 9.], len(holdout))
        design = PossessionDesignData(
            player_ids=np.arange(10),
            offense_indices=np.tile(np.arange(5), (len(games), 1)),
            defense_indices=np.tile(np.arange(5, 10), (len(games), 1)),
            points_per_possession=outcomes, game_ids=games,
        )
        with patch('possession_model.summarize_holdout_predictions',
                   wraps=summarize_holdout_predictions) as summarize:
            report = _select_ridge_penalty(design, (10, 1), tolerance=1e-10, max_iterations=None)
        self.assertEqual(summarize.call_count, 3)  # Baseline once, then each Ridge candidate.
        for call in summarize.call_args_list:
            np.testing.assert_array_equal(call.args[0], games[mask])
            np.testing.assert_array_equal(call.args[1], outcomes[mask])
        # Training outcomes average 1; including the holdout would give 2.2.
        np.testing.assert_array_equal(summarize.call_args_list[0].args[2], np.ones(mask.sum()))
        expected = {'holdout_games': 2, 'evaluated_possessions': 4,
                    'mae': 6., 'rmse': np.sqrt(40.)}
        self.assertEqual(report.baseline_diagnostics, expected)
        self.assertEqual(report.holdout_diagnostics, expected)
        self.assertEqual(report.training_game_ids, training)
        self.assertEqual(report.holdout_game_ids, holdout)
        for repeated in (design, replace(design, game_ids=games[::-1],
                                        points_per_possession=outcomes[::-1],
                                        offense_indices=design.offense_indices[::-1],
                                        defense_indices=design.defense_indices[::-1])):
            again = _select_ridge_penalty(repeated, (1, 10, 1), tolerance=1e-10, max_iterations=None)
            self.assertEqual(report, again)
            self.assertEqual(json.dumps(report.baseline_diagnostics, sort_keys=True),
                             json.dumps(again.baseline_diagnostics, sort_keys=True))

    def test_known_metrics_weight_possessions_and_ignore_row_order(self):
        games = np.array(['fixture-a', 'fixture-a', 'fixture-b'])
        observed = np.array([0., 2., 4.])
        predicted = np.array([1., 2., 1.])
        expected = {'holdout_games': 2, 'evaluated_possessions': 3,
                    'mae': 4 / 3, 'rmse': np.sqrt(10 / 3)}
        self.assertEqual(summarize_holdout_predictions(games, observed, predicted), expected)
        self.assertEqual(summarize_holdout_predictions(games[::-1], observed[::-1], predicted[::-1]),
                         expected)
        self.assertEqual(summarize_holdout_predictions(['fixture-a'], [2], [2]),
                         {'holdout_games': 1, 'evaluated_possessions': 1, 'mae': 0., 'rmse': 0.})

    def test_invalid_predictions_fail_closed(self):
        for games, observed, predicted in (
                ([], [], []), ([['fixture']], [1], [1]), ([''], [1], [1]),
                ([1], [1], [1]), (['fixture'], [1, 2], [1]),
                (['fixture'], [1], [[1]]), (['fixture'], [float('nan')], [1]),
                (['fixture'], [1], [float('inf')]), (['fixture'], [1e308], [-1e308])):
            with self.subTest(games=games, observed=observed, predicted=predicted):
                with self.assertRaises(ValueError):
                    summarize_holdout_predictions(games, observed, predicted)


class PossessionModelInputTests(unittest.TestCase):
    def test_failure_after_rename_restores_complete_output_set(self):
        from possession_model import _publish_estimate_outputs
        payloads = {'possession_players.csv': 'new csv\n',
                    'possession_model.json': '{"new": true}\n',
                    'possession_diagnostics.json': '{"schema_version": 1}\n'}
        for previous in ('all', 'none', 'estimates only'):
            for failed_name in payloads:
                for failure in (OSError, KeyboardInterrupt):
                    with self.subTest(previous=previous, failed_name=failed_name, failure=failure), \
                            tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        for name in payloads:
                            if previous == 'all' or (previous == 'estimates only'
                                                     and name != 'possession_diagnostics.json'):
                                (root / name).write_bytes(b'old ' + name.encode() + b'\x00\r\n')
                        before = {p.name: p.read_bytes() for p in root.iterdir()}
                        replace_file = os.replace
                        def interrupted_replace(source, destination):
                            replace_file(source, destination)
                            if Path(source).name == failed_name:
                                raise failure('interrupted immediately after rename')
                        with patch('possession_model.os.replace', interrupted_replace):
                            with self.assertRaises(failure):
                                _publish_estimate_outputs(payloads, root)
                        self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_staging_and_publication_failures_preserve_output_set(self):
        from possession_model import _publish_estimate_outputs
        payloads = {'possession_players.csv': 'new csv\n', 'possession_model.json': '{"new": true}\n',
                    'possession_diagnostics.json': '{"schema_version": 1}\n'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for preexisting in (True, False):
                for failure in ('generation', 'diagnostics generation', 'first publish',
                                'second publish', 'third publish'):
                    with self.subTest(preexisting=preexisting, failure=failure):
                        for name in payloads:
                            (root / name).unlink(missing_ok=True)
                            if preexisting:
                                (root / name).write_bytes(b'old estimate\x00\r\n')
                        before = {p.name: p.read_bytes() for p in root.iterdir()}
                        replace_file, write_bytes = os.replace, Path.write_bytes
                        calls = 0
                        def publish(source, destination):
                            nonlocal calls
                            calls += 1
                            if calls == {'first publish': 1, 'second publish': 2, 'third publish': 3}.get(failure):
                                raise OSError('injected publish failure')
                            return replace_file(source, destination)
                        def generate(path, content):
                            if path.name == ('possession_diagnostics.json' if failure == 'diagnostics generation'
                                             else 'possession_model.json'):
                                raise OSError('injected output generation failure')
                            return write_bytes(path, content)
                        generating = failure in ('generation', 'diagnostics generation')
                        target = 'pathlib.Path.write_bytes' if generating else 'possession_model.os.replace'
                        with patch(target, generate if generating else publish):
                            with self.assertRaises(OSError):
                                _publish_estimate_outputs(payloads, root)
                        self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_inputs_changed_during_fitting_are_rejected_before_output(self):
        real = self.real_design()
        status = complete_audit_fixture()
        take = np.arange(1230) % len(real.game_ids)
        design = replace(real, offense_indices=real.offense_indices[take],
                         defense_indices=real.defense_indices[take],
                         points_per_possession=real.points_per_possession[take],
                         game_ids=np.array(status["expected_game_ids"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = attach_lineage_fixture(status, root)
            status_path = root / "status.json"
            status_path.write_text(json.dumps(status))
            target = root / "games" / f"{status['expected_game_ids'][0]}.json"
            def change_input(fit, data):
                target.write_bytes(target.read_bytes() + b"\n")
                return summarize_player_impacts(fit, data)
            with patch("possession_model.summarize_player_impacts", side_effect=change_input):
                with self.assertRaisesRegex(IncompleteSeasonError, "normalized inputs"):
                    fit_possession_ridge(design, status_path=status_path, output_dir=root / "outputs")
            self.assertFalse((root / "outputs").exists())

    def test_missing_stale_and_unverifiable_lineage_blocks_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "status.json"
            status = attach_lineage_fixture(complete_audit_fixture(), root)
            gid = status["expected_game_ids"][0]
            raw = root / "pbpstats/pbp" / f"live_{gid}.json"
            normalized = root / "games" / f"{gid}.json"
            originals = {raw: raw.read_bytes(), normalized: normalized.read_bytes()}
            existing = root / "existing"
            existing.mkdir()
            for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                (existing / name).write_text("preserve existing estimate")
            for case in ("missing lineage", "unverifiable lineage", "missing attestation",
                         "stale raw", "stale normalized", "missing normalized", "missing raw",
                         "stale correction fingerprint", "corrupt normalized", "invalid normalized",
                         "attested invalid schema", "attested invalid possessions",
                         "audit rebuild mismatch", "missing raw validation",
                         *[f"{prefix}_{component}_sha256" for prefix in ("recorded", "replayed")
                           for component in ("possessions", "game_metadata", "parser_version", "normalized_schema_version")],
                         *[f"attested {component}_sha256" for component in
                           ("possessions", "game_metadata", "parser_version", "normalized_schema_version")]):
                with self.subTest(case=case):
                    changed = copy.deepcopy(status)
                    lineage = changed["game_results"][gid]["lineage"]
                    if case == "missing lineage":
                        del changed["game_results"][gid]["lineage"]
                    elif case == "unverifiable lineage":
                        lineage.update(status="unverifiable", verified=False)
                    elif case == "missing attestation":
                        del lineage["verified_inputs"]
                    elif case == "stale correction fingerprint":
                        lineage["recorded_corrections_sha256"] = "0" * 64
                    elif case.endswith("_sha256"):
                        if case.startswith("attested "):
                            del lineage["verified_inputs"][case.split()[1]]
                        else:
                            lineage[case] = "0" * 64
                    elif case == "audit rebuild mismatch":
                        lineage["status"] = "mismatch"
                    elif case == "missing raw validation":
                        lineage.update(status="unverifiable", verified=False)
                    elif case == "corrupt normalized":
                        normalized.write_text("{broken")
                    elif case == "invalid normalized":
                        normalized.write_text("[]")
                    elif case.startswith("attested invalid"):
                        from possessions_loader import _audit_lineage
                        saved = json.loads(originals[normalized])
                        saved["schema_version" if case.endswith("schema") else "possessions"] = 99
                        normalized.write_text(json.dumps(saved))
                        changed["game_results"][gid]["lineage"] = _audit_lineage(
                            raw, normalized, result=saved,
                            raw_before=hashlib.sha256(raw.read_bytes()).hexdigest())
                    elif case == "stale raw":
                        raw.write_bytes(originals[raw] + b"\n")
                    elif case == "stale normalized":
                        normalized.write_bytes(originals[normalized] + b"\n")
                    elif case == "missing raw":
                        raw.unlink()
                    else:
                        normalized.unlink()
                    status_path.write_text(json.dumps(changed))
                    for output in (root / "new", existing):
                        with patch("possession_model._solve_compact_ridge") as solver:
                            with self.assertRaisesRegex(IncompleteSeasonError, "normalized inputs"):
                                fit_possession_ridge(status_path=status_path, output_dir=output)
                            solver.assert_not_called()
                    self.assertFalse((root / "new").exists())
                    for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                        self.assertEqual((existing / name).read_text(), "preserve existing estimate")
                    for path, content in originals.items():
                        path.write_bytes(content)

    def test_audit_gate_accepts_consistent_complete_status_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            status = attach_lineage_fixture(complete_audit_fixture(), path.parent)
            path.write_text(json.dumps(status))
            self.assertEqual(require_complete_season(path), status)
            self.assertFalse((Path(temporary) / "possession_players.csv").exists())
            self.assertFalse((Path(temporary) / "possession_model.json").exists())

    def test_audit_gate_rejects_incomplete_failed_and_inconsistent_statuses(self):
        complete = complete_audit_fixture()
        gid = complete["expected_game_ids"][0]
        cases = {
            "unfinished audit": {"audit_complete": False},
            "missing game": {"complete": False, "games_missing": 1, "missing_game_ids": [gid]},
            "failed game": {"games_failed": 1, "failed_game_ids": [gid], "validation_errors": {gid: "invalid lineup"}},
            "errors despite zero count": {"validation_errors": {gid: "bad score"}},
            "failed outcome despite summary": {"game_results": {**complete["game_results"], gid: {"status": "failed", "error": "bad score"}}},
            "wrong validated IDs": {"validated_game_ids": complete["validated_game_ids"][:-1] + [gid]},
            "old status flags only": {"mode": "legacy", "audit_complete": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "status.json"
            existing = root / "existing"
            existing.mkdir()
            for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                (existing / name).write_text("preserve me")
            for label, changes in cases.items():
                status_path.write_text(json.dumps({**complete, **changes}))
                for output in (root / "new", existing):
                    with self.subTest(case=label, output=output.name):
                        with patch("possession_model._solve_compact_ridge") as solver, patch("possession_model.select_ridge_penalty") as selection:
                            with self.assertRaises(IncompleteSeasonError):
                                fit_possession_ridge(status_path=status_path, output_dir=output)
                            solver.assert_not_called()
                            selection.assert_not_called()
                self.assertFalse((root / "new").exists())
                for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                    self.assertEqual((existing / name).read_text(), "preserve me")

    def test_player_names_come_from_real_raw_cache_without_network_or_edits(self):
        before = hashlib.sha256(RAW_GAME.read_bytes()).digest()
        with patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden")) as network:
            lookup = load_cached_player_names(RAW_GAME.parent)
            network.assert_not_called()
        self.assertEqual(lookup.conflicts, {})
        self.assertEqual(len(lookup.names), 23)
        self.assertEqual(lookup.names[1628369], "J. Tatum")
        self.assertEqual(lookup.names[201143], "A. Horford")
        self.assertEqual(lookup.names[1628436], "L. Kornet")
        self.assertNotIn(0, lookup.names)
        events = json.loads(RAW_GAME.read_text())["game"]["actions"]
        for player, name in lookup.names.items():
            self.assertTrue(any(e.get("personId") == player and name in (e.get("playerNameI"), e.get("playerName")) for e in events))
        self.assertEqual(hashlib.sha256(RAW_GAME.read_bytes()).digest(), before)

    def test_conflicting_feed_names_are_reported_without_choosing_one(self):
        # Change only an in-memory copy of a real event; the cache stays intact.
        raw = json.loads(RAW_GAME.read_text())
        event = copy.deepcopy(next(e for e in raw["game"]["actions"] if e.get("personId") == 1628369))
        event["playerNameI"] = "Conflicting feed label"
        raw["game"]["actions"].append(event)
        with patch.object(Path, "glob", return_value=iter([RAW_GAME])), patch.object(Path, "read_text", return_value=json.dumps(raw)):
            lookup = load_cached_player_names(RAW_GAME.parent)
        self.assertNotIn(1628369, lookup.names)
        self.assertEqual(lookup.conflicts[1628369]["playerNameI"], ("Conflicting feed label", "J. Tatum"))
        self.assertEqual(lookup.names[1628436], "L. Kornet")

    def real_design(self):
        return build_design_data(build_possession_rows(json.loads(REAL_GAME.read_text())))

    def test_pipeline_incomplete_cache_leaves_outputs_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = root / "not-created"
            for output in (absent, root):
                if output == root:
                    for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                        (root / name).write_text("existing output")
                with patch("possession_model.select_ridge_penalty") as selection:
                    with self.assertRaises(IncompleteSeasonError):
                        fit_possession_ridge(status_path=STATUS, output_dir=output)
                    selection.assert_not_called()
            self.assertFalse(absent.exists())
            for name in ("possession_players.csv", "possession_model.json", "possession_diagnostics.json"):
                self.assertEqual((root / name).read_text(), "existing output")

    def test_pipeline_selects_refits_and_writes_separate_outputs(self):
        real = self.real_design()
        # Test-only 1,230-group fixture reusing real cached rows. It is not a
        # real season and all generated outputs live in an isolated temp folder.
        take = np.arange(1230) % len(real.game_ids)
        design = replace(real, offense_indices=real.offense_indices[take],
                         defense_indices=real.defense_indices[take],
                         points_per_possession=real.points_per_possession[take],
                         game_ids=np.array(complete_audit_fixture()["validated_game_ids"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "status.json"
            status.write_text(json.dumps(attach_lineage_fixture(complete_audit_fixture(), root)))
            output = root / "outputs"
            fit = fit_possession_ridge(design, status_path=status, output_dir=output)
            metadata = json.loads((output / "possession_model.json").read_text())
            diagnostics_path = output / 'possession_diagnostics.json'
            diagnostics = json.loads(diagnostics_path.read_text())
            self.assertEqual(diagnostics['schema_version'], 1)
            self.assertEqual(diagnostics['season'], 'test fixture')
            saved_status = json.loads(status.read_text())
            self.assertEqual(diagnostics['audit_sha256'], hashlib.sha256(json.dumps(
                saved_status, sort_keys=True, separators=(',', ':'), allow_nan=False,
            ).encode('utf-8')).hexdigest())
            self.assertEqual(diagnostics['holdout']['holdout_game_ids'], metadata['holdout_game_ids'])
            self.assertEqual(diagnostics['holdout']['training_game_ids'], metadata['training_game_ids'])
            self.assertEqual(diagnostics['model_settings']['selected_penalty'], metadata['selected_penalty'])
            self.assertEqual(diagnostics['model_settings']['candidate_penalties'], list(RIDGE_PENALTY_GRID))
            selection = _select_ridge_penalty(design, RIDGE_PENALTY_GRID,
                                              tolerance=1e-10, max_iterations=None)
            self.assertEqual(diagnostics['metrics']['ridge'], selection.holdout_diagnostics)
            self.assertEqual(diagnostics['metrics']['training_mean_baseline'], selection.baseline_diagnostics)
            self.assertEqual(diagnostics['stability'], selection.penalty_stability)
            previous_diagnostics = diagnostics_path.read_bytes()
            fit_possession_ridge(design, status_path=status, output_dir=output)
            self.assertEqual(diagnostics_path.read_bytes(), previous_diagnostics)
            self.assertNotIn('holdout_diagnostics', metadata)
            self.assertNotIn('baseline_diagnostics', metadata)
            self.assertEqual(len(metadata["holdout_game_ids"]), 246)
            self.assertEqual(len(metadata["training_game_ids"]), 984)
            self.assertEqual(metadata["games_used"], 1230)
            self.assertEqual(metadata["eligible_possessions"], 1230)
            self.assertFalse(metadata["intercept_penalized"])
            selected = metadata["selected_penalty"]
            expected = _solve_compact_ridge(design, selected)
            np.testing.assert_allclose(fit.offense_effects, expected.offense_effects)
            np.testing.assert_allclose(fit.defense_effects, expected.defense_effects)
            self.assertEqual(selected, float(min(metadata["holdout_errors"], key=lambda p: (metadata["holdout_errors"][p], float(p)))))
            summary = summarize_player_impacts(expected, design)
            with (output / "possession_players.csv").open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), len(design.player_ids))
            np.testing.assert_allclose([float(r["net_per_100"]) for r in records], summary.net_per_100)
            np.testing.assert_array_equal([int(r["offense_possessions"]) for r in records], summary.offense_possessions)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"possession_players.csv", "possession_model.json", "possession_diagnostics.json"})
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            for failure in ('serialization', 'diagnostics serialization', 'full fit', 'publication',
                            'diagnostics publication'):
                with self.subTest(failure=failure):
                    original_replace = os.replace
                    calls = 0
                    def fail_second(source, destination):
                        nonlocal calls
                        calls += 1
                        if calls == (3 if failure == 'diagnostics publication' else 2):
                            raise OSError('injected publication failure')
                        return original_replace(source, destination)
                    def fail_full_fit(data, penalty, **kwargs):
                        if len(set(data.game_ids)) == 1230:
                            raise RuntimeError('full fit failed')
                        return _solve_compact_ridge(data, penalty, **kwargs)
                    blocker = {
                        'serialization': patch('possession_model.summarize_player_impacts',
                                               side_effect=ValueError('generation failed')),
                        'diagnostics serialization': patch('possession_model.select_ridge_penalty', return_value=
                            replace(selection, baseline_diagnostics={**selection.baseline_diagnostics,
                                                                     'mae': float('nan')})),
                        'full fit': patch('possession_model._solve_compact_ridge', side_effect=fail_full_fit),
                    }.get(failure, patch('possession_model.os.replace', fail_second))
                    with blocker:
                        with self.assertRaises((ValueError, OSError, RuntimeError)):
                            fit_possession_ridge(design, status_path=status, output_dir=output)
                        calls = 0
                        absent = root / ('failed-' + failure)
                        with self.assertRaises((ValueError, OSError, RuntimeError)):
                            fit_possession_ridge(design, status_path=status, output_dir=absent)
                        self.assertFalse(any(p.is_file() for p in absent.rglob('*')))
                    self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_pipeline_rejects_partial_design_even_with_complete_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "status.json"
            status.write_text(json.dumps(attach_lineage_fixture(complete_audit_fixture(), root)))
            output = root / "outputs"
            with self.assertRaises(IncompleteSeasonError):
                fit_possession_ridge(self.real_design(), status_path=status, output_dir=output)
            self.assertFalse(output.exists())

    def test_game_split_is_deterministic_and_never_splits_a_game(self):
        ids = np.array([f"00224{i:05d}" for i in range(1230) for _ in range(1 + i % 3)])
        mask, training, holdout = _game_holdout(ids)
        self.assertEqual(len(holdout), 246)
        self.assertEqual(len(training), 984)
        self.assertTrue(set(training).isdisjoint(holdout))
        self.assertEqual(set(ids[mask]), set(holdout))
        self.assertEqual(set(ids[~mask]), set(training))
        reverse_mask, reverse_training, reverse_holdout = _game_holdout(ids[::-1])
        np.testing.assert_array_equal(reverse_mask, mask[::-1])
        self.assertEqual((training, holdout), (reverse_training, reverse_holdout))
        with self.assertRaisesRegex(ValueError, "at least two"):
            _game_holdout(np.array(["one-game", "one-game"]))

    def test_penalty_selection_scores_only_holdout_and_fits_only_training(self):
        real = self.real_design()
        # Artificial group labels exercise split mechanics only; never reported
        # as actual additional NBA games or a season estimate.
        design = replace(real, game_ids=np.array([f"test-group-{i % 5}" for i in range(len(real.game_ids))]))
        mask, training, holdout = _game_holdout(design.game_ids)
        original_solver = _solve_compact_ridge
        fitted = []
        def capture(data, penalty, **kwargs):
            self.assertEqual(set(data.game_ids), set(training))
            self.assertTrue(set(data.game_ids).isdisjoint(holdout))
            np.testing.assert_array_equal(data.points_per_possession, design.points_per_possession[~mask])
            fit = original_solver(data, penalty, **kwargs)
            fitted.append((penalty, fit))
            return fit
        with patch("possession_model._solve_compact_ridge", side_effect=capture):
            report = _select_ridge_penalty(design, RIDGE_PENALTY_GRID, tolerance=1e-10, max_iterations=None)
        for penalty, fit in fitted:
            predictions = fit.intercept + fit.offense_effects[design.offense_indices[mask]].sum(axis=1) - fit.defense_effects[design.defense_indices[mask]].sum(axis=1)
            self.assertAlmostEqual(report.holdout_errors[penalty], float(np.mean((design.points_per_possession[mask] - predictions)**2)))
        self.assertEqual(report.selected_penalty, min(report.holdout_errors, key=lambda p: (report.holdout_errors[p], p)))
        self.assertEqual(report.training_game_ids, training)
        self.assertEqual(report.holdout_game_ids, holdout)
        selected_fit = next(fit for penalty, fit in fitted if penalty == report.selected_penalty)
        predictions = (selected_fit.intercept
                       + selected_fit.offense_effects[design.offense_indices[mask]].sum(axis=1)
                       - selected_fit.defense_effects[design.defense_indices[mask]].sum(axis=1))
        residuals = design.points_per_possession[mask] - predictions
        self.assertEqual(report.holdout_diagnostics, {
            'holdout_games': len(holdout), 'evaluated_possessions': int(mask.sum()),
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(report.holdout_errors[report.selected_penalty])),
        })
        for bad_grid in ((), (0,), (float("nan"),)):
            with self.assertRaises(ValueError):
                _select_ridge_penalty(design, bad_grid, tolerance=1e-10, max_iterations=None)

    def test_penalty_selection_requires_full_season_before_solver(self):
        with patch("possession_model._select_ridge_penalty") as selection:
            with self.assertRaises(IncompleteSeasonError):
                select_ridge_penalty(self.real_design(), status_path=STATUS)
            selection.assert_not_called()

    def test_ridge_matches_small_reference_with_unpenalized_intercept(self):
        design = self.real_design()
        n, p = len(design.game_ids), len(design.player_ids)
        # Dense reference is restricted to this one-game test, never the solver.
        x = np.zeros((n, 1 + 2 * p))
        x[:, 0] = 1
        for i in range(n):
            x[i, 1 + design.offense_indices[i]] = 1
            x[i, 1 + p + design.defense_indices[i]] = -1
        penalty = 10.0
        regularizer = np.diag([0.0] + [penalty] * (2 * p))
        reference = np.linalg.solve(x.T @ x + regularizer, x.T @ design.points_per_possession)
        fit = _solve_compact_ridge(design, penalty)
        predictions = fit.intercept + fit.offense_effects[design.offense_indices].sum(axis=1) - fit.defense_effects[design.defense_indices].sum(axis=1)
        np.testing.assert_allclose(predictions, x @ reference, atol=1e-9)
        np.testing.assert_allclose(fit.offense_effects, reference[1:1+p], atol=1e-9)
        np.testing.assert_allclose(fit.defense_effects, reference[1+p:], atol=1e-9)
        self.assertAlmostEqual(fit.offense_effects.mean(), 0.0, places=12)
        self.assertAlmostEqual(fit.defense_effects.mean(), 0.0, places=12)
        self.assertAlmostEqual(float((design.points_per_possession - predictions).mean()), 0.0, places=10)
        # A constant response shift must be absorbed entirely by the intercept.
        shifted = _solve_compact_ridge(replace(design, points_per_possession=design.points_per_possession + 3), penalty)
        self.assertAlmostEqual(shifted.intercept, fit.intercept + 3, places=10)
        np.testing.assert_allclose(shifted.offense_effects, fit.offense_effects, atol=1e-9)
        np.testing.assert_allclose(shifted.defense_effects, fit.defense_effects, atol=1e-9)

    def test_separate_centering_preserves_predictions(self):
        design = self.real_design()
        fit = _solve_compact_ridge(design, 10)
        offense = fit.offense_effects + 2
        defense = fit.defense_effects - 3
        before = fit.intercept + offense[design.offense_indices].sum(axis=1) - defense[design.defense_indices].sum(axis=1)
        intercept, offense, defense = _center_effects(fit.intercept, offense, defense)
        after = intercept + offense[design.offense_indices].sum(axis=1) - defense[design.defense_indices].sum(axis=1)
        np.testing.assert_allclose(after, before, atol=1e-12)
        self.assertAlmostEqual(float(offense.mean()), 0, places=12)
        self.assertAlmostEqual(float(defense.mean()), 0, places=12)

    def test_public_fit_stops_at_full_season_gate(self):
        with patch("possession_model._solve_compact_ridge") as solver:
            with self.assertRaises(IncompleteSeasonError):
                fit_possession_ridge(self.real_design(), 10, status_path=STATUS)
            solver.assert_not_called()

    def test_solver_rejects_bad_penalties_and_nonconvergence(self):
        design = self.real_design()
        for penalty in (0, -1, float("inf"), float("nan")):
            with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                _solve_compact_ridge(design, penalty)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            _solve_compact_ridge(design, 10, max_iterations=1)

    def test_compact_design_data_matches_real_cached_possessions(self):
        rows = build_possession_rows(json.loads(REAL_GAME.read_text()))
        design = build_design_data(rows)
        expected_ids = sorted({p for row in rows for p in row.offense_players + row.defense_players})
        self.assertEqual(design.player_ids.tolist(), expected_ids)
        self.assertEqual(design.offense_indices.shape, (len(rows), 5))
        self.assertEqual(design.defense_indices.shape, (len(rows), 5))
        self.assertEqual(design.offense_indices.dtype, np.dtype("int32"))
        self.assertEqual(design.defense_indices.dtype, np.dtype("int32"))
        self.assertFalse(np.shares_memory(design.offense_indices, design.defense_indices))
        for i, row in enumerate(rows):
            self.assertEqual(design.player_ids[design.offense_indices[i]].tolist(), list(row.offense_players))
            self.assertEqual(design.player_ids[design.defense_indices[i]].tolist(), list(row.defense_players))
        self.assertEqual(design.points_per_possession.tolist(), [row.points for row in rows])
        self.assertEqual(design.game_ids.tolist(), [row.game_id for row in rows])
        self.assertEqual(design.points_per_possession.shape, (len(rows),))
        self.assertEqual(design.game_ids.shape, (len(rows),))
        reversed_design = build_design_data(reversed(rows))
        np.testing.assert_array_equal(reversed_design.player_ids, design.player_ids)
        np.testing.assert_array_equal(reversed_design.offense_indices, design.offense_indices[::-1])
        np.testing.assert_array_equal(reversed_design.defense_indices, design.defense_indices[::-1])
        np.testing.assert_array_equal(reversed_design.points_per_possession, design.points_per_possession[::-1])
        np.testing.assert_array_equal(reversed_design.game_ids, design.game_ids[::-1])

    def test_cached_game_yields_only_eligible_real_possessions(self):
        game = json.loads(REAL_GAME.read_text())
        rows = build_possession_rows(game)
        expected = [p for p in game["possessions"] if p["model_eligible"] and p["counted_possession"]]

        self.assertEqual(len(rows), len(expected))
        self.assertEqual(len(rows), 172)
        for row, source in zip(rows, expected):
            self.assertEqual(row.game_id, "0022400061")
            self.assertEqual(row.points, source["points"])
            self.assertEqual(len(row.offense_players), 5)
            self.assertEqual(len(row.defense_players), 5)
            self.assertTrue(set(row.offense_players).isdisjoint(row.defense_players))

    def test_local_loader_reads_the_cached_game_only(self):
        rows = load_cached_possession_rows(REAL_GAME.parent)
        self.assertEqual(len(rows), 172)
        self.assertEqual({row.game_id for row in rows}, {"0022400061"})

    def test_invalid_five_player_lineup_fails_closed(self):
        game = json.loads(REAL_GAME.read_text())
        changed = copy.deepcopy(game)
        eligible = next(p for p in changed["possessions"] if p["model_eligible"])
        eligible["offense_players"] = eligible["offense_players"][:-1]
        with self.assertRaisesRegex(ValueError, "must have five players"):
            build_possession_rows(changed)

    def test_incomplete_season_cannot_be_used_for_fitting(self):
        with self.assertRaisesRegex(IncompleteSeasonError, "1 of 1230 games validated"):
            require_complete_season(STATUS)


class RidgeSolverRealGameTests(unittest.TestCase):
    """Solver checks on the one real cached game, across penalties; nothing is written."""

    PENALTIES = (1.0, 10.0, 100.0, 1000.0)

    @classmethod
    def setUpClass(cls):
        cls.design = build_design_data(build_possession_rows(json.loads(REAL_GAME.read_text())))

    def dense_reference_predictions(self, penalty):
        # Uncentered dense ridge with an unpenalized intercept; affordable for one game only.
        design = self.design
        n, p = len(design.game_ids), len(design.player_ids)
        x = np.zeros((n, 1 + 2 * p))
        x[:, 0] = 1
        for i in range(n):
            x[i, 1 + design.offense_indices[i]] = 1
            x[i, 1 + p + design.defense_indices[i]] = -1
        regularizer = np.diag([0.0] + [penalty] * (2 * p))
        return x @ np.linalg.solve(x.T @ x + regularizer, x.T @ design.points_per_possession)

    def predictions(self, fit):
        d = self.design
        return (fit.intercept + fit.offense_effects[d.offense_indices].sum(axis=1)
                - fit.defense_effects[d.defense_indices].sum(axis=1))

    def test_offense_and_defense_are_separate_finite_blocks(self):
        p = len(self.design.player_ids)
        for penalty in self.PENALTIES:
            with self.subTest(penalty=penalty):
                fit = _solve_compact_ridge(self.design, penalty)
                self.assertEqual(fit.offense_effects.shape, (p,))
                self.assertEqual(fit.defense_effects.shape, (p,))
                self.assertTrue(np.isfinite(fit.intercept))
                self.assertTrue(np.all(np.isfinite(fit.offense_effects)))
                self.assertTrue(np.all(np.isfinite(fit.defense_effects)))
                self.assertFalse(np.shares_memory(fit.offense_effects, fit.defense_effects))
                # Both blocks carry signal and are not copies or mirrors of each other.
                self.assertGreater(np.abs(fit.offense_effects).max(), 0)
                self.assertGreater(np.abs(fit.defense_effects).max(), 0)
                self.assertFalse(np.allclose(fit.offense_effects, fit.defense_effects))
                self.assertFalse(np.allclose(fit.offense_effects, -fit.defense_effects))
                np.testing.assert_array_equal(fit.player_ids, self.design.player_ids)

    def test_each_block_is_centered(self):
        for penalty in self.PENALTIES:
            with self.subTest(penalty=penalty):
                fit = _solve_compact_ridge(self.design, penalty)
                self.assertAlmostEqual(float(fit.offense_effects.mean()), 0.0, places=12)
                self.assertAlmostEqual(float(fit.defense_effects.mean()), 0.0, places=12)

    def test_centering_preserves_dense_reference_predictions(self):
        for penalty in self.PENALTIES:
            with self.subTest(penalty=penalty):
                fit = _solve_compact_ridge(self.design, penalty)
                predicted = self.predictions(fit)
                self.assertTrue(np.all(np.isfinite(predicted)))
                np.testing.assert_allclose(predicted, self.dense_reference_predictions(penalty), atol=1e-8)
                # The unpenalized intercept leaves zero mean residual.
                self.assertAlmostEqual(float((self.design.points_per_possession - predicted).mean()), 0.0, places=10)

    def test_stronger_penalty_shrinks_both_blocks(self):
        weak = _solve_compact_ridge(self.design, 1.0)
        strong = _solve_compact_ridge(self.design, 1000.0)
        self.assertLess(np.linalg.norm(strong.offense_effects), np.linalg.norm(weak.offense_effects))
        self.assertLess(np.linalg.norm(strong.defense_effects), np.linalg.norm(weak.defense_effects))

    def test_player_summary_scales_counts_and_keeps_defense_as_prevented(self):
        d = self.design
        fit = _solve_compact_ridge(d, 100.0)
        summary = summarize_player_impacts(fit, d)
        np.testing.assert_array_equal(summary.player_ids, d.player_ids)
        for i, player in enumerate(d.player_ids):
            self.assertEqual(summary.offense_possessions[i], int((d.offense_indices == i).sum()), player)
            self.assertEqual(summary.defense_possessions[i], int((d.defense_indices == i).sum()), player)
        self.assertEqual(summary.offense_possessions.sum(), 5 * len(d.game_ids))
        self.assertEqual(summary.defense_possessions.sum(), 5 * len(d.game_ids))
        np.testing.assert_allclose(summary.offense_per_100, 100 * fit.offense_effects)
        # No sign flip: positive defense is points prevented, so it adds to net.
        np.testing.assert_allclose(summary.defense_per_100, 100 * fit.defense_effects)
        np.testing.assert_allclose(summary.net_per_100, summary.offense_per_100 + summary.defense_per_100)
        with self.assertRaisesRegex(ValueError, "player IDs differ"):
            summarize_player_impacts(replace(fit, player_ids=fit.player_ids[::-1].copy()), d)


if __name__ == "__main__":
    unittest.main()
