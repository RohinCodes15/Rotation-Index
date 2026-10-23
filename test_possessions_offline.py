"""Offline audits use temp cache copies; HTTP and socket attempts are counted."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import possessions_loader as pl
import audit_possessions as command

REAL_ROOT = pl.ROOT
GAME_ID = '0022400061'
REAL_RAW = REAL_ROOT / 'pbpstats/pbp' / f'live_{GAME_ID}.json'


class OfflineAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'cache/possessions'  # app.DATA_DIR layout
        self.raw = self.root / 'pbpstats'
        self.normalized = self.root / 'games'
        for name, value in (('ROOT', self.root), ('RAW', self.raw), ('NORMALIZED', self.normalized)):
            mock = patch.object(pl, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.network_mocks = []
        for target in ('requests.sessions.Session.request', 'socket.socket.connect', 'socket.create_connection'):
            mock = patch(target, side_effect=AssertionError('network forbidden during offline audit'))
            self.network_mocks.append(mock.start())
            self.addCleanup(mock.stop)
        self.real_hash = hashlib.sha256(REAL_RAW.read_bytes()).digest()

    def tearDown(self):
        self.assertEqual(sum(mock.call_count for mock in self.network_mocks), 0)
        self.assertEqual(hashlib.sha256(REAL_RAW.read_bytes()).digest(), self.real_hash)

    def copy_cache(self):
        (self.raw / 'pbp').mkdir(parents=True)
        shutil.copyfile(REAL_ROOT / 'games.json', self.root / 'games.json')
        target = self.raw / 'pbp' / REAL_RAW.name
        shutil.copyfile(REAL_RAW, target)
        return target

    def copy_normalized(self):
        self.normalized.mkdir(parents=True, exist_ok=True)
        path = self.normalized / f'{GAME_ID}.json'
        shutil.copyfile(REAL_ROOT / 'games' / path.name, path)
        return path

    def test_manifest_rejects_missing_replay_components_and_correction_identity(self):
        raw = self.copy_cache()
        normalized = self.copy_normalized()
        saved = json.loads(normalized.read_text())
        before = hashlib.sha256(raw.read_bytes()).hexdigest()
        baseline = pl._audit_lineage(raw, normalized, result=saved, raw_before=before)
        self.assertEqual(baseline['schema_version'], pl.LINEAGE_SCHEMA_VERSION)
        self.assertEqual(baseline, pl._audit_lineage(raw, normalized, result=saved, raw_before=before))
        for field in ('possessions', 'game', 'pbpstats_version', 'schema_version', 'applied_corrections'):
            changed = {k: v for k, v in saved.items() if k != field}
            with self.subTest(field=field):
                manifest = pl._audit_lineage(raw, normalized, result=changed, raw_before=before)
                self.assertEqual(manifest['status'], 'unverifiable')
                self.assertFalse(manifest['verified'])
                self.assertNotIn('verified_inputs', manifest)
        for field in ('game_id', 'action_number', 'expected_feed_player1_id', 'corrected_player1_id'):
            changed = json.loads(json.dumps(saved))
            del changed['applied_corrections'][0][field]
            with self.subTest(correction_field=field):
                manifest = pl._audit_lineage(raw, normalized, result=changed, raw_before=before)
                self.assertEqual(manifest['status'], 'unverifiable')
        failed = pl._audit_lineage(raw, normalized)
        self.assertEqual(failed['schema_version'], pl.LINEAGE_SCHEMA_VERSION)
        self.assertEqual(failed['status'], 'unverifiable')

    def readonly_command(self, *args):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        output = io.StringIO()
        with patch.object(pl, 'save', side_effect=AssertionError('read-only audit must not save')):
            with contextlib.redirect_stdout(output):
                code = command.main(list(args))
        self.assertEqual({p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}, before)
        return code, output.getvalue()

    def test_readonly_command_real_opener_and_missing_season(self):
        self.copy_cache()
        self.copy_normalized()
        code, output = self.readonly_command('--json')
        self.assertEqual(code, 1)
        report = json.loads(output)
        self.assertEqual(len(report['game_results']), 1230)
        self.assertEqual([report['games_' + k] for k in ('cached', 'validated', 'failed', 'missing')], [1, 1, 0, 1229])
        self.assertEqual(report['input_counts'], dict(fresh=1, missing=0, missing_raw=1229, stale=0, unverifiable=0))
        self.assertEqual(report['counted_possessions'], 180)
        self.assertEqual(report['model_eligible_possessions'], 172)
        self.assertEqual(report['game_results'][GAME_ID]['applied_corrections'][0]['action_number'], 509)
        self.assertEqual(self.readonly_command('--json'), (code, output))
        summary_code, summary = self.readonly_command()
        self.assertEqual(summary_code, 1)
        for text in ('cached=1', 'missing=1229', 'counted_possessions=180', 'correction:', 'incomplete cache'):
            self.assertIn(text, summary)

    def test_readonly_command_invalid_cached_game_precedes_incomplete(self):
        self.copy_cache().write_text('{bad json')
        code, output = self.readonly_command('--json')
        report = json.loads(output)
        self.assertEqual(code, 2)
        self.assertEqual(report['games_failed'], 1)
        self.assertEqual(report['games_missing'], 1229)
        self.assertIn('JSONDecodeError', report['validation_errors'][GAME_ID])

    def test_readonly_command_stale_and_unverifiable_inputs(self):
        self.copy_cache()
        path = self.copy_normalized()
        original = json.loads(path.read_text())
        changed = {**original, 'pbpstats_version': 'test-only stale version'}
        for content, state in ((json.dumps(changed), 'stale'), ('{bad json', 'unverifiable')):
            with self.subTest(state=state):
                path.write_text(content)
                code, output = self.readonly_command('--json')
                report = json.loads(output)
                self.assertEqual(code, 2)
                self.assertEqual(report['games_validated'], 1)
                self.assertEqual(report['input_counts'][state], 1)
                self.assertEqual(report['input_counts']['missing_raw'], 1229)
                self.assertEqual(report['game_results'][GAME_ID]['input_state'], state)

    def test_readonly_command_unusable_list(self):
        code, output = self.readonly_command('--json')
        self.assertEqual(code, 3)
        self.assertFalse(json.loads(output)['audit_complete'])
        self.assertFalse(self.root.exists())

    def test_matching_real_cache_has_verified_deterministic_lineage(self):
        raw = self.copy_cache()
        normalized = self.copy_normalized()
        before = (raw.read_bytes(), normalized.read_bytes())
        status = self.run_cli()
        lineage = status['game_results'][GAME_ID]['lineage']
        self.assertTrue(lineage['verified'])
        self.assertEqual(lineage['status'], 'verified')
        self.assertEqual(lineage['raw_sha256'], hashlib.sha256(before[0]).hexdigest())
        self.assertEqual(lineage['normalized_sha256'], hashlib.sha256(before[1]).hexdigest())
        self.assertEqual(lineage['recorded_corrections_sha256'], lineage['replayed_corrections_sha256'])
        self.assertEqual(lineage['normalized_content_sha256'], lineage['replayed_normalized_sha256'])
        self.assertEqual(lineage['method'], 'canonical_normalization_replay_v1')
        saved = json.loads(before[1])
        for label, field in (('possessions', 'possessions'), ('game_metadata', 'game'),
                             ('parser_version', 'pbpstats_version'), ('normalized_schema_version', 'schema_version')):
            expected = pl._json_fingerprint(saved[field])
            self.assertEqual(lineage[f'recorded_{label}_sha256'], expected)
            self.assertEqual(lineage[f'replayed_{label}_sha256'], expected)
            self.assertEqual(lineage['verified_inputs'][f'{label}_sha256'], expected)
        self.assertEqual(status['lineage_counts']['verified'], 1)
        self.assertEqual(self.run_cli(), status)
        self.assertEqual((raw.read_bytes(), normalized.read_bytes()), before)

    def test_changed_raw_content_does_not_inherit_old_normalized_lineage(self):
        raw_path = self.copy_cache()
        normalized = self.copy_normalized()
        original_raw = raw_path.read_bytes()
        saved_normalized = normalized.read_bytes()
        raw = json.loads(original_raw)
        event = next(e for e in raw['game']['actions'] if e.get('actionType') == '3pt')
        event['description'] += ' [test-only change]'
        raw_path.write_text(json.dumps(raw))
        status = self.run_cli()
        self.assertEqual(status['games_validated'], 1)  # Raw basketball checks still pass.
        lineage = status['game_results'][GAME_ID]['lineage']
        self.assertEqual(lineage['status'], 'mismatch')
        self.assertFalse(lineage['verified'])
        self.assertNotIn('verified_inputs', lineage)
        self.assertNotEqual(lineage['raw_sha256'], hashlib.sha256(original_raw).hexdigest())
        self.assertEqual(normalized.read_bytes(), saved_normalized)

    def test_changed_normalized_points_or_recorded_corrections_fail_lineage(self):
        self.copy_cache()
        normalized = self.copy_normalized()
        original = json.loads(normalized.read_text())
        for change in ('points', 'corrections'):
            with self.subTest(change=change):
                modified = json.loads(json.dumps(original))
                if change == 'points':
                    modified['possessions'][0]['points'] += 1
                else:
                    modified['applied_corrections'][0]['reason'] += ' [test-only change]'
                normalized.write_text(json.dumps(modified))
                before = normalized.read_bytes()
                status = self.run_cli()
                lineage = status['game_results'][GAME_ID]['lineage']
                self.assertEqual(lineage['status'], 'mismatch')
                self.assertFalse(lineage['verified'])
                self.assertNotIn('verified_inputs', lineage)
                self.assertEqual(normalized.read_bytes(), before)
                if change == 'corrections':
                    self.assertNotEqual(lineage['recorded_corrections_sha256'], lineage['replayed_corrections_sha256'])

    def test_component_changes_are_individually_fingerprinted(self):
        self.copy_cache()
        normalized = self.copy_normalized()
        original = json.loads(normalized.read_text())
        for field, label in (('possessions', 'possessions'), ('game', 'game_metadata'),
                             ('pbpstats_version', 'parser_version'), ('schema_version', 'normalized_schema_version')):
            with self.subTest(field=field):
                saved = json.loads(json.dumps(original))
                if field == 'possessions':
                    saved[field][0]['points'] += 1
                elif field == 'game':
                    saved[field]['Date'] = 'test-only changed metadata'
                elif field == 'pbpstats_version':
                    saved[field] = 'test-only changed parser'
                else:
                    saved[field] += 1
                normalized.write_text(json.dumps(saved))
                lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
                self.assertEqual(lineage['status'], 'mismatch')
                self.assertFalse(lineage['verified'])
                self.assertNotIn('verified_inputs', lineage)
                self.assertNotEqual(lineage[f'recorded_{label}_sha256'], lineage[f'replayed_{label}_sha256'])

    def test_correction_hash_includes_full_fail_closed_identity(self):
        self.copy_cache()
        normalized = self.copy_normalized()
        original = json.loads(normalized.read_text())
        for field in ('game_id', 'action_number', 'expected_feed_player1_id', 'corrected_player1_id'):
            with self.subTest(field=field):
                saved = json.loads(json.dumps(original))
                correction = saved['applied_corrections'][0]
                correction[field] = 'changed identity' if field == 'game_id' else correction[field] + 1
                normalized.write_text(json.dumps(saved))
                lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
                self.assertEqual(lineage['status'], 'mismatch')
                self.assertNotIn('verified_inputs', lineage)
                self.assertNotEqual(lineage['recorded_corrections_sha256'], lineage['replayed_corrections_sha256'])

    def test_current_correction_change_invalidates_old_normalization(self):
        self.copy_cache()
        self.copy_normalized()
        changed = tuple({**rule, 'reason': rule['reason'] + ' [test-only evidence change]'} for rule in pl.VERIFIED_CORRECTIONS)
        with patch.object(pl, 'VERIFIED_CORRECTIONS', changed):
            lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
        self.assertEqual(lineage['status'], 'mismatch')
        self.assertFalse(lineage['verified'])
        self.assertNotEqual(lineage['recorded_corrections_sha256'], lineage['replayed_corrections_sha256'])

    def test_canonical_fingerprint_ignores_dictionary_key_order(self):
        saved = json.loads((REAL_ROOT / 'games' / f'{GAME_ID}.json').read_text())
        reordered = json.loads(json.dumps(saved, sort_keys=True))
        self.assertEqual(pl._json_fingerprint(saved), pl._json_fingerprint(reordered))

    def test_missing_or_invalid_normalized_file_has_no_verified_lineage(self):
        self.copy_cache()
        lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
        self.assertEqual(lineage['status'], 'missing_normalized')
        self.assertIsNone(lineage['normalized_sha256'])
        self.assertNotIn('verified_inputs', lineage)
        normalized = self.copy_normalized()
        normalized.write_text('{bad json')
        lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
        self.assertEqual(lineage['status'], 'invalid_normalized')
        self.assertFalse(lineage['verified'])
        self.assertNotIn('verified_inputs', lineage)

    def test_normalized_formatting_changes_only_its_byte_fingerprint(self):
        self.copy_cache()
        normalized = self.copy_normalized()
        first = self.run_cli()['game_results'][GAME_ID]['lineage']
        normalized.write_text(json.dumps(json.loads(normalized.read_text()), indent=3, sort_keys=True))
        second = self.run_cli()['game_results'][GAME_ID]['lineage']
        self.assertTrue(second['verified'])
        self.assertNotEqual(first['normalized_sha256'], second['normalized_sha256'])
        self.assertEqual(first['normalized_content_sha256'], second['normalized_content_sha256'])

    def test_normalized_file_without_raw_input_is_not_attested(self):
        raw = self.copy_cache()
        normalized = self.copy_normalized()
        raw.unlink()  # Only the test's temporary raw copy is removed.
        before = normalized.read_bytes()
        lineage = self.run_cli()['game_results'][GAME_ID]['lineage']
        self.assertEqual(lineage['status'], 'missing_raw')
        self.assertFalse(lineage['verified'])
        self.assertIsNone(lineage['raw_sha256'])
        self.assertEqual(lineage['normalized_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(lineage['recorded_possessions_sha256'], pl._json_fingerprint(json.loads(before)['possessions']))
        self.assertNotIn('replayed_possessions_sha256', lineage)
        self.assertNotIn('verified_inputs', lineage)
        self.assertEqual(normalized.read_bytes(), before)

    def run_cli(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stop:
                pl.main(offline=True)
        self.assertEqual(stop.exception.code, 1)
        return json.loads((self.root / 'status.json').read_text())

    def test_missing_game_list_records_error_without_downloading(self):
        status = self.run_cli()
        self.assertFalse(status['audit_complete'])
        self.assertFalse(status['complete'])
        self.assertIn('missing cached game list', status['error'])
        self.assertFalse(self.raw.exists())

    def test_all_missing_games_are_taken_from_existing_list(self):
        self.root.mkdir(parents=True)
        shutil.copyfile(REAL_ROOT / 'games.json', self.root / 'games.json')
        expected = sorted(g['GameId'] for g in json.loads((self.root / 'games.json').read_text())['results'])
        status = self.run_cli()
        self.assertEqual(status['expected_game_ids'], expected)
        self.assertEqual(status['missing_game_ids'], expected)
        self.assertEqual(status['games_missing'], len(expected))
        self.assertEqual(status['games_cached'], 0)
        self.assertEqual(status['games_validated'], 0)
        self.assertEqual(status['games_failed'], 0)
        self.assertEqual(status['validation_errors'], {})
        self.assertEqual(status['counted_possessions'], 0)
        self.assertTrue(status['audit_complete'])
        self.assertFalse(status['complete'])
        self.assertFalse(self.raw.exists())

    def test_gaps_do_not_block_real_game_and_status_is_deterministic(self):
        target = self.copy_cache()
        before = target.read_bytes()
        path = self.root / 'games.json'
        games = json.loads(path.read_text())
        # Missing games precede the real game; source order does not matter.
        games['results'] = sorted(games['results'], key=lambda g: g['GameId'] == GAME_ID)
        path.write_text(json.dumps(games))
        status = self.run_cli()
        self.assertTrue(status['audit_complete'])
        self.assertFalse(status['complete'])
        self.assertEqual([status[f'games_{k}'] for k in ('expected', 'cached', 'validated', 'failed', 'missing')], [1230, 1, 1, 0, 1229])
        self.assertEqual(status['validated_game_ids'], [GAME_ID])
        self.assertEqual((status['possession_segments'], status['counted_possessions'], status['model_eligible_possessions']), (182, 180, 172))
        result = status['game_results'][GAME_ID]
        self.assertEqual(result['score_by_team'], {'1610612738': 132, '1610612752': 109})
        self.assertEqual(result['counted_possessions_by_team'], {'1610612738': 90, '1610612752': 90})
        correction = result['applied_corrections'][0]
        self.assertEqual((correction['action_number'], correction['expected_feed_player1_id'], correction['corrected_player1_id']), (509, 201143, 1628436))
        first = (self.root / 'status.json').read_bytes()
        # Reverse the game list, then verify identical report bytes.
        games['results'].reverse()
        path.write_text(json.dumps(games))
        self.assertEqual(self.run_cli(), status)
        self.assertEqual((self.root / 'status.json').read_bytes(), first)
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(self.normalized.exists())

    def test_invalid_game_files_record_errors_and_continue(self):
        self.copy_cache()
        ids = sorted(g['GameId'] for g in json.loads((self.root / 'games.json').read_text())['results'])
        invalid = [gid for gid in ids if gid < GAME_ID][:2]
        self.assertEqual(len(invalid), 2)
        (self.raw / 'pbp' / f'live_{invalid[0]}.json').write_text('{invalid json')
        (self.raw / 'pbp' / f'live_{invalid[1]}.json').write_text('{"game":{}}')
        before = {p.name: p.read_bytes() for p in (self.raw / 'pbp').iterdir()}
        status = self.run_cli()
        self.assertEqual([status[f'games_{k}'] for k in ('cached', 'validated', 'failed', 'missing')], [3, 1, 2, 1227])
        self.assertEqual(status['failed_game_ids'], invalid)
        expected = set(ids)
        cached = set(status['cached_game_ids'])
        validated = set(status['validated_game_ids'])
        failed = set(status['failed_game_ids'])
        missing = set(status['missing_game_ids'])
        self.assertEqual(cached | missing, expected)
        self.assertTrue(cached.isdisjoint(missing))
        self.assertEqual(validated | failed, cached)
        self.assertTrue(validated.isdisjoint(failed))
        self.assertEqual(set(status['validation_errors']), set(invalid))
        self.assertIn('JSONDecodeError', status['validation_errors'][invalid[0]])
        self.assertEqual(status['game_results'][GAME_ID]['status'], 'validated')
        self.assertEqual({p.name: p.read_bytes() for p in (self.raw / 'pbp').iterdir()}, before)

    def test_changed_substitution_guard_fails_without_repairing_raw(self):
        target = self.copy_cache()
        raw = json.loads(target.read_text())
        next(e for e in raw['game']['actions'] if e['actionNumber'] == 509)['personId'] = 1628436
        target.write_text(json.dumps(raw))
        before = target.read_bytes()
        status = self.run_cli()
        self.assertEqual(status['games_failed'], 1)
        self.assertEqual(status['games_validated'], 0)
        self.assertIn('correction guard failed', status['validation_errors'][GAME_ID])
        self.assertEqual(status['game_results'][GAME_ID]['lineage']['status'], 'unverifiable')
        self.assertNotIn('verified_inputs', status['game_results'][GAME_ID]['lineage'])
        self.assertEqual(target.read_bytes(), before)

    def test_five_player_check_still_rejects_uncorrected_real_game(self):
        self.copy_cache()
        # Test-only removal of the correction must expose the original failure.
        with patch.object(pl, 'VERIFIED_CORRECTIONS', ()):
            status = self.run_cli()
        self.assertEqual(status['games_failed'], 1)
        self.assertIn('invalid five-player lineup', status['validation_errors'][GAME_ID])

    def tree(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def assert_rebuild_failed(self, expected_error):
        """The rebuild fails and leaves every cached file, including the old normalized one, as it was."""
        before = self.tree()
        report = pl.rebuild_cached_game(GAME_ID)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['written'])
        self.assertIn(expected_error, report['error'])
        self.assertEqual(self.tree(), before)

    def assert_opener_counts(self, report):
        self.assertEqual((report['possession_segments'], report['counted_possessions'],
                          report['model_eligible_possessions']), (182, 180, 172))
        self.assertEqual(report['score_by_team'], {'1610612738': 132, '1610612752': 109})
        correction = report['applied_corrections'][0]
        self.assertEqual((correction['action_number'], correction['expected_feed_player1_id'],
                          correction['corrected_player1_id'], correction['corrected_player_name']),
                         (509, 201143, 1628436, 'Luke Kornet'))

    def test_rebuild_dry_run_writes_nothing(self):
        self.copy_cache()
        before = self.tree()
        report = pl.rebuild_cached_game(GAME_ID, dry_run=True)
        self.assertEqual((report['status'], report['written']), ('would_create', False))
        self.assert_opener_counts(report)
        self.assertEqual(self.tree(), before)
        self.assertFalse(self.normalized.exists())
        path = self.copy_normalized()
        path.write_text('{bad json')
        before = self.tree()
        report = pl.rebuild_cached_game(GAME_ID, dry_run=True)
        self.assertEqual((report['status'], report['previous_lineage_status']), ('would_replace', 'invalid_normalized'))
        self.assertEqual(self.tree(), before)

    def test_rebuild_is_deterministic_and_reproduces_cached_file(self):
        raw = self.copy_cache()
        raw_bytes = raw.read_bytes()
        target = self.normalized / f'{GAME_ID}.json'
        report = pl.rebuild_cached_game(GAME_ID)
        self.assertEqual((report['status'], report['written'], report['previous_lineage_status']),
                         ('rebuilt', True, 'missing_normalized'))
        self.assert_opener_counts(report)
        self.assertEqual(target.read_bytes(), (REAL_ROOT / 'games' / target.name).read_bytes())
        lineage = report['lineage']
        self.assertEqual((lineage['status'], lineage['normalized_sha256']),
                         ('verified', hashlib.sha256(target.read_bytes()).hexdigest()))
        first = target.read_bytes()
        again = pl.rebuild_cached_game(GAME_ID)
        self.assertEqual((again['status'], again['written']), ('unchanged', False))
        self.assertEqual(again['lineage']['verified_inputs'], lineage['verified_inputs'])
        target.write_text('{"stale": true}')
        self.assertEqual(pl.rebuild_cached_game(GAME_ID)['status'], 'rebuilt')
        self.assertEqual(target.read_bytes(), first)
        self.assertEqual(raw.read_bytes(), raw_bytes)
        self.assertEqual(sorted(p.name for p in self.normalized.iterdir()), [target.name])

    def test_rebuild_correction_fails_closed_and_keeps_old_file(self):
        raw = self.copy_cache()
        self.copy_normalized()
        original = raw.read_bytes()
        feed = json.loads(original)
        next(e for e in feed['game']['actions'] if e['actionNumber'] == 509)['personId'] = 1628436
        raw.write_text(json.dumps(feed))
        self.assert_rebuild_failed('correction guard failed')
        raw.write_bytes(original)
        correction = pl.VERIFIED_CORRECTIONS[0]
        for label, changed, error in (
                ('wrong action number', {**correction, 'action_number': 99999}, 'matched 0 events'),
                ('wrong expected player', {**correction, 'expected_player1_id': 1628436}, 'correction guard failed'),
                # Not applied to this game, so the bad feed lineup reaches the five-player check.
                ('other game', {**correction, 'game_id': '0022400062'}, 'invalid five-player lineup')):
            with self.subTest(label), patch.object(pl, 'VERIFIED_CORRECTIONS', (changed,)):
                self.assert_rebuild_failed(error)

    def test_rebuild_rejects_malformed_raw_and_waits_on_missing_games(self):
        raw = self.copy_cache()
        self.copy_normalized()
        raw.write_text('{bad json')
        self.assert_rebuild_failed('JSONDecodeError')
        raw.write_text('{"game": {}}')
        self.assert_rebuild_failed('KeyError')
        missing = next(g['GameId'] for g in json.loads((self.root / 'games.json').read_text())['results']
                       if g['GameId'] != GAME_ID)
        before = self.tree()
        report = pl.rebuild_cached_game(missing)
        self.assertEqual((report['status'], report['written']), ('missing_raw', False))
        self.assertEqual(self.tree(), before)
        with self.assertRaises(ValueError):
            pl.rebuild_cached_game('0022499999')
        self.assertEqual(self.tree(), before)

    def test_failed_write_preserves_old_normalized_file(self):
        self.copy_cache()
        self.copy_normalized().write_text('{"old": true}')
        with patch.object(Path, 'replace', side_effect=OSError('disk full')):
            self.assert_rebuild_failed('disk full')
        with patch.object(pl, '_audit_lineage', side_effect=[
                {'status': 'mismatch'}, {'status': 'mismatch'}]):
            self.assert_rebuild_failed('staged file did not verify')

    def test_rebuilt_inputs_pass_dashboard_and_gate_only_after_reaudit(self):
        import app as dashboard
        import possession_model

        self.copy_cache()
        stale = self.copy_normalized()
        stale.write_text(json.dumps({**json.loads(stale.read_text()), 'pbpstats_version': 'old'}))
        status_path = self.root / 'status.json'

        def both_views():
            status = json.loads(status_path.read_text())
            status['expected_game_ids'] = [GAME_ID]  # judge the cached game only
            dashboard._file_cache.clear()
            with patch.object(dashboard, 'DATA_DIR', Path(self.temp.name)):
                view = dashboard.input_freshness(status, lambda gid, fields: {**fields, 'game_id': gid})
            try:
                possession_model._require_current_lineage(status, status_path)
                gate = 'fresh'
            except possession_model.IncompleteSeasonError as error:
                gate = error
            return view, gate

        self.run_cli()
        view, gate = both_views()
        self.assertEqual(view['problems'][0]['category'], 'stale')
        self.assertIn('(stale: ', str(gate))
        audited = status_path.read_bytes()
        self.assertEqual(pl.rebuild_cached_game(GAME_ID)['status'], 'rebuilt')
        self.assertEqual(status_path.read_bytes(), audited)  # rebuild never rewrites the audit
        view, gate = both_views()
        self.assertEqual(view['problems'][0]['category'], 'stale')
        self.assertIn('(stale: ', str(gate))
        self.run_cli()
        view, gate = both_views()
        self.assertTrue(view['fresh'])
        self.assertEqual((view['counts']['fresh'], view['problems']), (1, []))
        self.assertEqual(gate, 'fresh')


if __name__ == '__main__':
    unittest.main(verbosity=2)
