"""Online download path with mocked HTTP only. Run: python -m unittest test_possessions_download."""

import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import requests

import possessions_loader as pl

IDS = [f"00224{n:05d}" for n in range(1, 1231)]  # synthetic, not real game IDs


def feed(game_id, periods=4):
    return {"game": {"gameId": game_id,
                     "actions": [{"actionNumber": p, "period": p} for p in range(1, periods + 1)]}}


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code, self.body, self.text = status, body, text

    def json(self):
        return json.loads(self.text) if self.text is not None else self.body


class DownloadMissingGamesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "cache/possessions"
        self.pbp = self.root / "pbpstats/pbp"
        for name, value in (("ROOT", self.root), ("RAW", self.root / "pbpstats"),
                            ("NORMALIZED", self.root / "games")):
            mock = patch.object(pl, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        for target in ("requests.sessions.Session.request", "socket.socket.connect", "socket.create_connection"):
            mock = patch(target, side_effect=AssertionError("real network forbidden"))
            mock.start()
            self.addCleanup(mock.stop)
        self.sleeps = []
        mock = patch.object(pl.time, "sleep", side_effect=self.sleeps.append)
        mock.start()
        self.addCleanup(mock.stop)
        # Downloading must never normalize, audit, rebuild, or fit anything.
        for name in ("normalize", "audit_cached_season", "rebuild_cached_season", "rebuild_cached_game"):
            mock = patch.object(pl, name, side_effect=AssertionError(f"{name} must not run while downloading"))
            mock.start()
            self.addCleanup(mock.stop)
        self.calls = Counter()
        self.scripts = {}
        mock = patch.object(pl.requests, "get", side_effect=self.get)
        mock.start()
        self.addCleanup(mock.stop)

    def get(self, url, **kwargs):
        self.assertEqual(kwargs.get("timeout"), 30)
        if url == pl.GAME_LIST_URL:
            self.calls["games.json"] += 1
            return self.game_list
        game_id = url.rsplit("_", 1)[1].removesuffix(".json")
        self.assertEqual(url, pl.LIVE_PBP_URL.format(game_id))
        self.calls[game_id] += 1
        script = self.scripts.get(game_id)
        # The last scripted outcome repeats, so a one-item script fails forever.
        outcome = script[min(self.calls[game_id], len(script)) - 1] if script else FakeResponse(body=feed(game_id))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def write_game_list(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "games.json").write_text(json.dumps({"results": [{"GameId": i} for i in IDS]}))

    def report(self):
        return json.loads((self.root / "download_report.json").read_text())

    def test_downloads_only_missing_games_and_never_replaces_cached_files(self):
        self.write_game_list()
        self.pbp.mkdir(parents=True)
        cached = {IDS[0]: json.dumps(feed(IDS[0]), indent=2).encode(),
                  IDS[5]: (json.dumps(feed(IDS[5])) + "\n").encode()}
        for game_id, content in cached.items():
            (self.pbp / f"live_{game_id}.json").write_bytes(content)

        report = pl.download_missing_games()

        for game_id, content in cached.items():
            self.assertEqual((self.pbp / f"live_{game_id}.json").read_bytes(), content)
            self.assertNotIn(game_id, self.calls)
        missing = [i for i in IDS if i not in cached]
        self.assertEqual(report, {"season": "2024-25", "games_listed": 1230, "already_cached": 2,
                                  "downloaded_game_ids": missing, "failed_game_ids": [], "failures": {}})
        self.assertEqual(self.report(), report)
        self.assertEqual(json.loads((self.pbp / f"live_{IDS[1]}.json").read_text()), feed(IDS[1]))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["download_report.json", "games.json", "pbpstats"])  # no status.json or games/
        self.assertEqual(list(self.pbp.glob("*.tmp")), [])
        self.assertEqual(self.calls["games.json"], 0)

        # A second run has nothing left to fetch.
        self.calls.clear()
        report = pl.download_missing_games()
        self.assertEqual((report["already_cached"], report["downloaded_game_ids"]), (1230, []))
        self.assertEqual(sum(self.calls.values()), 0)

    def test_resume_retries_only_unresolved_games_and_preserves_download_only_files(self):
        self.write_game_list()
        self.pbp.mkdir(parents=True)
        failed, invalid, later = IDS[1:4]
        unresolved = [failed, invalid]
        cached = {}
        for game_id in IDS:
            if game_id not in (failed, invalid, later):
                path = self.pbp / f"live_{game_id}.json"
                cached[path] = (json.dumps(feed(game_id), indent=2) + "\n").encode()
                path.write_bytes(cached[path])
        self.scripts = {failed: [requests.Timeout()], invalid: [FakeResponse(text='{"game":')]}

        def assert_download_only_files(raw_ids):
            directory = self.root.parent.parent
            expected = {self.root / "games.json", self.root / "download_report.json"}
            expected.update(self.pbp / f"live_{game_id}.json" for game_id in raw_ids)
            self.assertEqual({p for p in directory.rglob("*") if p.is_file()}, expected)

        report = pl.download_missing_games()
        self.assertEqual(self.calls, Counter({failed: pl.MAX_ATTEMPTS, invalid: 1, later: 1}))
        self.assertEqual(report["already_cached"], len(cached))
        self.assertEqual(report["downloaded_game_ids"], [later])
        self.assertEqual(report["failed_game_ids"], unresolved)
        self.assertEqual(report["failures"], {
            failed: f"timeout_after_{pl.MAX_ATTEMPTS}_attempts", invalid: "invalid_json"})
        self.assertEqual(self.report(), report)
        self.assertEqual(json.loads((self.pbp / f"live_{later}.json").read_text()), feed(later))
        for game_id in unresolved:
            self.assertFalse((self.pbp / f"live_{game_id}.json").exists())
        for path, content in cached.items():
            self.assertEqual(path.read_bytes(), content)
        assert_download_only_files([i for i in IDS if i not in unresolved])

        saved = {path: path.read_bytes() for path in self.pbp.iterdir()}
        self.calls.clear()
        self.scripts.clear()
        report = pl.download_missing_games()
        self.assertEqual(self.calls, Counter({game_id: 1 for game_id in unresolved}))
        self.assertEqual(report["already_cached"], len(saved))
        self.assertEqual(report["downloaded_game_ids"], unresolved)
        self.assertEqual(report["failed_game_ids"], [])
        self.assertEqual(report["failures"], {})
        self.assertEqual(self.report(), report)
        for game_id in unresolved:
            self.assertEqual(json.loads((self.pbp / f"live_{game_id}.json").read_text()), feed(game_id))
        for path, content in saved.items():
            self.assertEqual(path.read_bytes(), content)
        assert_download_only_files(IDS)

    def test_transient_failures_retry_with_bounded_exponential_backoff(self):
        self.write_game_list()
        recovers, server_error, times_out, refused = IDS[10], IDS[11], IDS[12], IDS[13]
        self.scripts = {
            recovers: [requests.Timeout(), requests.ConnectionError(), FakeResponse(429), FakeResponse(500),
                       FakeResponse(body=feed(recovers))],
            server_error: [FakeResponse(502), FakeResponse(503), FakeResponse(504)],
            times_out: [requests.ReadTimeout()],
            refused: [requests.ConnectionError()],
        }

        report = pl.download_missing_games()

        self.assertEqual(self.calls[recovers], 5)
        self.assertIn(recovers, report["downloaded_game_ids"])
        self.assertEqual(json.loads((self.pbp / f"live_{recovers}.json").read_text()), feed(recovers))
        for game_id in (server_error, times_out, refused):
            self.assertEqual(self.calls[game_id], pl.MAX_ATTEMPTS)
            self.assertFalse((self.pbp / f"live_{game_id}.json").exists())
        self.assertEqual(report["failures"], {server_error: "http_504_after_5_attempts",
                                              times_out: "timeout_after_5_attempts",
                                              refused: "connection_error_after_5_attempts"})
        self.assertEqual(report["failed_game_ids"], [server_error, times_out, refused])
        self.assertEqual(len(report["downloaded_game_ids"]), 1230 - 3)  # later games continued
        backoff = [s for s in self.sleeps if s != 0.15]
        self.assertEqual(backoff, [1.0, 2.0, 4.0, 8.0] * 4)  # no wait after the final attempt

    def assert_retry_recovers(self, first_outcome):
        game_id = IDS[0]
        expected = feed(game_id)
        self.scripts[game_id] = [first_outcome, FakeResponse(body=expected),
                                 AssertionError("request made after success")]
        results = []
        for run in range(2):
            with self.subTest(run=run):
                self.calls.clear()
                self.sleeps.clear()
                results.append(pl.fetch_json(pl.LIVE_PBP_URL.format(game_id)))
                self.assertEqual(self.calls, Counter({game_id: 2}))
                self.assertEqual(self.sleeps, [1.0])
        self.assertEqual(results, [expected, expected])

    def test_temporary_timeout_then_success(self):
        self.assert_retry_recovers(requests.Timeout("temporary timeout"))

    def test_http_503_then_success(self):
        self.assert_retry_recovers(FakeResponse(503))

    def test_http_429_then_success(self):
        self.assert_retry_recovers(FakeResponse(429))

    def test_persistent_transient_failure_stops_at_retry_limit(self):
        game_id = IDS[0]
        for outcome, reason in ((requests.Timeout("persistent timeout"), "timeout"),
                                (FakeResponse(503), "http_503"),
                                (FakeResponse(429), "http_429")):
            errors = []
            self.scripts[game_id] = [outcome] * pl.MAX_ATTEMPTS + [
                AssertionError("retry limit exceeded")]
            for run in range(2):
                with self.subTest(reason=reason, run=run):
                    self.calls.clear()
                    self.sleeps.clear()
                    with self.assertRaises(pl.DownloadError) as error:
                        pl.fetch_json(pl.LIVE_PBP_URL.format(game_id))
                    errors.append(str(error.exception))
                    self.assertEqual(self.calls, Counter({game_id: pl.MAX_ATTEMPTS}))
                    self.assertEqual(self.sleeps, [1.0, 2.0, 4.0, 8.0])
            self.assertEqual(errors, [f"{reason}_after_{pl.MAX_ATTEMPTS}_attempts"] * 2)

    def test_bad_or_incomplete_responses_are_not_saved_or_retried(self):
        self.write_game_list()
        cases = {
            IDS[20]: (FakeResponse(404), "http_404"),
            IDS[21]: (FakeResponse(403), "http_403"),
            IDS[22]: (FakeResponse(text='{"game": {"gameId": "'), "invalid_json"),
            IDS[23]: (FakeResponse(body=feed(IDS[24])), "wrong_or_missing_game_id"),
            IDS[24]: (FakeResponse(body={"results": []}), "wrong_or_missing_game_id"),
            IDS[25]: (FakeResponse(body={"game": {"gameId": IDS[25], "actions": []}}), "missing_actions"),
            IDS[26]: (FakeResponse(body=feed(IDS[26], periods=3)), "incomplete_game"),
            IDS[27]: (FakeResponse(body={"game": {"gameId": IDS[27], "actions": [{"period": "4"}]}}),
                      "incomplete_game"),
            IDS[28]: (requests.TooManyRedirects(), "request_error_TooManyRedirects"),
        }
        self.scripts = {game_id: [outcome] for game_id, (outcome, _) in cases.items()}

        report = pl.download_missing_games()

        self.assertEqual(report["failures"], {game_id: reason for game_id, (_, reason) in cases.items()})
        for game_id in cases:
            self.assertEqual(self.calls[game_id], 1)
            self.assertFalse((self.pbp / f"live_{game_id}.json").exists())
        self.assertEqual(len(report["downloaded_game_ids"]), 1230 - len(cases))
        self.assertEqual([s for s in self.sleeps if s != 0.15], [])

    def test_failed_write_keeps_no_partial_file_and_later_games_continue(self):
        self.write_game_list()
        real_save, broken = pl.save, IDS[30]

        def save(path, data, **kwargs):
            if path.name == f"live_{broken}.json":
                raise OSError("disk full")
            return real_save(path, data, **kwargs)

        with patch.object(pl, "save", side_effect=save):
            report = pl.download_missing_games()
        self.assertEqual(report["failures"], {broken: "write_error_OSError"})
        self.assertFalse((self.pbp / f"live_{broken}.json").exists())
        self.assertIn(IDS[31], report["downloaded_game_ids"])

    def test_game_list_is_validated_before_it_is_cached(self):
        self.game_list = FakeResponse(body={"results": [{"GameId": i} for i in IDS[:-1]]})
        with self.assertRaisesRegex(ValueError, "expected 1,230 distinct"):
            pl.download_missing_games()
        self.assertFalse((self.root / "games.json").exists())
        self.assertEqual(set(self.calls), {"games.json"})

        self.game_list = FakeResponse(503)
        with self.assertRaisesRegex(pl.DownloadError, "http_503_after_5_attempts"):
            pl.download_missing_games()
        self.assertFalse((self.root / "games.json").exists())

        self.game_list = FakeResponse(body={"results": [{"GameId": i} for i in IDS]})
        report = pl.download_missing_games()
        self.assertEqual(len(report["downloaded_game_ids"]), 1230)
        self.assertEqual(json.loads((self.root / "games.json").read_text()), self.game_list.body)

    def test_online_command_reports_and_exits_without_auditing(self):
        self.write_game_list()
        for scripts, code in (({IDS[0]: [FakeResponse(404)]}, 1), ({}, 0)):
            with self.subTest(code=code):
                self.scripts = scripts
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                        self.assertRaises(SystemExit) as exit_:
                    pl.cli([])
                self.assertEqual(exit_.exception.code, code)
                self.assertEqual(json.loads(stdout.getvalue()), self.report())
                self.assertIn("Nothing was normalized or audited", stderr.getvalue())
                self.assertFalse((self.root / "status.json").exists())
                self.assertFalse((self.root / "verification.json").exists())
                self.assertFalse((self.root / "games").exists())


if __name__ == "__main__":
    unittest.main()
