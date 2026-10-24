"""Read-only storage estimates with temporary caches and mocked free space."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import possessions_loader as pl
from test_possessions_download import IDS


class StoragePreflightTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.root = self.directory / "cache/possessions"
        self.raw = self.root / "pbpstats/pbp"
        self.normalized = self.root / "games"
        self.raw.mkdir(parents=True)
        self.normalized.mkdir()
        (self.root / "games.json").write_text(json.dumps({"results": [{"GameId": i} for i in IDS]}))
        for name, value in (("ROOT", self.root), ("RAW", self.raw.parent), ("NORMALIZED", self.normalized)):
            self.enterContext(patch.object(pl, name, value))
        self.disk = self.enterContext(patch.object(pl.shutil, "disk_usage", return_value=SimpleNamespace(free=10**9)))
        for target in ("requests.get", "socket.socket.connect", "socket.create_connection",
                       "possessions_loader.download_missing_games", "possessions_loader.normalize",
                       "possessions_loader.audit_cached_season", "possessions_loader.rebuild_cached_season",
                       "possessions_loader.rebuild_cached_game", "possessions_loader.main"):
            self.enterContext(patch(target, side_effect=AssertionError(f"preflight must not call {target}")))

    def snapshot(self):
        return {p.relative_to(self.directory): p.read_bytes()
                for p in self.directory.rglob("*") if p.is_file()}

    def samples(self):
        for game_id, raw_size, normalized_size in ((IDS[0], 100, 30), (IDS[1], 200, 50)):
            (self.raw / f"live_{game_id}.json").write_bytes(b"r" * raw_size)
            (self.normalized / f"{game_id}.json").write_bytes(b"n" * normalized_size)

    def test_estimate_uses_largest_samples_and_counts_missing_files_separately(self):
        self.samples()
        (self.raw / f"live_{IDS[2]}.json").write_bytes(b"r" * 150)
        before = self.snapshot()
        report = pl.storage_preflight()
        self.assertEqual(report, {
            "available_bytes": 10**9, "missing_games": 1227, "missing_normalized_games": 1228,
            "raw_sample_games": 3, "normalized_sample_games": 2,
            "raw_max_bytes": 200, "normalized_max_bytes": 50,
            "estimated_required_bytes": 1227 * 200 + 1228 * 50,
            "safety_required_bytes": 2 * (1227 * 200 + 1228 * 50),
            "passes_safety_margin": True, "reason": None,
        })
        self.disk.assert_called_once_with(self.root)
        self.assertEqual(pl.storage_preflight(), report)
        self.assertEqual(self.snapshot(), before)

    def test_cli_safety_boundary_and_failed_check_are_read_only(self):
        self.samples()
        before = self.snapshot()
        required = 2 * (1228 * 250)
        for available, code in ((required, 0), (required - 1, 1)):
            with self.subTest(available=available):
                self.disk.return_value = SimpleNamespace(free=available)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as result:
                    pl.cli(["--storage-preflight"])
                report = json.loads(output.getvalue())
                self.assertEqual(result.exception.code, code)
                self.assertEqual(report["passes_safety_margin"], code == 0)
                self.assertEqual(report["safety_required_bytes"], required)
                self.assertEqual(report["available_bytes"], available)
                self.assertEqual(self.snapshot(), before)

    def test_missing_or_empty_samples_cannot_pass(self):
        for kind in ("none", "raw_only", "empty_normalized"):
            with self.subTest(kind=kind):
                if kind != "none":
                    (self.raw / f"live_{IDS[0]}.json").write_bytes(b"r" * 100)
                if kind == "empty_normalized":
                    (self.normalized / f"{IDS[0]}.json").write_bytes(b"")
                before = self.snapshot()
                report = pl.storage_preflight()
                self.assertIsNone(report["estimated_required_bytes"])
                self.assertIsNone(report["safety_required_bytes"])
                self.assertFalse(report["passes_safety_margin"])
                self.assertIn("samples", report["reason"])
                self.assertEqual(self.snapshot(), before)

    def test_complete_cache_needs_no_additional_space(self):
        for game_id in IDS:
            (self.raw / f"live_{game_id}.json").write_bytes(b"r")
            (self.normalized / f"{game_id}.json").write_bytes(b"n")
        self.disk.return_value = SimpleNamespace(free=0)
        before = self.snapshot()
        report = pl.storage_preflight()
        self.assertEqual(report["missing_games"], 0)
        self.assertEqual(report["missing_normalized_games"], 0)
        self.assertEqual(report["estimated_required_bytes"], 0)
        self.assertTrue(report["passes_safety_margin"])
        self.assertEqual(self.snapshot(), before)

    def test_bad_or_missing_game_list_fails_without_fallback(self):
        path = self.root / "games.json"
        for content in ('{', '{"results": []}', '{}'):
            with self.subTest(content=content):
                path.write_text(content)
                before = self.snapshot()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    pl.cli(["--storage-preflight"])
                self.assertEqual(result.exception.code, 2)
                self.assertEqual(self.snapshot(), before)
        path.unlink()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            pl.cli(["--storage-preflight"])
        self.assertEqual(result.exception.code, 2)
        self.assertFalse(path.exists())

    def test_preflight_cannot_be_combined_with_processing_flags(self):
        before = self.snapshot()
        for flags in (("--offline",), ("--rebuild-all",), ("--dry-run",), ("--rebuild", IDS[0])):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as result:
                pl.cli(["--storage-preflight", *flags])
            self.assertEqual(result.exception.code, 2)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
