"""Tiny synthetic cache files for gate tests; never a claimed NBA season.

Only use inside temporary test directories. This simulates an audit attestation
so tests can exercise file freshness without downloading or fabricating raw NBA data.
"""
import hashlib
import json

from possessions_loader import _audit_lineage


def attach_lineage_fixture(status, root):
    raw_dir, normalized_dir = root / "pbpstats/pbp", root / "games"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    for gid, record in status["game_results"].items():
        if record["status"] != "validated":
            continue
        raw = raw_dir / f"live_{gid}.json"
        normalized_path = normalized_dir / f"{gid}.json"
        normalized = {"game": {"GameId": gid}, "pbpstats_version": "test fixture",
                      "schema_version": 1, "possessions": [],
                      "applied_corrections": record.get("applied_corrections", [])}
        # Preserve files between display requests so freshness tests are meaningful.
        if not raw.exists():
            raw.write_text(json.dumps({"test_fixture_game_id": gid}))
        if not normalized_path.exists():
            normalized_path.write_text(json.dumps(normalized))
        record["lineage"] = _audit_lineage(
            raw, normalized_path, result=normalized,
            raw_before=hashlib.sha256(raw.read_bytes()).hexdigest(),
        )
    return status
