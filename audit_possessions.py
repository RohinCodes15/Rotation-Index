"""Read-only offline possession audit; no ingestion or estimate writes."""
import argparse
import json
from collections import Counter

import possessions_loader as loader
from possession_model import check_game_inputs


def audit():
    report = loader.audit_cached_season(write_status=False)
    counts = Counter(dict.fromkeys(("fresh", "stale", "missing", "missing_raw", "unverifiable"), 0))
    for gid in report["expected_game_ids"]:
        record = report["game_results"][gid]
        check = check_game_inputs(gid, record, loader.RAW / "pbp", loader.NORMALIZED)
        record["input_state"] = check.category
        record["input_reason"] = check.reason
        counts[check.category] += 1
    report["input_counts"] = dict(sorted(counts.items()))
    # Invalid cached inputs take precedence over an otherwise incomplete season.
    if not report["audit_complete"]:
        code = 3
    elif report["games_failed"] or any(counts[k] for k in ("stale", "missing", "unverifiable")):
        code = 2
    elif report["games_missing"]:
        code = 1
    else:
        code = 0
    return report, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full deterministic audit report")
    args = parser.parse_args(argv)
    report, code = audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"{loader.SEASON} regular season: " + ", ".join(
            f"{name}={report['games_' + name]}" for name in
            ("expected", "cached", "validated", "failed", "missing")))
        print("Inputs: " + ", ".join(f"{k}={v}" for k, v in report["input_counts"].items()))
        print("Possessions: " + ", ".join(f"{k}={report[k]}" for k in
              ("possession_segments", "counted_possessions", "model_eligible_possessions")))
        if report.get("error"):
            print(report["error"])
        for gid, record in sorted(report["game_results"].items()):
            if record.get("error"):
                print(f"{gid}: {record['error']}")
            if record["input_state"] not in ("fresh", "missing_raw"):
                print(f"{gid}: {record['input_state']}: {record['input_reason']}")
            for correction in record.get("applied_corrections", []):
                print(f"{gid} correction: {json.dumps(correction, sort_keys=True)}")
        print(f"Exit {code}: " + {0: "ready", 1: "incomplete cache; available inputs valid",
              2: "invalid, missing normalized, stale, or unverifiable cached inputs",
              3: "audit could not scan the expected season list"}[code])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
