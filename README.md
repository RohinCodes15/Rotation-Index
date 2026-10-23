# Rotation-Index
NBA lineup impact analysis using adjusted plus-minus to estimate player contributions while accounting for teammates and opponents.

## Phase 1: what it does

1. `possessions_loader.py` caches real 2024-25 regular-season possessions and both on-court lineups
   from NBA live play-by-play, parsed with `pbpstats`, and validates each game.
2. `possession_model.py` selects a ridge penalty by game holdout, fits separate offense
   and defense effects, and writes summaries only after the full-season audit gate passes.
3. `data_loader.py` is the legacy lineup-total loader using the public
   [pbpstats.com API](https://api.pbpstats.com). `model.py` fits the legacy lineup-total model;
   it does not consume the possession cache or produce possession-based season estimates.
4. `app.py` is a local Flask dashboard showing current data counts and verification status.
   Legacy lineup-total estimates are displayed only when possession status is absent and
   aggregate status, model metadata, and `rapm.csv` are all present.

## Setup

Needs Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installing packages may use the network to download dependencies.

## Season readiness

The dashboard shows current cache counts and completion status; consult it rather than fixed
counts in this README. A cached raw game is not necessarily a validated game.

**Season estimates require all 1,230 regular-season possession games to be validated.**
The model gate requires a finished offline audit with matching expected, cached, and
validated game IDs, no failed or missing games, no validation errors, and verified
lineage for every normalized game. A partial or
selectively valid sample cannot be published as the season baseline. All games passing
is a data-coverage prerequisite, not evidence that the estimates are precise.
`model.py` still uses legacy lineup totals; use `possession_model.py` for the possession pipeline.

## Run the local dashboard

```bash
source .venv/bin/activate
python app.py           # then open http://127.0.0.1:5000 (stop with Ctrl-C)
```

The app runs only on your machine, reads files in `data/`, and makes no network requests.
With an incomplete possession audit, the page explains that player estimates are not certified.
A complete audit plus matching saved possession-model outputs enables their display;
missing or mismatched outputs are withheld. When neither status exists, it shows
"No data loaded yet". Legacy estimates are hidden whenever possession status exists.

## Audit the possession cache offline

`possessions_loader.py` builds a possession-level cache from NBA live play-by-play (parsed with `pbpstats`).
See [POSSESSION_DATA.md](POSSESSION_DATA.md) for validation and correction details.

```bash
python possessions_loader.py --offline        # full cached-game audit, zero network requests
```

The audit reads the existing 2024–25 game list, examines every listed game, and continues
past missing or invalid files. It writes only `data/cache/possessions/status.json`, leaving
raw play-by-play and normalized game caches untouched. A missing or malformed game list
cannot be fetched in offline mode. Reports are deterministic for unchanged inputs.

| Status field | Meaning |
| --- | --- |
| `games_expected` | Games in the validated season list: 1,230. |
| `games_cached` | Listed games whose raw cache paths exist, including invalid files. |
| `games_validated` | Cached games that passed all current validation rules. |
| `games_failed` | Cached games that failed parsing or validation. |
| `games_missing` | Listed games with no raw cache file. |

Sorted ID lists accompany these counts. Cached = validated + failed; expected = cached +
missing. `validation_errors` records per-game failures, and `game_results` records successful
counts and applied corrections. Possession totals include validated games only: segments,
counted possessions under pbpstats rules, and model-eligible possessions after exclusions.

`audit_complete` means the list was fully scanned; `complete` means all 1,230 games validated.
The audit exits 1 for an incomplete season or unusable list, and 0 for a fully validated
season. Read the report and dashboard for current counts; exit 1 does not mean later cached
games were skipped.

Without `--offline` the loader downloads any missing files, so only run it that way when you want to download data.

## Commands that can download game data

The integrity workflow is offline-only: use existing local data for auditing, fitting,
dashboard display, and tests. Do not download or repair inputs as part of verification.
The commands below are separate ingestion tools and require explicit authorization to use.

```bash
python possessions_loader.py   # can download missing possession play-by-play and the game list
python data_loader.py          # legacy loader: can download missing game-list and lineup-total data
```

These are network-enabled commands. Rerunning either normal loader can download missing
files even when some games are already cached. The `--offline` flag guarantees that the
possession loader makes no network requests; the legacy `data_loader.py` has no offline flag.

## Run the gated possession model

After a successful audit and with normalized caches for all audited games:

```bash
python -c "from possession_model import fit_possession_ridge; fit_possession_ridge()"
```

This local-only command checks the audit before loading inputs or fitting. It selects from
penalties 1, 10, 100, 1,000, and 10,000 using a deterministic 20% holdout of whole game IDs,
then refits all eligible possessions. It writes `data/possession_players.csv` and
`data/possession_model.json`, separate from legacy outputs. A rejected audit leaves existing
outputs untouched and creates no new ones. Running `python possession_model.py` alone does
not invoke this pipeline.

The summary contains player IDs, offensive/defensive possession counts, and offense,
defense, and net effects in points per 100 possessions. Positive defense means points
prevented; net = offense + defense. Metadata records the chosen penalty, holdout MSEs in
squared points per possession, split game IDs, intercept, and conventions. These MSEs tune
the penalty; they are not independent final evaluation scores. The dashboard can display
matching saved possession-model outputs after the same audit gate passes.

The offline audit does not regenerate normalized game files. The model requires verified
lineage for each game, checks current raw and normalized fingerprints against the audit,
and requires its design to cover the exact audited game-ID set. Missing or stale normalized
inputs block fitting and output writes. Refresh them separately and rerun the audit; inputs
are rechecked before output publication. Single-season estimates remain noisy and subject
to lineup/role confounding.

Fitting is blocked by any of the following:

- An absent, unfinished, inconsistent, or older non-audit status report.
- Any missing or failed game, or fewer than 1,230 validated games.
- Missing/unreadable normalized inputs, absent or unverified lineage, or mismatches in
  raw bytes, normalized bytes/content, recorded corrections, game metadata, or parser-version
  records compared with the audit. Even whitespace-only file changes need a fresh audit.
- Invalid normalized schemas or possession records, or missing/mismatched recorded,
  replayed, or attested component fingerprints for possessions, metadata, parser version,
  and normalized schema version. An audit reporting a fresh-rebuild mismatch always blocks fitting.
- A design whose game IDs differ from the audited set, or a cache/audit change detected
  before publishing the fitted outputs.

Read `game_results[game_id].lineage` and `lineage_counts` in the audit report for freshness
details. `complete=true` describes raw-game validation; it alone does not certify normalized
model inputs. Refresh missing or mismatched normalized files separately, rerun
`python possessions_loader.py --offline`, and inspect the new report before fitting.
The offline audit never downloads or repairs those inputs.

## Integrity workflow

The scope is the **2024–25 NBA regular season: all 1,230 games**. Raw validation
and input freshness are separate requirements. `audit_complete=true` means the audit
finished scanning; `complete=true` means every raw game validated, not that every saved
model input is fresh. The dashboard and fitting gate share the same comparison helper.
Estimates stay hidden until both requirements pass for every game and saved outputs
match the audit. Legacy estimates remain hidden whenever possession status exists.

| Input state | Meaning |
| --- | --- |
| Waiting on raw data (`missing_raw`) | The audit found no raw play-by-play to validate. These games are counted separately, not as stale or unverifiable normalized inputs. |
| Fresh | Current raw and normalized files match verified audit records and all required component fingerprints. |
| Stale | Files or recorded components disagree with the audit, or the audit found saved inputs differed from a fresh rebuild. Even formatting-only byte changes count. |
| Missing | A normalized input was absent at audit time, or a previously audited raw or normalized file has since been removed. |
| Unverifiable | Records are absent/unverified, files cannot be read, or normalized JSON/schema/possession records are invalid. |

For each game, the offline audit rebuilds normalized inputs in memory from validated raw
play-by-play using the guarded corrections and parser. It compares that rebuild with the
saved normalized file. SHA-256 fingerprints link raw bytes, normalized bytes and canonical
JSON content, plus possessions, applied corrections, game metadata, parser version, and
normalized schema version. Recorded, replayed, and verified component values must agree;
file existence or a successful raw validation alone is insufficient. Fingerprints establish
consistency with the audit, not independent authentication of an edited audit report.
Rerun the audit after changing correction/validation code or the parser environment.

Gate rejection, solver failure, or serialization failure occurs before output files or
directories are created or replaced. Immediately before publishing, fitting performs a
complete final gate check: it rereads the audit and every game's input bytes without the
dashboard's in-memory cache, checks all fingerprints, and rejects a changed audit snapshot.
Only after this succeeds are both output payloads staged and destinations replaced.
Thus a rejected fit preserves previously saved estimates and creates no new estimates.

The dashboard caches file facts by path, modification time in nanoseconds, size, and inode,
and caches gate results using those signatures and the audit-file signature. An edit that
preserves all these values (or a permission change without a signature change) can leave
the running dashboard showing cached results. Restart the dashboard to discard its
in-memory caches. Fitting's uncached final recheck does not inherit this limitation.
There is no filesystem lock: concurrent edits after the final check remain possible, and
the two output replacements are individually atomic, not a crash-safe pair transaction.

## Offline checks

```bash
python -B -m unittest test_possessions_offline test_possession_model test_dashboard -v
python -B test_model.py   # legacy model's synthetic offline checks
```

Audit tests block/count HTTP and socket attempts and use temporary cache copies. Model
tests cover gate rejections, numerical behavior, and isolated temporary output fixtures;
they do not generate real season estimates. No test downloads data or changes raw caches.

## Legacy lineup-total data checks

A game is left out of the model, and listed on the dashboard, if:
- either team has no lineups,
- a lineup does not have 5 distinct players,
- the lineups' points don't add up to the team's final score,
- the lineups' offensive possessions are more than 2 away from the game's possession count.

Lineups with zero possessions are dropped. Games that fail to download are listed too.

Possession validation and correction rules are documented in [POSSESSION_DATA.md](POSSESSION_DATA.md).
Normal online loading stops at the first source or validation failure; offline auditing
records failures and continues through the list.

## Legacy lineup-total model

`python model.py` reads `data/lineups.csv` and writes legacy `data/rapm.csv` and `data/model.json`.
It does not download data. `python test_model.py` checks this model and legacy data checks
using synthetic data; those checks do not certify season data or possession-based estimates.

Each row is one five-man lineup in one game:

- **outcome:** net points per 100 possessions, `100 * (points for - points against) / possessions`
- **weight:** possessions
- **inputs:** one column per player (1 if in the lineup), one per opponent team, plus home court

Ridge regression shrinks every estimate toward 0, which means an average player. The penalty (lambda)
is chosen by holding out 20% of games and keeping the value with the lowest prediction error.

**Unit:** points per 100 possessions better or worse than an average player, holding teammates fixed.

## Limitations

- **The legacy model is not textbook RAPM.** Its data gives full-game totals per lineup, not single possessions.
  Which opposing five was on the floor is not available, so opponents are adjusted by team, not by player.
- The possession loader uses the working `pbpstats` live-data interface. Possession data and
  both lineups are cached separately from the legacy model's lineup totals.
- One season of plus-minus is noisy. The legacy dashboard marks players with under 1,000
  possessions as "low sample".
- The legacy model combines offense and defense; the possession model estimates them separately.
