# Possession data step

## Current season configuration

The configured season is **2025–26 Regular Season**. The authoritative pbpstats
manifest is pinned separately at `manifests/2025-26-regular-season.json` (retrieved
2026-10-24 from `get-games/nba?Season=2025-26&SeasonType=Regular%20Season`). Its
actual 1,230 game IDs, not a prefix-generated range or audit-declared count, define
the full-season gate. `season_config.py` owns the configured season and manifest.

Existing production caches have not been migrated or overwritten. A cached
2024–25 game list or audit cannot certify 2025–26, and `0022400061` is excluded
from the current manifest. Old raw/normalized leftovers are ignored by current
model loading; wrong identities inside current files are rejected. Current
metadata must match the pinned manifest. Missing, stale, unverifiable, or
wrong-season inputs block fitting/publication and hide dashboard estimates.
Historical 2024–25 examples and correction fixtures below remain regression
references, not current-season inputs. The tests explicitly isolate those
historical configurations; no play-by-play was fetched for this migration.

**Raw download, 2026-10-25:** an authorized run of `possessions_loader.py` fetched
raw play-by-play for all 1,230 manifest games sequentially (1,230 downloaded,
0 already cached, 0 failed; about 464 MB). No 2024–25 game was requested; the
existing `live_0022400061.json` was left untouched. Nothing was normalized or
audited by that run.

**Rebuild and offline audit, 2026-10-25:** `--rebuild-all` then `--offline`, local
files only. 1,223 of 1,230 games validated with verified lineage and fresh inputs;
7 failed closed, 0 missing, 0 stale; 0 corrections applied (the only verified
correction is for 2024–25). The model gate is **not ready** and no model was fit.
Failures, each needing official evidence before any correction:

- `0022500147` (DET/DAL, 2025-11-01): the live feed treats DET as home; the
  manifest lists DAL as home, so the final score check fails.
- `0022500458`, `0022500521`, `0022500550`, `0022500637`, `0022501029`,
  `0022501044`: pbpstats finds only four period-5 starters for one team
  (`InvalidNumberOfStartersException`), so the five-player lineup check fails.

**Model publication verification, 2026-10-25:** the existing full-season fit
entry point rejected this audit before invoking the solver. Read-only freshness
checks found 1,223 fresh games and 7 unverifiable games. The production dashboard
returned HTTP 200 with both possession and legacy estimates hidden; diagnostics
export returned HTTP 404. No production estimates or diagnostics were created or
overwritten. These seven data failures still require verified evidence before
the full-season fit can run; no validation or correction guards were relaxed.

**Evidence-backed fixes (audited in the rerun below):** three of the
seven blockers now normalize locally; four remain failed. `0022500550` and
`0022501044`: the missing OT starter is named by the period-5 opening jump ball,
which pbpstats' live parser ignores. `0022500147`: `VERIFIED_HOME_TEAMS` in
`possessions_loader.py` applies the official home team DET to the score check
only, guarded by the exact game ID and manifest teams, with the manifest
untouched. Evidence is in `fixtures/official_evidence_2025-26.json`, from the
official NBA live box score (`.../NBA/liveData/boxscore/boxscore_<game_id>.json`:
`homeTeam`, gameCode `20251101/DALDET`, Arena CDMX). The four OT games below stay
failed: that box score is end-of-game state only (`starter`, `oncourt` at the
final buzzer) and does not name who started overtime. An earlier minutes-arithmetic
correction was inference and has been removed. The NBA.com box score filtered to
OT1, recorded in the same fixture with screenshots in `fixtures/evidence/`, shows
MIN 5:00 (the whole period) for Aldama (MEM) and Grimes (PHI) in `0022500458`,
both teams' full five, Hauser (BOS) in `0022500637` and Washington (DAL) in
`0022501029`; for `0022500521` it shows "NO BOX SCORE DATA AVAILABLE", so BKN is
unresolved. `VERIFIED_PERIOD_STARTERS` in `possessions_loader.py` applies only those
four resolved entries: each matches the exact game, period 5, team and player (who
the feed must place on that team), and adds the player only when pbpstats found
exactly the recorded four starters for that team, cross-checked against the OT1
table; the five-player check still applies. There is no general "5:00 means
starter" rule, and Pritchard's `4:60` display is not used. Locally `0022500458`,
`0022500637` and `0022501029` now normalize; `0022500521` (BKN) has no
correction and still fails. Production audit not rerun since this change.
Original diagnosis:

- `0022500147`: not a parser bug. Team totals agree (DET 122, DAL 110) and the
  game passes every other check with the sides reversed in a scratch copy; the
  live feed's `scoreHome` tracks DET while the pinned pbpstats manifest lists DAL
  as home. No local scoreboard or box score breaks the tie. Required: an official
  NBA box score or scoreboard for this game ID naming the home team.
- `0022500458` (MEM S. Aldama `1630583`, PHI Q. Grimes `1629656`), `0022500521`
  (BKN N. Clowney `1641730`), `0022500637` (BOS S. Hauser `1630573`),
  `0022501029` (DAL P. Washington `1629023`): the named player finished period 4
  on court but appears in no period-5 action (no shot, foul, assist, block,
  steal, jump ball or substitution). Carrying the lineup across the break is
  inference, so no fallback is applied. Required: an official period-5 box score
  or official OT starters showing that player on court at 5:00 of overtime.

**Rebuild and offline audit rerun, 2026-10-25:** `--rebuild-all` (rebuilt 3,
unchanged 1,223, failed 4) then `--offline`, local files only, no network.
1,226 of 1,230 games validated, 4 failed closed, 0 missing; 247,298 possession
segments, 245,272 counted, 232,389 model-eligible. `complete` is false, so the
model gate is still **not ready** and no model was fit. Failed:
`0022500458`, `0022500521`, `0022500637`, `0022501029`, each with
`InvalidNumberOfStartersException` (four period-5 starters for one team), with no
normalized file written. In the real cache, `0022500147` scores DET 122, DAL 110
with manifest metadata still DAL home, and `0022500550` (Bona, PHI) and
`0022501044` (Washington, DAL) include the opening-tip starter in the first
eligible period-5 lineup; every eligible possession has five distinct players
per side and no correction was applied. A before/after SHA-256 comparison of every
project file found changes only to those three normalized games and
`status.json`: raw play-by-play, the manifest and all estimate and diagnostics
outputs are byte-identical, and none were created.

A separate three-game, test-only 2025–26 manifest exercises successful gated
publication entirely in a temporary directory. Its dashboard displays possession
estimates and exports matching 2025–26 diagnostics. Repeated fits produce identical
player CSVs, diagnostics, and model metadata except for `created_at`. Ridge
mathematics and output formats are unchanged. Production data and graphify outputs
remain untouched; verification uses no downloads or network requests.


Run with the existing virtual environment (verified with `pbpstats==1.3.11`):

```sh
.venv/bin/python possessions_loader.py --offline
```

This separate loader does not replace the existing aggregate-lineup loader or
connect incomplete data to season estimates. It requires `requests` and
`pbpstats==1.3.11`. The dashboard reads audit status; the possession model uses
the full-season audit as its fitting gate.

## Verified interface and sources

The installed package supports
`Client({"dir": cache, "Possessions": {"source": "web", "data_provider": "live"}}).Game(game_id)`.
Possessions are in `game.possessions.items`; each possession has `events`, whose
`current_players` maps both team IDs to player IDs. The loader downloads the same
NBA live JSON endpoint explicitly, checks HTTP status, and atomically caches the
response before using `source="file"` for parsing. This avoids repeatedly fetching
raw data and preserves the exact response for investigation.

Sources:

- [pbpstats package and current source](https://github.com/dblackrun/pbpstats)
- [pbpstats quickstart](https://github.com/dblackrun/pbpstats/blob/main/docs/quickstart.rst)
- [pbpstats.com API schema](https://api.pbpstats.com/openapi.json)

The cached API game list has 1,230 distinct regular-season game IDs beginning
`00224`. The public `/get-possessions/nba` endpoint responds for 2024–25 but its
records do not supply the ten on-court players; full-game lineup aggregates also
cannot supply the opposing five for each possession.

## Boundary and attribution convention

- Use pbpstats' live possession segmentation, which follows the feed's changes of
  offensive team and period ends. Keep all segments, including end-period tails.
  Record `count_as_possession` separately: pbpstats excludes certain possessions
  starting in the final two seconds. Never silently treat segment count as its
  published possession count.
- An offensive rebound continues the possession. Placeholder rebounds after
  nonfinal missed free throws are not real offensive rebounds.
- Ordinary made field goals and nontechnical free throws by the offense form
  the possession's points response. For free throws, record the associated
  nontechnical foul and its lineups for attribution, rather than assigning points
  to substitute players entering between free throws. Unmatched fouls flag the
  possession as unsuitable for modeling.
- Preserve technical free throws and any opponent scoring in a separate
  exceptional-points ledger. Flag affected possessions; do not pretend these
  points are ordinary offensive-possession outcomes. All scoring remains in the
  team total used for score reconciliation.
- Preserve both event lineups throughout each possession. A fixed pair of five-
  player lists is supplied only when the basketball-action attribution lineups
  agree. Flag changing lineups and exclude them from a future simple fixed-lineup
  regression; retain their events and scoring for audit. This exclusion may itself
  select the sample and must be disclosed when fitting a model.

## Validation and failure behavior

Check complete periods, game identity, exact once-only event coverage, five
distinct positive player IDs per team at basketball actions, disjoint opposing
lineups, and scoring events against both the API final score and the live feed's
final scoreboard. Sub-out/sub-in bookkeeping events can temporarily have fewer
than five players; those are retained but are not treated as basketball actions.
Count differences versus API possessions are reported explicitly, since provider
conventions may differ. Offline auditing records each missing or invalid game and
continues through the entire game list, as described below.

## Offline season audit

`python possessions_loader.py --offline` checks every one of the 1,230 game IDs in
the pinned 2025–26 manifest in sorted order; any old cached `games.json` is ignored.
A missing early game does not prevent processing later cached games. It never
fetches the game list or play-by-play, and pbpstats is configured with
`source="file"`; there is no network fallback.

The audit writes only `data/cache/possessions/status.json`, using an atomic
replacement with indented, sorted JSON. It leaves raw responses, normalized game
files, and `verification.json` unchanged. Existing normalized files are not proof
that a game passed this audit; consult its current status.

## Read-only storage preflight

Run `.venv/bin/python possessions_loader.py --storage-preflight` before downloading.
It counts the 2025–26 manifest games still missing and never fetches or writes files.
The JSON report includes available bytes, missing raw and normalized game counts,
non-empty sample counts, and the largest cached file size of each type. Estimated
additional space covers all missing raw and normalized files; the safety check
requires twice that estimate. Exit 0 means sufficient space, 1 means insufficient
space or an unknown estimate, and 2 means the local inputs could not be inspected.
Cached files from any season count as size samples (never as cached games), so a
new season can start from older samples. Without any non-empty sample for a needed
file type, the estimate is unknown and the check fails. This is an approximation: future games may exceed the sampled sizes,
and disk space can change after the check. The download command runs this same
check itself and stops before requesting any game if it fails; the standalone flag
never starts a download.

## Downloading missing raw files

The separate network-enabled command `python possessions_loader.py` only acquires
missing raw play-by-play for the pinned 2025–26 manifest; it never requests a game list.
It first runs the storage preflight above and stops, requesting no game and writing
no report, unless that check passes. Then for each listed game it:

- downloads sequentially and skips games already cached; an existing raw file is never
  re-downloaded or replaced, and one that fails validation is reported as `cached_<reason>`;
- retries timeouts, connection errors and HTTP 429/500/502/503/504 up to 5 attempts,
  waiting 1, 2, 4 and 8 seconds between them; other HTTP errors fail at once;
- saves the feed atomically only if it parses as JSON for that game ID with actions
  through at least period 4; anything else is not written;
- records a failure and continues with the next game.

An interrupted run (Ctrl-C, crash) leaves only whole files: the feed being written exists
only as a `.tmp` file, and rerunning the same command resumes from the first missing game.

It writes `data/cache/possessions/download_report.json` (`downloaded_game_ids`,
`failed_game_ids`, and `failures` mapping each failed ID to a stable reason such as
`http_404`, `invalid_json`, `incomplete_game` or `http_503_after_5_attempts`) and
exits 1 if any game failed. It never normalizes, writes `status.json` or
`verification.json`, or runs the model. Afterwards run `--rebuild-all` to build
normalized files from the new raw files, then `--offline` to audit them.

The offline audit report (`status.json`) contains:

- `games_expected` is the required season-list size (1,230); `games_cached` counts
  listed raw cache paths that exist; `games_validated` counts those passing all
  checks; `games_failed` counts cached files that fail; `games_missing` counts
  absent files. Corresponding sorted game-ID lists identify each category.
  Cached = validated + failed; expected = cached + missing, once a valid list is read.
- `validation_errors`, keyed by failed game ID, and `game_results` with per-cached-
  game validation outcomes. Invalid JSON, parsing errors, score mismatches, and
  lineup/correction failures are failures, not missing data.
- Total segments, counted possessions, and model-eligible possessions from
  **validated games only**, plus per-game counts, team scores/possession counts,
  exclusion flags, count differences versus the API, and applied corrections.
  Counts from failed games are not included as valid observations.
- `audit_complete=true` when the whole list was examined, even if files were
  missing or invalid. `complete=true` only when all 1,230 games validated.
  An incomplete season or unusable list exits with code 1; a complete validated
  season exits with code 0. This audit does not fit a model.

Repeated audits of identical inputs produce byte-identical reports: no run
timestamps, sorted IDs/keys, and errors without machine-specific raw-cache paths.
The dashboard shows current counts; this document's opener example is not a
claim of season readiness.

Run the relevant offline tests with:

```sh
.venv/bin/python -B -m unittest test_possessions_offline test_possession_model test_dashboard -v
```

Tests use temporary cache copies, reject and count HTTP/socket attempts, and
assert zero network calls. They cover gaps before the real cached opener,
malformed game files, deterministic reporting, unchanged raw bytes, the guarded
substitution correction, and the unchanged five-player check. Model tests check
incomplete, failed, inconsistent, and complete audit fixtures, including no new or
overwritten outputs on rejection. Successful output tests use temporary fixtures,
not real season estimates.

## Run the possession model after a complete audit

With dependencies already installed, use:

```sh
.venv/bin/python possessions_loader.py --offline
# Only after all games validate and normalized game files are available:
.venv/bin/python -c "from possession_model import fit_possession_ridge; fit_possession_ridge()"
```

Neither command downloads data. The second command writes model outputs only if
the audit gate passes. Calling `python possession_model.py` alone does not run a fit.
The gate requires `mode="offline_audit"`, `audit_complete=true`, `complete=true`,
all 1,230 games expected/cached/validated, zero failed/missing games, empty error
records, matching ID sets, and a validated per-game outcome for each expected ID.
The model's design must contain that exact game-ID set, and every game must also
pass the lineage/freshness checks below. Old online progress flags or raw-complete
counts alone cannot authorize a fit. Incomplete/failed audits are rejected before output
creation or replacement, including any existing estimates.

Requiring all games prevents a partial or selectively validated cache from being
presented as the full regular-season baseline. It does not remove single-season
noise, teammate collinearity, coaching/role confounding, or selection caused by
excluding ambiguous possessions within validated games.

The pipeline selects a common offense/defense ridge penalty from
`1, 10, 100, 1000, 10000` for its sum-of-squared-errors objective. It holds out 20%
of whole games, using a fixed SHA-256 ranking of game IDs, and compares mean
squared error with equal weight per held-out possession. No game contributes
possessions to both subsets. It then refits all eligible possessions using the
chosen penalty. The intercept is unpenalized; offense and defense coefficients
are centered separately with a prediction-preserving intercept adjustment.

Outputs, separate from legacy `rapm.csv`/`model.json`:

- `data/possession_players.csv`: player IDs, offensive/defensive possession counts,
  offense/defense effects per 100 possessions, and net = offense + defense.
  Positive defense means points prevented. Names are not inferred or added to this CSV.
- `data/possession_model.json`: selected penalty and candidate holdout MSEs,
  training/holdout game IDs, sample sizes, intercept, units, and limitations.
  Holdout errors use squared points per possession and are tuning scores rather
  than an independent final evaluation.

## Per-game cache lineage

Every expected game's `game_results` entry now includes `lineage`. The audit hashes
the existing raw and normalized file bytes with SHA-256 and replays normalization
from current raw data, game-list metadata, pbpstats, and guarded corrections. It
compares the complete replayed output with the saved normalized JSON using a
canonical fingerprint (sorted keys, compact UTF-8 JSON). Formatting-only changes
alter the file-byte hash but not that canonical content fingerprint.

`lineage.status="verified"` and `verified=true` mean the existing normalized data
matches this replay. Only then is `verified_inputs` recorded, linking raw bytes,
normalized content, applied corrections, game metadata, and pbpstats version.
This establishes reproducibility from the current inputs, not historical proof
of when or how an older file was created. Observed hashes alone are not lineage.

Missing raw/normalized files, unreadable or invalid normalized JSON, failed raw
validation, files changing during replay, and content mismatches remain explicitly
unverified. They never receive `verified_inputs`. Available file hashes are still
reported. Recorded versus replayed correction fingerprints make stale correction
records visible. `lineage_counts` summarizes these outcomes across the game list.
The audit creates no normalized files and changes no raw files.

The model readiness gate requires verified lineage for every expected game and
rechecks current raw/normalized bytes, normalized content, correction records,
and metadata against the audit fingerprints. Missing, stale, or unverifiable
inputs block fitting and output writes even when every raw game validated.
Fitting rechecks inputs and the audit immediately before publishing; a change
during the fit leaves existing estimate outputs untouched. Refresh stale
normalized files separately and rerun the offline audit before trying again.
Even formatting-only file changes require a fresh audit to record the new bytes.

The audit still does not regenerate normalized files, and raw-game completion
counts remain separate from lineage readiness. The dashboard uses the same
readiness gate and its saved-output metadata checks.

To diagnose a rejection, inspect `validation_errors` for raw-game failures and
`game_results[game_id].lineage` for normalized-input failures:

| Audit/lineage condition | Effect on fitting |
| --- | --- |
| Audit unfinished, errors present, or expected/cached/validated IDs disagree | Blocked before penalty selection. |
| A game is missing or failed, even if a summary flag claims completion | Blocked; all 1,230 games must validate. |
| Lineage absent, unverified, missing raw/normalized input, or invalid normalized JSON | Blocked; no lineage is inferred from file existence. |
| Raw or normalized bytes changed after the audit | Blocked until re-audited, even for formatting-only changes. |
| Canonical normalized content, correction fingerprints, game metadata, or parser-version records disagree | Blocked as stale or unverifiable input. |
| All verified inputs match and the design covers the audited IDs | Penalty selection/full-data fitting may proceed. |
| Inputs or audit change before output publication | Output creation/replacement is rejected. |

These checks establish consistency with the recorded audit; they do not establish
historical provenance, independently authenticate an edited status report, or repair
source data. Rerun the audit after changing validation/correction code or the parser
environment: the model gate compares recorded fingerprints, rather than rerunning
the normalization algorithm itself. A fully ready cache still does not remove
single-season statistical limitations.

Cache locations:

- `manifests/2025-26-regular-season.json`: pinned season game list (an old cached
  `data/cache/possessions/games.json` is no longer read).
- `data/cache/possessions/pbpstats/pbp/live_<game_id>.json`: original NBA response.
- `data/cache/possessions/games/<game_id>.json`: validated normalized possessions,
  only written after a whole game passes.
- `data/cache/possessions/status.json`: season audit counts, IDs, and errors (or
  progress/failure status from a normal online load).
- `data/cache/possessions/verification.json`: local inspection of the first game.

## Verified opener example

For game `0022400061` (NYK at BOS, October 22, 2024), scoring reconciles exactly to
BOS 132, NYK 109. The five-player check correctly caught six Boston players at
action 511 (4:04 in Q4).

The verified cause is action 509 at 4:20 in Q4: the live feed says Al Horford
was substituted out, but the official NBA game book records Horford leaving for
Luke Kornet at 8:35 and gives Kornet 4:15 and Queta 4:20 in Q4. The outgoing
player when Queta entered at 4:20 was Kornet. On the parsed pbpstats event, the
loader changes `player1_id` from Horford (201143) to Kornet (1628436) before any
lineup is requested. It does not alter the cached raw feed or modify pbpstats'
lineup logic. The existing five-player check is unchanged; after this correction
the basketball-action lineups in this game pass and the score still reconciles.

This correction is explicitly guarded by game ID, action number, Q4, 4:20 clock,
Boston team, substitution type `out`, and the expected original player ID. If the
feed changes, that game's normalization fails with an error. The offline audit
records the failure and continues. Each additional correction must
have its own verified table entry. The offline audit records the applied correction
and evidence under that game's `game_results` entry in `status.json`. An existing
`verification.json` from earlier loads is left unchanged.

Only cached data is used in the offline audit; it does not resume downloads.
Season readiness requires all 1,230 games validated; see the current audit status
and dashboard counts. The official correction evidence is the [NBA game book](https://statsdmz.nba.com/pdfs/20241022/20241022_NYKBOS_book.pdf).
