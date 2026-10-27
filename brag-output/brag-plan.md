# Rotation Index — brag plan

**What it is:** a local dashboard that shows how many points each NBA player adds per 100 possessions (offense and defense), using a ridge model fit on every possession of the 2025–26 regular season.
**For:** basketball nerds who want lineup-adjusted impact numbers and don't want numbers built on bad data.
**What sets it apart:** it won't show anything until all 1,230 games pass validation. At 1,229 the model stays "Locked".
**Most impressive / funniest claim:** its own diagnostics say it beats "guess the average" by 0.28%, and it shows that result in a green box.
**Visual hook:** the real masthead scoreboard at `1,229 / 1,230`, with Model: **Locked**. It ticks over to 1,230 and flips to **Available**.
**Real UI shown:** masthead, player estimates table, filter form, possession audit card, model diagnostics. All are cloned from the live rendered dashboard (`python app.py`) and use the project's own `style.css`.
**Tone:** polished, with one deadpan beat (the diagnostics).
**Share caption:** "Every 2025–26 NBA possession, fit with ridge regression. It refuses to show you a single number until all 1,230 games check out."

**Angle:** a stats tool that would rather show nothing than show something wrong.

**Data note:** player names are public NBA players taken from the real outputs. There are no secrets, hostnames or personal data. The "1,229 / Locked" opening state is a real state the app renders (`model_state = "Locked"` when the audit isn't ready); the video recreates that state from the real markup.

## Identity
NBA navy `#0b1f45` and `#17408b`, red `#c8102e`, good green `#17703f`, page bg `#f4f6fa`. Fonts are the system sans stack, plus mono for IDs. A court-line SVG sits in the masthead.

## Music
120 BPM (beat = 0.5s, bar = 2s), A minor, Am–F–C–G. During the hook, a basketball dribble thump lands on each beat. The full groove drops at 3.0s, when the model unlocks. All SFX are pitched to A and sit on the grid. Total: 21s, which is 42 beats.

## Storyboard
| # | Time | Beats | Scene | On-screen text |
|---|---|---|---|---|
| 1 Hook | 0.0–3.0 | 6 | Navy stage, real scoreboard large: `1,229 / 1,230`, Model **Locked** (amber), badge "Estimates hidden · 1 waiting". Dribbles on beats. | "1,229 games is not a season." |
| 2 Reveal | 3.0–6.0 | 6 | On the downbeat: counter → 1,230, Locked → Available (green), badge flips green. Camera pulls back to the full masthead. Music drops. | Real masthead: "Rotation Index / Ridge-adjusted NBA player impact per 100 possessions" |
| 3 Product | 6.0–11.0 | 10 | Estimates table: rows rise in, bars grow, SGA +10.43 net glows. At 8.5s "Wemb" is typed into the real filter, Filter is clicked, and the table collapses to V. Wembanyama. | "Every player. Offense + defense, per 100 possessions." |
| 4 Rigor | 11.0–15.0 | 8 | Audit card: progress bar fills to 1,230 of 1,230, counted possessions tick to 246,121, "All fresh" pill pops. | "246,121 possessions. Every file fingerprinted." |
| 5 Deadpan | 15.0–18.0 | 6 | Diagnostics: MAE row, 1.1346 vs 1.1378, then −0.0032 (−0.28%) gets highlighted. | "Beats the baseline by 0.28%. Says so." |
| 6 Outro | 18.0–21.0 | 6 | Navy, court lines, "Rotation Index", tagline, repo URL. | "NBA impact numbers. Only when every game checks out." / github.com/RohinCodes15/Rotation-Index |

Cuts land on beats 6, 12, 22, 30 and 36 (s = 3, 6, 11, 15, 18). Scene changes dip through the background, so busy layouts never double-expose.
