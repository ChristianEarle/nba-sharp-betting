# Archive — nba-sharp-betting Skill

Preserved contents of this repository prior to its rename and repurposing.
The files are unchanged from commit `905b6e6`; nothing here is maintained.

## Contents

| File | Lines | What it covers |
|------|-------|----------------|
| [`SKILL.md`](SKILL.md) | 446 | The skill itself. Sharp betting philosophy, Kalshi NBA market types, a five-step evaluation framework, Kelly sizing, timing windows, public-bias playbook, sportsbook↔Kalshi arbitrage, modeling metrics, bankroll management, daily workflow. |
| [`references/data-pipeline.md`](references/data-pipeline.md) | 523 | The Odds API and Kalshi API integration, injury feeds, schedule and back-to-back detection, advanced stats, edge-detection pipeline, backtesting framework. |
| [`references/kalshi-mechanics.md`](references/kalshi-mechanics.md) | 289 | Contract structure, order types, API endpoints, fees, position and exit management, platform quirks, live trading. |
| [`references/situational-edges.md`](references/situational-edges.md) | 246 | Back-to-backs, rest advantages, travel spots, home court, national-TV effects, seasonal context, referee tendencies, confluence scoring. |

`SKILL.md` carries YAML frontmatter naming the skill `nba-sharp-betting`. If
this repository is used as a skill source again, that frontmatter will still be
discoverable at its new path — move or remove it if that is not wanted.

## Notes

Nothing here is executable. The Python in `data-pipeline.md` is illustrative
prose, not a runnable package: there is no dependency manifest, no tests, and no
entry point.

The original root `README.md` contained only the repository name and carried no
information beyond it, so it is not reproduced here.
