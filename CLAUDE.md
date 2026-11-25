# FPL predictor 2.0

Daljit's personal Fantasy Premier League model: predicts every player's points from free data, then picks a squad or suggests transfers within FPL's rules. Personal project, not Lumen work.

- Code: https://github.com/daljitj/fpl-predictor-2.0 (public)
- Local folder: `C:\Users\Daljit Johal\Documents\Other\fpl-predictor-2.0` (Windows, PowerShell)
- Full background, test results and decisions: the "FPL prediction model: project handover" doc on claude.ai
- FPL team ID: 6645275 (set as `FPL_TEAM_ID` in `pyproject.toml`)

## Working with Daljit

- He is a data scientist, fluent in R, knows some Python. Explain Python and Windows specifics briefly; don't over-explain modelling.
- When changing code, show what changed and why, so he can review key decisions, and always say where each file was saved.
- He makes all FPL transfers himself. Never automate anything that changes his FPL team.
- Plain British English, no em dashes.

## Files

| File | What it does |
| --- | --- |
| `fpl_model.py` | Main script: data, features, model, forecast, optimiser, saves outputs. Holds `CONFIG` and the switches (`USE_UNDERSTAT`, `USE_ODDS`, `LIVE_CURRENT`, `HORIZON`) |
| `fpl_api.py` | Official FPL API, cached in `fpl_api_cache/` (6 hours; team data 15 minutes) |
| `understat_loader.py` | Understat stats and FPL-Understat player matching, cached in `understat_cache/`; falls back to saved data if Understat fails |
| `odds_loader.py` | football-data.co.uk odds to expected goals; team ratings for fixtures without odds; cached in `odds_cache/` |
| `ranges.py` | Points distribution: 80% range and haul (10+) chance |
| `optimiser.py` | Integer programme (HiGHS via `scipy.optimize.milp`): 15-man squad, XI and captain per gameweek |
| `my_team.py` | Squad, bank, selling prices and free transfers from the public FPL API (prices and free transfers are estimates) |
| `tune.py` | Walk-forward tuning; `--odds-only` compares with and without odds |
| `track.py` | Scores saved forecasts against actual points, odds vs no-odds, range calibration |
| `due.py` | Decides whether a deadline is 3 to 20 hours away and no plan is saved yet |
| `notify.py` | Builds the phone message from `output/` and posts it to ntfy (`--dry-run` prints it) |
| `.github/workflows/weekly.yml` | Twice-daily GitHub Actions check; runs track, transfers (falls back to no Understat), notify, then commits `output/` |

## Commands (from the project folder)

Always prefix with `poetry run`; bare `python` misses the environment.

```
poetry run poe weekly      # track, then transfers: the weekly routine before the deadline
poetry run poe transfers   # transfer options, line-up, captain
poetry run poe wildcard    # best squad from scratch within £100.0m
poetry run poe quick       # without Understat
poetry run poe no-odds     # without betting odds
poetry run poe track       # tracking report
poetry run poe tune        # walk-forward tuning, every 4 to 6 weeks
poetry run poe refresh     # clear FPL and odds caches
```

## Outputs

An automatic run before each deadline (GitHub Actions, see `README.md`) sends the suggestions to Daljit's phone with ntfy; the topic name is the `NTFY_TOPIC` repository secret and must never be written into the repo.

`output/` holds `predictions_<season>_gw<N>.csv`, `plan_<season>_gw<N>.json`, `shadow_no-odds_<season>_gw<N>.csv` and `tracking_*.csv`. `output/` is committed each gameweek as a record (by the workflow, or by Daljit), so don't add it to `.gitignore`; `git pull` before committing locally. `track.py` ignores forecasts saved after a deadline.

## Model in brief

- One `HistGradientBoostingRegressor` on player-match rows from 2022-23 onwards; target is FPL points; inputs use only earlier matches (shift of one).
- Form: exponentially weighted, half-life 2 matches, carried across seasons via FPL's `code`.
- Inputs: FPL form stats, Understat form stats, opponent rolling xG, odds-implied goals for and against, games played, price, home/away, position.
- Forecast: next 3 gameweeks; doubles summed, blanks zero; scaled by FPL's chance of playing.
- Judge changes with walk-forward tests (`tune.py`) and live tracking (`track.py`), RMSE on regulars as the main measure. Differences of 0.02 to 0.03 RMSE can be noise.

## Data sources

- Past seasons: vaastav/Fantasy-Premier-League archive on GitHub (read as UTF-8).
- Current season: official FPL API. Returns 503 while FPL updates around deadlines and after matches; wait and retry.
- Understat: JSON endpoints with an `X-Requested-With` header; sometimes returns an HTML page instead of data.
- football-data.co.uk: `mmz4281/<yyyy>/E0.csv` and `fixtures.csv` (refreshed Friday afternoons and Tuesdays). Use average pre-match odds, not closing odds.
- Not used: FBref (advanced stats removed January 2026), StatsBomb (too old).

## Gotchas

- Save `.toml` files as UTF-8. Notepad or PowerShell `>` can write UTF-16 and break Poetry (`'\x00'` errors).
- After moving or renaming the folder, delete `.venv` and run `poetry install`; the old environment keeps absolute paths.
- Club names differ between sources and FPL renames clubs between seasons; matching is by shared match dates per season, never a hand-written table.
- Test changes end to end in both modes (from scratch and `--team-id`); an indentation slip once broke only transfer mode.
- Free transfers are estimated by replaying the history (no public endpoint gives them). In a Wildcard or Free Hit week saved transfers are kept but no new one is added; confirmed against Daljit's app in GW12. Override with `--free-transfers` or the workflow's `free_transfers` input.
- Without network access to FPL, test with `LIVE_CURRENT = False` (uses the archive) and mocked `my_team.load_my_team`.

## Roadmap (next)

Minutes model (chance of starting and of 60+ minutes); then predict scoring events separately; multi-week transfer planner; separate models per horizon.
