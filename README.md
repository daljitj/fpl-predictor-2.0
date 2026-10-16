# FPL predictor 2.0

Predicts Fantasy Premier League points for every player from free data, then picks a squad or suggests transfers within FPL's rules.

## Data

- Past seasons (2022-23 onwards): the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive
- Current season: the official FPL API
- Shot detail: Understat
- Betting odds: [football-data.co.uk](https://www.football-data.co.uk)

## How it works

- A gradient-boosted model (scikit-learn) predicts each player's expected points per fixture from recent form, opponent strength and odds-implied expected goals.
- A second model predicts the spread of likely scores, giving an 80% range and a haul (10+) chance.
- An integer programme (HiGHS via SciPy) picks 15 players and a separate XI and captain for each of the next 3 gameweeks.

## Files

| File | What it does |
| --- | --- |
| `fpl_model.py` | Main script: data, features, model, forecast, squad or transfers |
| `fpl_api.py` | Official FPL API, cached in `fpl_api_cache/` |
| `understat_loader.py` | Understat stats and player matching, cached in `understat_cache/` |
| `odds_loader.py` | Betting odds to expected goals, cached in `odds_cache/` |
| `ranges.py` | Points distribution: 80% range and haul chance |
| `optimiser.py` | Squad and transfer optimiser |
| `my_team.py` | Your squad, bank, selling prices and free transfers from your team ID |
| `tune.py` | Walk-forward tuning |
| `track.py` | Scores saved forecasts against actual points |

## Setup

Python 3.11+ and Poetry 2.

```
poetry install
poetry run poe          # list commands
```

Put your FPL team ID in `pyproject.toml` (`FPL_TEAM_ID`).

## Weekly use

```
poetry run poe weekly       # score last gameweek, then suggest transfers
poetry run poe wildcard     # best squad from scratch within £100.0m
poetry run poe track        # tracking report
poetry run poe tune         # walk-forward tuning, every 4 to 6 weeks
```

Outputs go to `output/` (not committed). The scripts never change your FPL team; make transfers in the FPL app yourself.
