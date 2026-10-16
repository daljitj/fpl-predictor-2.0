"""
Betting odds inputs for the FPL model, from football-data.co.uk (free).

Past seasons: mmz4281/<yyyy>/E0.csv (results plus odds for every match).
Upcoming games: fixtures.csv, refreshed by the site on Friday afternoons
for weekend games and on Tuesdays for midweek games.

The average pre-match odds across bookmakers (AvgH/AvgD/AvgA and
Avg>2.5/Avg<2.5) are used in both training and forecasting. They are
collected at about the same time as the fixtures.csv odds, so the model
learns from the same kind of number it is given for upcoming games.
Closing odds are sharper but aren't available before a deadline.

Each match's odds are turned into expected goals for each side, assuming
each side's goals follow a Poisson distribution (a standard assumption,
not an exact one). The bookmaker margin is removed proportionally. The
pair of expected goals that best reproduces the home win, away win and
over 2.5 goals probabilities is chosen.

Games without odds yet (usually gameweeks 2 and 3 of the forecast) get
expected goals from team ratings fitted to recent odds; see
ratings_goals(). Checked on 2025-26: ratings fitted on a date were
within about 0.09 goals, on average, of the odds-implied goals for games
one to three weeks later.
"""
import io
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import poisson

BASE = "https://www.football-data.co.uk"
CACHE_DIR = Path("odds_cache")
MAX_AGE_HOURS = 6
HEADERS = {"User-Agent": "Mozilla/5.0 (personal FPL model)"}
ODDS_COLS = ["odds_goals_for", "odds_goals_against"]
RATING_HALFLIFE_DAYS = 30   # ratings: a match a month old counts half
RATING_RIDGE = 0.05         # pull towards average for clubs with few matches
MAX_GOALS = 10


# ---------------------------------------------------------------- download
KEEP = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]


def _get_csv(path, cache_name, max_age_hours):
    f = CACHE_DIR / cache_name
    fresh = f.exists() and (max_age_hours is None or
                            time.time() - f.stat().st_mtime < max_age_hours * 3600)
    if not fresh:
        r = requests.get(BASE + path, headers=HEADERS, timeout=30)
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
    text = f.read_bytes().decode("utf-8-sig", errors="replace")
    d = pd.read_csv(io.StringIO(text))
    return d[[c for c in KEEP if c in d]].copy()   # only the columns used


def season_code(season):
    """'2025-26' -> '2526'."""
    return season[2:4] + season[5:7]


def load_matches(seasons, current):
    """One row per Premier League match: date, teams and pre-match odds.
    Finished seasons are cached for good; the current season and the
    upcoming fixtures are refreshed every MAX_AGE_HOURS."""
    frames = []
    for s in seasons:
        age = MAX_AGE_HOURS if s == current else None
        d = _get_csv(f"/mmz4281/{season_code(s)}/E0.csv",
                     f"E0_{season_code(s)}.csv", age)
        frames.append(d.assign(season=s, upcoming=False))
    fx = _get_csv("/fixtures.csv", "fixtures.csv", MAX_AGE_HOURS)
    fx = fx[fx["Div"] == "E0"]
    frames.append(fx.assign(season=current, upcoming=True))
    m = pd.concat(frames, ignore_index=True)
    m = m.dropna(subset=["HomeTeam", "AwayTeam"])
    m["date"] = pd.to_datetime(m["Date"], dayfirst=True)
    # a match can appear in both files for a few days; keep the result row
    m = (m.sort_values("upcoming")
          .drop_duplicates(["date", "HomeTeam", "AwayTeam"])
          .reset_index(drop=True))
    return implied_goals(m)


# ---------------------------------------------------------------- odds -> goals
def _probs(lh, la):
    """P(home win), P(away win), P(over 2.5) for independent Poisson goals."""
    g = np.arange(MAX_GOALS + 1)
    ph, pa = poisson.pmf(g, lh), poisson.pmf(g, la)
    grid = np.outer(ph, pa)                 # [home goals, away goals]
    i, j = np.meshgrid(g, g, indexing="ij")
    return (grid[i > j].sum(), grid[i < j].sum(), grid[i + j > 2].sum())


def _solve(p_home, p_away, p_over):
    target = np.array([p_home, p_away, p_over])
    res = least_squares(lambda x: np.array(_probs(*x)) - target,
                        x0=[1.5, 1.2], bounds=([0.05, 0.05], [6, 6]))
    return res.x


def implied_goals(m):
    """Adds goals_h and goals_a: expected goals for each side, derived from
    the average odds. Rows without complete odds get NaN."""
    m = m.copy()
    r = 1 / m[["AvgH", "AvgD", "AvgA"]].to_numpy(float)
    r = r / r.sum(axis=1, keepdims=True)    # remove the margin
    o = 1 / m[["Avg>2.5", "Avg<2.5"]].to_numpy(float)
    p_over = o[:, 0] / o.sum(axis=1)
    out = np.full((len(m), 2), np.nan)
    ok = np.isfinite(r).all(axis=1) & np.isfinite(p_over)
    for k in np.flatnonzero(ok):
        out[k] = _solve(r[k, 0], r[k, 2], p_over[k])
    m["goals_h"], m["goals_a"] = out[:, 0], out[:, 1]
    m["p_home"], m["p_away"], m["p_over"] = r[:, 0], r[:, 2], p_over
    return m


# ---------------------------------------------------------------- teams
def map_teams(fpl_fixtures, matches):
    """(season, football-data team name) -> FPL team name, without a
    hand-written table. Per season, every pairing of a football-data club
    with an FPL club is scored by the dates on which both played at home,
    plus the dates on which both played away, plus a bonus for similar
    names ('Man United' / 'Man Utd'); the best one-to-one assignment wins.
    The season's full FPL fixture list is used, so promoted clubs match
    from gameweek 1. Per season because FPL renames clubs ('Ipswich'
    became 'Ipswich Town')."""
    out = {}
    for season, o in matches.groupby("season"):
        f = fpl_fixtures[fpl_fixtures["season"] == season]
        f = f.dropna(subset=["home_name", "away_name"])
        if f.empty:
            continue
        fd = sorted(set(o["HomeTeam"]) | set(o["AwayTeam"]))
        fp = sorted(set(f["home_name"]) | set(f["away_name"]))
        score = np.zeros((len(fd), len(fp)))
        for side_o, side_f in (("HomeTeam", "home_name"), ("AwayTeam", "away_name")):
            j = (o[["date", side_o]].drop_duplicates()
                   .merge(f[["date", side_f]].drop_duplicates(), on="date"))
            n = j.groupby([side_o, side_f]).size()
            for (t, name), c in n.items():
                score[fd.index(t), fp.index(name)] += c
        for i, t in enumerate(fd):
            for k, name in enumerate(fp):
                score[i, k] += 3 * _similar(t, name)
        rows, cols = linear_sum_assignment(-score)
        out.update({(season, fd[i]): fp[k] for i, k in zip(rows, cols)})
    return out


def _similar(a, b):
    """0 to 1: 1 for the same name or one name starting the other
    ('Hull' / 'Hull City'), otherwise the share of matching letters."""
    a, b = a.lower().replace(".", ""), b.lower().replace(".", "")
    if a == b or a.startswith(b) or b.startswith(a):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _fpl_names(matches, names):
    """FPL names for the home and away sides of each odds row."""
    h = [names.get((s, t)) for s, t in zip(matches["season"], matches["HomeTeam"])]
    a = [names.get((s, t)) for s, t in zip(matches["season"], matches["AwayTeam"])]
    return matches.assign(home_name=h, away_name=a)


def fixtures_from_rows(gw, team_names):
    """(season, fixture, kickoff, home and away team) from player rows."""
    h = gw[gw["was_home"].astype(bool)]
    fx = (h.groupby(["season", "fixture"], as_index=False)
            .agg(kickoff_time=("kickoff_time", "first"),
                 team_h=("team_id", "first"),
                 team_a=("opponent_team", "first")))
    return add_names(fx, team_names)


def add_names(fx, team_names):
    """team_names: {(season, team id): FPL name}. UK date of kick-off,
    because football-data dates are UK dates."""
    fx = fx.copy()
    ko = pd.to_datetime(fx["kickoff_time"], utc=True)
    fx["date"] = ko.dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
    fx["home_name"] = [team_names.get((s, t)) for s, t in zip(fx["season"], fx["team_h"])]
    fx["away_name"] = [team_names.get((s, t)) for s, t in zip(fx["season"], fx["team_a"])]
    return fx


def attach_match_odds(fx, matches, names):
    """Adds goals_h / goals_a to FPL fixtures where odds exist. names comes
    from map_teams. Matches on the date and both teams, allowing one day
    either side for games moved late."""
    m = _fpl_names(matches, names)
    unmapped = sorted(set(zip(m.loc[m["home_name"].isna(), "season"],
                              m.loc[m["home_name"].isna(), "HomeTeam"])))
    if unmapped:
        print("Odds: no FPL team found for",
              ", ".join(f"{t} ({s})" for s, t in unmapped))
    m = m.dropna(subset=["home_name", "goals_h"])
    out = fx.copy()
    out["goals_h"] = np.nan
    out["goals_a"] = np.nan
    for shift in (0, -1, 1):
        need = out["goals_h"].isna()
        k = out.loc[need, ["season", "home_name", "away_name", "date"]].copy()
        k["date"] = k["date"] + pd.Timedelta(days=shift)
        k = k.reset_index().merge(
            m[["season", "home_name", "away_name", "date", "goals_h", "goals_a"]],
            on=["season", "home_name", "away_name", "date"], how="inner"
        ).drop_duplicates("index").set_index("index")
        out.loc[k.index, ["goals_h", "goals_a"]] = k[["goals_h", "goals_a"]]
    return out


# ---------------------------------------------------------------- ratings
def ratings_goals(matches, fx, as_of, names):
    """Expected goals for fixtures that have no odds yet, from team ratings
    fitted to odds-implied goals before as_of:
        log(home goals) = base + home advantage + attack(home) - defence(away)
        log(away goals) = base + attack(away) - defence(home)
    Recent matches weigh more (half-life RATING_HALFLIFE_DAYS). A small
    ridge penalty pulls clubs with few matches towards average. Clubs are
    identified by their football-data name, which doesn't change between
    seasons; fx's FPL names are translated back."""
    m = matches[(matches["date"] < as_of) & matches["goals_h"].notna()].copy()
    m = m.rename(columns={"HomeTeam": "home_name", "AwayTeam": "away_name"})
    back = {(s, n): t for (s, t), n in names.items()}
    fx = fx.copy()
    fd_h = [back.get((s, n)) for s, n in zip(fx["season"], fx["home_name"])]
    fd_a = [back.get((s, n)) for s, n in zip(fx["season"], fx["away_name"])]
    fx_h, fx_a = pd.Series(fd_h, index=fx.index), pd.Series(fd_a, index=fx.index)
    teams = sorted(set(m["home_name"]) | set(m["away_name"])
                   | set(fx_h.dropna()) | set(fx_a.dropna()))
    ix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    w = 0.5 ** ((as_of - m["date"]).dt.days / RATING_HALFLIFE_DAYS).to_numpy()
    n = len(m)
    # columns: base, home advantage, attack x T, defence x T
    X = np.zeros((2 * n, 2 + 2 * T))
    X[:, 0] = 1
    X[:n, 1] = 1
    h = m["home_name"].map(ix).to_numpy()
    a = m["away_name"].map(ix).to_numpy()
    r = np.arange(n)
    X[r, 2 + h] = 1; X[r, 2 + T + a] = -1
    X[n + r, 2 + a] = 1; X[n + r, 2 + T + h] = -1
    y = np.log(np.r_[m["goals_h"], m["goals_a"]])
    sw = np.sqrt(np.r_[w, w])
    ridge = np.sqrt(RATING_RIDGE) * np.eye(2 + 2 * T)[2:]
    A = np.vstack([X * sw[:, None], ridge])
    b = np.r_[y * sw, np.zeros(2 * T)]
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    base, hfa = coef[0], coef[1]
    att, dfn = coef[2:2 + T], coef[2 + T:]

    out = fx
    hi = fx_h.map(ix)
    ai = fx_a.map(ix)
    ok = hi.notna() & ai.notna()
    hi, ai = hi[ok].astype(int).to_numpy(), ai[ok].astype(int).to_numpy()
    out.loc[ok, "rating_h"] = np.exp(base + hfa + att[hi] - dfn[ai])
    out.loc[ok, "rating_a"] = np.exp(base + att[ai] - dfn[hi])
    return out


# ---------------------------------------------------------------- player rows
def to_player_rows(rows, fx):
    """Adds odds_goals_for / odds_goals_against to player rows, which need
    season, fixture and was_home."""
    f = fx[["season", "fixture", "goals_h", "goals_a"]]
    rows = rows.drop(columns=[c for c in ODDS_COLS if c in rows], errors="ignore")
    rows = rows.merge(f, on=["season", "fixture"], how="left")
    home = rows["was_home"].astype(bool)
    rows["odds_goals_for"] = np.where(home, rows["goals_h"], rows["goals_a"])
    rows["odds_goals_against"] = np.where(home, rows["goals_a"], rows["goals_h"])
    return rows.drop(columns=["goals_h", "goals_a"])
