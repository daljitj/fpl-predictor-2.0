"""
FPL points predictor and squad optimiser, built only on free data.

Data: vaastav/Fantasy-Premier-League on GitHub (a free archive of the
official FPL API) plus Understat via understat_loader.py.

poetry install                              # once, sets up the environment
poetry run python fpl_model.py              # train, forecast, pick a squad
poetry run python fpl_model.py --no-understat   # quicker first test
poetry run python fpl_model.py --team-id 1234567   # transfer suggestions
poetry run python tune.py                   # walk-forward tuning (occasionally)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import fpl_api
import odds_loader
import optimiser
import ranges
from understat_loader import US_COLS, add_understat

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]  # xG from 2022-23
CURRENT = "2026-27"
HORIZON = 3         # number of upcoming gameweeks to optimise for
BUDGET = 1000       # FPL prices are in tenths, so 1000 = £100.0m
BENCH_WEIGHT = 0.1  # how much bench points count in the objective
USE_UNDERSTAT = True
USE_ODDS = True     # betting odds for team strength (football-data.co.uk)
LIVE_CURRENT = True  # current season from the official FPL API, not the archive

# Model settings. tune.py compares alternatives; paste its winner here.
# Current values: tune.py winner on FPL data only (Understat not included).
CONFIG = {
    "form": "ewm",            # "ewm" (recent games weigh more) or "rolling"
    "halflife": 2,            # ewm: a game 2 matches ago counts half as much
    "window": 5,              # rolling: number of matches, equally weighted
    "carry_over": True,       # form continues across the summer break
    "sample_halflife": None,  # days; older training rows weigh less. None = equal
    "per_position": False,    # one model per position instead of one shared
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
}

# FPL columns; tackles, recoveries and CBI exist from 2025-26 onwards only
FPL_COLS = ["minutes", "starts", "total_points", "expected_goals",
            "expected_assists", "expected_goals_conceded", "bps",
            "ict_index", "saves", "defensive_contribution", "tackles",
            "recoveries", "clearances_blocks_interceptions"]
POS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
TEAM_WINDOW = 5


def roll_cols():
    return FPL_COLS + (US_COLS if USE_UNDERSTAT else [])


def features():
    return ([f"r_{c}" for c in roll_cols()] +
            ["games_played", "value", "was_home", "pos",
             "opp_xg_for", "opp_xg_against"] +
            (odds_loader.ODDS_COLS if USE_ODDS else []))


# ---------------------------------------------------------------- data
def season_files(season):
    """(gameweek rows, teams, players) for a season, from the live API for
    the current season and from the GitHub archive otherwise."""
    if season == CURRENT and LIVE_CURRENT:
        return fpl_api.gameweek_history(), fpl_api.teams(), fpl_api.players_raw()
    return (pd.read_csv(f"{BASE}/{season}/gws/merged_gw.csv", encoding="utf-8"),
            pd.read_csv(f"{BASE}/{season}/teams.csv"),
            pd.read_csv(f"{BASE}/{season}/players_raw.csv", encoding="utf-8"))


def current_reference():
    """(players, fixtures, teams) for the season being forecast."""
    if LIVE_CURRENT:
        return fpl_api.players_raw(), fpl_api.fixtures(), fpl_api.teams()
    return (pd.read_csv(f"{BASE}/{CURRENT}/players_raw.csv"),
            pd.read_csv(f"{BASE}/{CURRENT}/fixtures.csv"),
            pd.read_csv(f"{BASE}/{CURRENT}/teams.csv"))


def load_season(season):
    print(f"Loading {season}", flush=True)
    gw, teams, raw = season_files(season)
    teams = teams[["id", "name"]]
    gw["season"] = season
    gw["team_id"] = gw["team"].map(dict(zip(teams["name"], teams["id"])))
    # 'code' is FPL's player ID that stays the same across seasons
    gw["code"] = gw["element"].map(dict(zip(raw["id"], raw["code"])))
    gw["web_name"] = gw["element"].map(dict(zip(raw["id"], raw["web_name"])))
    gw["position"] = gw["position"].replace({"GKP": "GK"})
    gw = gw[gw["position"].isin(POS)]
    for c in FPL_COLS:
        if c not in gw:
            gw[c] = np.nan  # e.g. tackles before 2025-26
    gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"])
    if USE_UNDERSTAT:
        gw = add_understat(gw)
    return gw


def load_history(seasons):
    hist = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    if USE_ODDS:
        hist = add_odds(hist, seasons)
    return hist.sort_values(["kickoff_time", "fixture"]).reset_index(drop=True)


_odds = {}   # odds matches and team-name mapping, reused by forecast()


def add_odds(hist, seasons):
    """Odds-implied expected goals for and against, per player per match."""
    print("Loading betting odds", flush=True)
    try:
        matches = odds_loader.load_matches(seasons, CURRENT)
    except Exception as e:
        raise SystemExit(f"Couldn't download betting odds ({e}). Try again "
                         "later, or run without them: poetry run poe no-odds")
    names = dict(zip(zip(hist["season"], hist["team_id"]), hist["team"]))
    fx = odds_loader.fixtures_from_rows(hist, names)
    # the full fixture list for this season also lets promoted clubs be
    # matched before they have played at home
    known = fx
    if CURRENT in seasons:
        _, cur_fx, cur_teams = current_reference()
        cur_names = {(CURRENT, i): n for i, n in zip(cur_teams["id"], cur_teams["name"])}
        known = pd.concat([fx, odds_loader.add_names(
            cur_fx[["team_h", "team_a", "kickoff_time"]].assign(season=CURRENT),
            cur_names)])
    team_map = odds_loader.map_teams(known, matches)
    fx = odds_loader.attach_match_odds(fx, matches, team_map)
    cover = fx.groupby("season")["goals_h"].apply(lambda x: x.notna().mean())
    print("Odds found for " + ", ".join(f"{s} {v:.0%}" for s, v in cover.items())
          + " of matches")
    _odds.update(matches=matches, team_map=team_map)
    return odds_loader.to_player_rows(hist, fx)


def upcoming_odds(upcoming, teams, quiet=False):
    """Expected goals for upcoming FPL fixtures: from this week's odds where
    football-data has them, otherwise from team ratings fitted to recent
    odds. Returns one row per fixture and a note on the split."""
    names = {(CURRENT, i): n for i, n in zip(teams["id"], teams["name"])}
    fx = upcoming[["id", "event", "team_h", "team_a", "kickoff_time"]].rename(
        columns={"id": "fixture"}).assign(season=CURRENT)
    fx = odds_loader.add_names(fx, names)
    m, team_map = _odds["matches"], _odds["team_map"]
    fx = odds_loader.attach_match_odds(fx, m, team_map)
    fx = odds_loader.ratings_goals(m, fx, pd.Timestamp.now().normalize(), team_map)
    from_odds = fx["goals_h"].notna()
    fx["goals_h"] = fx["goals_h"].fillna(fx["rating_h"])
    fx["goals_a"] = fx["goals_a"].fillna(fx["rating_a"])
    by_gw = (fx.assign(o=from_odds).groupby("event")["o"]
               .agg(lambda x: f"{x.sum()}/{len(x)}"))
    if not quiet:
        print("Fixtures with bookmaker odds (the rest use odds-based team "
              "ratings): " + ", ".join(f"GW{int(e)} {v}" for e, v in by_gw.items()))
    return fx


def team_strength(gw, shift):
    """Rolling xG for and against per team, from player-level rows."""
    tf = (gw.groupby(["season", "fixture", "team_id"], as_index=False)
            .agg(xg_for=("expected_goals", "sum"),
                 kickoff_time=("kickoff_time", "first")))
    opp = tf.rename(columns={"team_id": "opp_id", "xg_for": "xg_against"})
    tf = tf.merge(opp[["season", "fixture", "opp_id", "xg_against"]],
                  on=["season", "fixture"])
    tf = tf[tf["team_id"] != tf["opp_id"]].sort_values("kickoff_time")
    g = tf.groupby(["season", "team_id"])
    for c in ["xg_for", "xg_against"]:
        s = g[c].shift(1) if shift else tf[c]
        tf[f"team_{c}"] = (s.groupby([tf["season"], tf["team_id"]])
                            .transform(lambda x: x.rolling(TEAM_WINDOW, 1).mean()))
    return tf


def smooth(df, cols, key, cfg, shift):
    """Per-player form. shift=True uses only earlier games, so a training
    row never contains the result it is trying to predict."""
    vals = (df.groupby(key, sort=False, dropna=False)[cols].shift(1)
            if shift else df[cols])
    grouped = vals.groupby([df[k] for k in key], sort=False, dropna=False)
    if cfg["form"] == "ewm":
        out = grouped.ewm(halflife=cfg["halflife"], min_periods=1).mean()
    else:
        out = grouped.rolling(cfg["window"], min_periods=1).mean()
    return out.reset_index(level=list(range(len(key))), drop=True).loc[df.index]


def add_features(gw, cfg, shift=True):
    gw = gw.sort_values("kickoff_time")
    # carry_over: one history per player across seasons (players new to the
    # Premier League still start blank); otherwise reset every season
    key = ["code"] if cfg["carry_over"] else ["season", "element"]
    gw[[f"r_{c}" for c in roll_cols()]] = smooth(gw, roll_cols(), key, cfg,
                                                 shift).to_numpy()
    gw["games_played"] = (gw.groupby(["season", "element"]).cumcount()
                          + (0 if shift else 1))
    return gw


def attach_opponent(rows, ts):
    o = ts[["season", "fixture", "team_id", "team_xg_for", "team_xg_against"]]
    o = o.rename(columns={"team_id": "opponent_team",
                          "team_xg_for": "opp_xg_for",
                          "team_xg_against": "opp_xg_against"})
    return rows.merge(o, on=["season", "fixture", "opponent_team"], how="left")


def prep(df):
    df["pos"] = df["position"].map(POS)
    df["was_home"] = df["was_home"].astype(int)
    return df


def build_training_table(hist, cfg):
    df = add_features(hist.copy(), cfg, shift=True)
    df = prep(attach_opponent(df, team_strength(df, shift=True)))
    # 'regular': averaged 60+ minutes over his previous three games this
    # season. Used only to report errors on players you'd consider picking.
    prev = df.groupby(["season", "element"])["minutes"].shift(1)
    df["regular"] = (prev.groupby([df["season"], df["element"]])
                       .transform(lambda x: x.rolling(3, 3).mean()) >= 60)
    return df


# ---------------------------------------------------------------- model
class PointsModel:
    """One gradient-boosted model, or one per position."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.models = {}
        self.cols = {}

    def _new(self, cols):
        c = self.cfg
        cat = [f == "pos" for f in cols] if not c["per_position"] else None
        return HistGradientBoostingRegressor(
            learning_rate=c["learning_rate"], max_iter=c["max_iter"],
            max_leaf_nodes=c["max_leaf_nodes"],
            min_samples_leaf=c["min_samples_leaf"],
            categorical_features=cat, early_stopping=False, random_state=0)

    def _weights(self, df):
        hl = self.cfg["sample_halflife"]
        if not hl:
            return None
        age = (df["kickoff_time"].max() - df["kickoff_time"]).dt.days
        return 0.5 ** (age / hl)

    def fit(self, df):
        groups = df.groupby("pos") if self.cfg["per_position"] else [(0, df)]
        for p, d in groups:
            # skip inputs with no values at all in this training data, e.g.
            # tackles when every training match predates 2025-26
            cols = [f for f in features() if d[f].notna().any()]
            self.cols[p] = cols
            self.models[p] = self._new(cols).fit(
                d[cols], d["total_points"], sample_weight=self._weights(d))
        return self

    def predict(self, df):
        if not self.cfg["per_position"]:
            return self.models[0].predict(df[self.cols[0]])
        out = pd.Series(0.0, index=df.index)
        for p, m in self.models.items():
            idx = df.index[df["pos"] == p]
            if len(idx):
                out[idx] = m.predict(df.loc[idx, self.cols[p]])
        return out.to_numpy()


# ---------------------------------------------------------------- forecast
def forecast(model, hist, cfg, quiet=False, dist_model=None):
    players, fixtures, teams_full = current_reference()
    teams = teams_full[["id", "short_name"]]

    # latest form per player, including his most recent game; with
    # carry_over this reaches back into previous seasons
    snap = add_features(hist.copy(), cfg, shift=False)
    snap = snap[snap["season"] == CURRENT] if not cfg["carry_over"] else snap
    snap = snap.groupby("code").tail(1)
    games = (hist[hist["season"] == CURRENT].groupby("element").size()
             .rename("games_played"))
    snap = snap[["code"] + [f"r_{c}" for c in roll_cols()]]

    cur = hist[hist["season"] == CURRENT]
    ts = team_strength(cur, shift=False).groupby("team_id").tail(1)
    ts = ts[["team_id", "team_xg_for", "team_xg_against"]]

    nxt = fixtures.loc[~fixtures["finished"].astype(bool), "event"].min()
    upcoming = fixtures[fixtures["event"].between(nxt, nxt + HORIZON - 1)]
    home = upcoming.rename(columns={"team_h": "team", "team_a": "opponent_team"})
    home["was_home"] = 1
    away = upcoming.rename(columns={"team_a": "team", "team_h": "opponent_team"})
    away["was_home"] = 0
    fx = pd.concat([home, away]).rename(columns={"id": "fixture"})
    fx = fx[["fixture", "event", "team", "opponent_team", "was_home"]]

    p = players[["id", "code", "web_name", "element_type", "team", "now_cost",
                 "status", "chance_of_playing_next_round"]].copy()
    p = p.rename(columns={"id": "element", "now_cost": "value"})
    p["position"] = p["element_type"].map({v: k for k, v in POS.items()})
    rows = (p.merge(fx, on="team").merge(snap, on="code", how="left")
              .merge(games, left_on="element", right_index=True, how="left")
              .merge(ts.rename(columns={"team_id": "opponent_team",
                                        "team_xg_for": "opp_xg_for",
                                        "team_xg_against": "opp_xg_against"}),
                     on="opponent_team", how="left"))
    rows["games_played"] = rows["games_played"].fillna(0)
    if USE_ODDS:
        rows = odds_loader.to_player_rows(rows.assign(season=CURRENT),
                                          upcoming_odds(upcoming, teams_full, quiet))
    rows = prep(rows).reset_index(drop=True)
    # tiny negative predictions are noise for fringe players; floor at 0
    rows["pred"] = np.clip(model.predict(rows), 0, None)

    avail = rows["chance_of_playing_next_round"].fillna(100) / 100
    avail[rows["status"].isin(["u", "n"])] = 0  # left club / ineligible
    rows["pred"] *= avail

    # one column per gameweek (double gameweeks summed, blanks zero),
    # one row per player, including players with no fixture at all
    events = list(range(int(nxt), int(nxt) + HORIZON))
    per_gw = (rows.groupby(["element", "event"])["pred"].sum().unstack()
                  .reindex(columns=events).fillna(0))
    per_gw.columns = [f"gw{e}" for e in events]
    out = p[p["position"].notna()][["element", "web_name", "position",
                                     "team", "value"]]
    out = out.merge(per_gw, left_on="element", right_index=True, how="left")
    gw_cols = list(per_gw.columns)
    out[gw_cols] = out[gw_cols].fillna(0)
    out["pred"] = out[gw_cols].sum(axis=1)
    if dist_model is not None:
        dist = ranges.with_availability(dist_model.predict(rows), avail)
        rng = ranges.per_gameweek(rows[["element", "event"]], dist, events)
        out = out.merge(rng, on="element", how="left")
        rcols = [c for c in rng.columns if c != "element"]
        out[rcols] = out[rcols].fillna(0)
    out = out.merge(teams.rename(columns={"id": "team"}), on="team")
    if not quiet:
        print(f"Forecast covers gameweeks {events[0]} to {events[-1]}")
    return out.reset_index(drop=True), gw_cols


def show_team(preds, result):
    print(optimiser.describe_lineup(preds, result).to_string(index=False))
    show_captains(preds, result)
    print("\nPredicted team points (captain counted twice):")
    pts = optimiser.describe_points(result)
    print(pts.to_string(index=False))
    print(f"Total over the {len(pts)} gameweeks: "
          f"{pts['team points'].sum():.1f}")


def show_captains(preds, result, n=5):
    """The XI's top scorers for the first gameweek, with ranges. A player
    with a similar average but a higher ceiling or haul chance can be the
    better captain, since the captain's points count twice."""
    g = result["lineups"][0]["gw"]
    if f"low_{g}" not in preds:
        return
    xi = preds[preds["element"].isin(result["lineups"][0]["start"])]
    top = xi.sort_values(g, ascending=False).head(n)
    t = pd.DataFrame({"player": top["web_name"], "club": top["short_name"],
                      "predicted": top[g].round(1),
                      "80% range": [f"{int(a)}-{int(b)}" for a, b in
                                    zip(top[f"low_{g}"], top[f"high_{g}"])],
                      "haul (10+) %": (100 * top[f"haul_{g}"]).round(0).astype(int)})
    print(f"\nCaptain options, gameweek {g[2:]} (80% range: he should score "
          "below it about 1 week in 10 and above it about 1 week in 10):")
    print(t.to_string(index=False))


OUTPUT = Path("output")


def _run_info(gw_cols):
    """When this forecast was made, and the deadline of its first gameweek,
    so track.py can ignore forecasts made after a deadline had passed."""
    first = int(gw_cols[0][2:])
    deadline = None
    if LIVE_CURRENT:
        ev = {e["id"]: e for e in fpl_api.bootstrap()["events"]}
        deadline = ev.get(first, {}).get("deadline_time")
    return {"season": CURRENT, "gameweek": first,
            "made_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "deadline": deadline}


def save_predictions(preds, gw_cols):
    """Every player's predictions, for checking alternatives yourself and
    for track.py. Includes FPL's own expected points (ep_next) as a
    benchmark, since it can't be recovered after the deadline."""
    OUTPUT.mkdir(exist_ok=True)
    info = _run_info(gw_cols)
    f = OUTPUT / f"predictions_{CURRENT}_gw{info['gameweek']}.csv"
    df = preds.copy()
    df["fpl_ep_next"] = np.nan
    if LIVE_CURRENT:
        ep = fpl_api.players_raw().set_index("id")["ep_next"]
        df["fpl_ep_next"] = df["element"].map(pd.to_numeric(ep, errors="coerce"))
    df["price"] = df["value"] / 10
    df["made_at"], df["deadline"] = info["made_at"], info["deadline"]
    rcols = [f"{k}_{g}" for g in gw_cols for k in ("low", "high", "haul")
             if f"{k}_{g}" in df]
    cols = (["element", "web_name", "short_name", "position", "price"]
            + gw_cols + ["pred"] + rcols
            + ["fpl_ep_next", "made_at", "deadline"])
    (df[cols].rename(columns={"web_name": "player", "short_name": "club",
                              "pred": "total"})
       .sort_values("total", ascending=False).round(3)
       .to_csv(f, index=False))
    print(f"\nAll players' predictions saved to {f}")


def save_shadow(model, hist, table):
    """A second forecast without the odds inputs, saved alongside, so
    track.py can compare the two on this season's games. Remove once the
    comparison has run for 5 to 10 gameweeks."""
    global USE_ODDS
    USE_ODDS = False
    try:
        shadow = PointsModel(CONFIG).fit(table)
    finally:
        USE_ODDS = True
    preds, gw_cols = forecast(shadow, hist, CONFIG, quiet=True)
    info = _run_info(gw_cols)
    df = preds[["element"] + gw_cols].assign(made_at=info["made_at"],
                                             deadline=info["deadline"])
    f = OUTPUT / f"shadow_no-odds_{CURRENT}_gw{info['gameweek']}.csv"
    df.round(3).to_csv(f, index=False)


def save_plan(result, gw_cols, mode, team=None, options=None):
    """The recommended squad, line-ups and transfers, for track.py."""
    OUTPUT.mkdir(exist_ok=True)
    info = _run_info(gw_cols)
    squad = [int(e) for e in result["squad"]["element"]]
    plan = {**info, "mode": mode, "squad": squad,
            "lineups": [{"gameweek": int(lu["gw"][2:]),
                         "start": sorted(int(e) for e in lu["start"]),
                         "captain": int(lu["captain"]),
                         "predicted": round(float(lu["points"]), 2)}
                        for lu in result["lineups"]]}
    if team is not None:
        plan["transfers_out"] = [int(e) for e in team["squad"] if e not in squad]
        plan["transfers_in"] = [e for e in squad if e not in team["squad"]]
        plan["hits"] = int(result["hits"])
        plan["options"] = [{"transfers": r["k"], "net": round(r["net"], 2),
                            "gain": round(r["gain"], 2)} for r in options]
    f = OUTPUT / f"plan_{CURRENT}_gw{info['gameweek']}.json"
    f.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def main():
    global USE_UNDERSTAT, USE_ODDS
    import argparse
    ap = argparse.ArgumentParser(description="FPL squad picker")
    ap.add_argument("--no-understat", action="store_true",
                    help="FPL data only (faster; use for a first test)")
    ap.add_argument("--no-odds", action="store_true",
                    help="leave out betting odds (if football-data.co.uk is down)")
    ap.add_argument("--team-id", type=int,
                    help="your FPL team ID: suggests transfers instead of "
                         "picking a squad from scratch")
    ap.add_argument("--free-transfers", type=int,
                    help="override the estimated number of free transfers")
    ap.add_argument("--bank", type=float,
                    help="override money in the bank, in £m (e.g. 1.5)")
    ap.add_argument("--max-transfers", type=int, default=3,
                    help="largest number of transfers to consider (default 3)")
    args = ap.parse_args()
    if args.no_understat:
        USE_UNDERSTAT = False
    if args.no_odds:
        USE_ODDS = False

    hist = load_history(TRAIN_SEASONS + [CURRENT])
    table = build_training_table(hist, CONFIG)
    model = PointsModel(CONFIG).fit(table)
    dist = ranges.PointsDistribution(
        [f for f in features() if table[f].notna().any()]).fit(table)
    preds, gw_cols = forecast(model, hist, CONFIG, dist_model=dist)
    pd.set_option("display.width", 200)

    if args.team_id is None:
        r = optimiser.from_scratch(preds, gw_cols, bench_weight=BENCH_WEIGHT,
                                   budget=BUDGET)
        print(f"\nBest squad from scratch (e.g. for a Wildcard), line-up for "
              f"gameweek {gw_cols[0][2:]}:")
        show_team(preds, r)
        print(f"Squad cost: £{r['squad']['value'].sum() / 10:.1f}m")
        save_predictions(preds, gw_cols)
        if USE_ODDS:
            save_shadow(model, hist, table)
        save_plan(r, gw_cols, "from scratch")
        return

    import my_team
    team = my_team.load_my_team(args.team_id, fpl_api.players_raw())
    if args.free_transfers is not None:
        team["free_transfers"] = args.free_transfers
    if args.bank is not None:
        team["bank"] = round(args.bank * 10)
    print(f"\nYour squad after gameweek {team['event']}: "
          f"bank £{team['bank'] / 10:.1f}m, free transfers "
          f"{team['free_transfers']}"
          + ("" if args.free_transfers is not None else " (estimated)"))
    for note in team["notes"]:
        print("Note:", note)

    options = optimiser.transfer_options(
        preds, gw_cols, team, max_transfers=args.max_transfers,
        bench_weight=BENCH_WEIGHT)
    print(f"\nOptions over gameweeks {gw_cols[0][2:]} to {gw_cols[-1][2:]} "
          f"(points after hits; gain is versus making no transfers):")
    for r in options:
        sold, bought = optimiser.describe_transfers(preds, team, r)
        print(f"  {r['k']} transfer(s): {r['net']:6.1f} pts  gain {r['gain']:+5.1f}"
              f"  hits -{r['hits']}\n      out: {sold}\n      in:  {bought}")
    best = max(options, key=lambda r: r["net"])
    print(f"\nHighest predicted total: {best['k']} transfer(s). "
          f"Line-up for gameweek {gw_cols[0][2:]} under that plan:")
    show_team(preds, best)
    if best["hits"]:
        print(f"Hits of -{best['hits']} are not included in the team points.")
    save_predictions(preds, gw_cols)
    if USE_ODDS:
        save_shadow(model, hist, table)
    save_plan(best, gw_cols, "transfers", team, options)
    print("\nSmall gains are within the model's error; holding a free "
          "transfer for next week has value the model doesn't count.")


if __name__ == "__main__":
    main()
