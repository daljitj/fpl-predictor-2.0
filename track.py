"""
Live tracking: how did this season's forecasts and suggestions do?

Reads the files fpl_model.py saves in output/ each week, fetches actual
points for every finished gameweek from the FPL API, and reports:

* player accuracy, by how far ahead the forecast was (1, 2 or 3 weeks),
  by position and by gameweek, against FPL's own expected points
* calibration: do players predicted 4 points actually average 4?
* the suggested line-up's predicted and actual points, next to your
  own team's (with --team-id)
* ranges: how often actual points fell below, inside and above the 80%
  range, and predicted haul chances against actual hauls
* the model with betting odds against the same model without them, on
  the same players, while fpl_model.py saves the shadow forecasts

poetry run python track.py
poetry run python track.py --team-id 1234567

Only forecasts saved before a gameweek's deadline are used, so the
numbers aren't flattered by forecasts made with results already known.
"""
import argparse
import json

import numpy as np
import pandas as pd

import fpl_api
from fpl_model import CURRENT, OUTPUT

LIKELY = 2.0  # "likely starters": players predicted at least this many points


# ---------------------------------------------------------------- loading
def load_forecasts(prefix="predictions"):
    """One row per player per forecast gameweek, with the horizon."""
    frames, skipped = [], []
    for f in sorted(OUTPUT.glob(f"{prefix}_{CURRENT}_gw*.csv")):
        df = pd.read_csv(f)
        made = pd.to_datetime(df["made_at"].iloc[0], utc=True)
        deadline = pd.to_datetime(df["deadline"].iloc[0], utc=True)
        if pd.notna(deadline) and made > deadline:
            skipped.append(f.name)
            continue
        gw_cols = [c for c in df.columns if c.startswith("gw")]
        first = int(gw_cols[0][2:])
        ids = [c for c in ["element", "player", "club", "position",
                           "fpl_ep_next"] if c in df]
        long = df.melt(id_vars=ids, value_vars=gw_cols, var_name="gw",
                       value_name="pred")
        for k in ("low", "high", "haul"):   # ranges, saved from 16 Oct 2026
            if f"{k}_{gw_cols[0]}" in df:
                r = df.melt(id_vars=["element"],
                            value_vars=[f"{k}_{g}" for g in gw_cols],
                            var_name="gw", value_name=k)
                r["gw"] = r["gw"].str[len(k) + 1:]
                long = long.merge(r, on=["element", "gw"], how="left")
        long["gameweek"] = long["gw"].str[2:].astype(int)
        long["horizon"] = long["gameweek"] - first + 1
        long["forecast_gw"] = first
        # FPL's ep_next only refers to the next gameweek
        if "fpl_ep_next" in long:
            long.loc[long["horizon"] != 1, "fpl_ep_next"] = np.nan
        frames.append(long.drop(columns="gw"))
    if skipped:
        print("Ignored (saved after the deadline):", ", ".join(skipped))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_actuals(fc):
    done = set(fpl_api.finished_events())
    fc = fc[fc["gameweek"].isin(done)]
    if fc.empty:
        return fc
    live = pd.concat([fpl_api.event_live(g).assign(gameweek=g)
                      for g in sorted(fc["gameweek"].unique())])
    return fc.merge(live, on=["element", "gameweek"], how="left").fillna(
        {"actual": 0, "minutes": 0})


# ---------------------------------------------------------------- accuracy
def metrics(d, col="pred"):
    e = d[col] - d["actual"]
    return pd.Series({
        "players": len(d),
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "bias": e.mean(),                      # + means predicted too high
        "correlation": d[col].corr(d["actual"]),
    })


def show(t):
    t = t.round(2)
    t["players"] = t["players"].astype(int)
    print(t.to_string())


def accuracy_report(df):
    likely = df[df["pred"] >= LIKELY]

    print(f"\nPLAYER ACCURACY (likely starters: predicted {LIKELY:.0f}+ points)")
    print("By forecast horizon:")
    t = likely.groupby("horizon").apply(metrics, include_groups=False)
    show(t)

    h1 = likely[likely["horizon"] == 1]
    bench = h1.dropna(subset=["fpl_ep_next"])
    if len(bench):
        print("\nOne week ahead, model vs FPL's own expected points "
              "(same players):")
        cmp = pd.DataFrame({"model": metrics(bench),
                            "FPL ep_next": metrics(bench, "fpl_ep_next")}).T
        show(cmp)

    print("\nOne week ahead, by position:")
    show(h1.groupby("position").apply(metrics, include_groups=False)
           .reindex(["GK", "DEF", "MID", "FWD"]).dropna(how="all"))

    print("\nOne week ahead, by gameweek:")
    show(h1.groupby("gameweek").apply(metrics, include_groups=False))

    print("\nCalibration, one week ahead, all players with a forecast:")
    a = df[df["horizon"] == 1].copy()
    a["predicted"] = pd.cut(a["pred"], [-0.01, 1, 2, 3, 4, 5, 6, 99],
                            labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6",
                                    "6+"])
    cal = a.groupby("predicted", observed=True).agg(
        players=("pred", "size"), mean_predicted=("pred", "mean"),
        mean_actual=("actual", "mean"))
    print(cal.round(2).to_string())
    return t, h1


def range_report(df):
    """Are the 80% ranges and haul chances honest? Likely starters only."""
    if "low" not in df:
        return None
    d = df.dropna(subset=["low"])
    d = d[d["pred"] >= LIKELY]
    if d.empty:
        return None
    act = d["actual"].clip(lower=0)   # ranges treat negative scores as 0
    d = d.assign(below=act < d["low"], above=act > d["high"],
                 hauled=act >= 10)
    t = d.groupby("horizon").agg(
        players=("pred", "size"),
        below_low=("below", "mean"), above_high=("above", "mean"),
        haul_predicted=("haul", "mean"), haul_actual=("hauled", "mean"))
    t.insert(2, "inside", 1 - t["below_low"] - t["above_high"])
    print("\nRANGES (likely starters): share of actual scores below, inside "
          "and above the 80% range, and haul (10+) chance vs actual")
    out = t.copy()
    for c in ["below_low", "inside", "above_high", "haul_predicted", "haul_actual"]:
        out[c] = (100 * out[c]).round(1).astype(str) + "%"
    print(out.to_string())
    print("About 10% above the range is the target. Below the range is "
          "usually rarer, because the low end is often 0 or 1. "
          "Inside the range ends up above 80% because scores are whole "
          "numbers and many land exactly on an end.")
    return t


def odds_report(df):
    """With odds vs without, same forecast runs and same players."""
    sh = load_forecasts("shadow_no-odds")
    if sh.empty:
        return None
    key = ["forecast_gw", "gameweek", "element"]
    both = df.merge(sh[key + ["pred"]].rename(columns={"pred": "no_odds"}),
                    on=key)
    both = both[both["pred"] >= LIKELY]
    if both.empty:
        return None
    print("\nBETTING ODDS: model with odds vs the same model without "
          "(likely starters, same players)")
    rows = []
    for h, d in both.groupby("horizon"):
        for name, col in (("with odds", "pred"), ("without odds", "no_odds")):
            rows.append({"horizon": h, "model": name, **metrics(d, col)})
    t = pd.DataFrame(rows).set_index(["horizon", "model"])
    show(t)
    print("Lower RMSE is better. Week 1 uses this week's odds; weeks 2 and "
          "3 use team ratings from recent odds.")
    return t


# ---------------------------------------------------------------- teams
def team_report(df, team_id):
    done = set(fpl_api.finished_events())
    one = df[df["horizon"] == 1].set_index(["gameweek", "element"])
    rows = []
    for f in sorted(OUTPUT.glob(f"plan_{CURRENT}_gw*.json")):
        plan = json.loads(f.read_text(encoding="utf-8"))
        lu = plan["lineups"][0]
        gw = lu["gameweek"]
        if gw not in done or (gw, lu["captain"]) not in one.index:
            continue
        made = pd.to_datetime(plan["made_at"], utc=True)
        if plan["deadline"] and made > pd.to_datetime(plan["deadline"], utc=True):
            continue
        xi = one.loc[[(gw, e) for e in lu["start"]]]
        cap = one.loc[(gw, lu["captain"])]
        row = {"gameweek": gw,
               "suggested: predicted": lu["predicted"],
               "suggested: actual": xi["actual"].sum() + cap["actual"],
               "suggested: hits": plan.get("hits", 0)}

        if team_id:
            picks = fpl_api.entry_picks(team_id, gw)
            mult = {p["element"]: p["multiplier"] for p in picks["picks"]}
            pred = sum(m * one["pred"].get((gw, e), 0) for e, m in mult.items())
            hist = picks["entry_history"]
            row.update({"yours: predicted": pred,
                        "yours: actual": hist["points"],
                        "yours: hits": hist["event_transfers_cost"]})
        rows.append(row)

    if not rows:
        print("\nNo saved plans for finished gameweeks yet.")
        return None
    t = pd.DataFrame(rows).set_index("gameweek").round(1)
    t.loc["total"] = t.sum()
    print("\nTEAM POINTS (before hits; hits shown separately)")
    print(t.to_string())
    print("Suggested line-up actuals don't include automatic substitutions;"
          " your actual points do.")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", type=int,
                    help="also compare your own team's points")
    args = ap.parse_args()

    fc = load_forecasts()
    if fc.empty:
        print(f"No forecasts for {CURRENT} in {OUTPUT}/ yet. Run "
              "fpl_model.py before a deadline, then track.py once that "
              "gameweek has finished.")
        return
    df = attach_actuals(fc)
    if df.empty:
        print("None of the saved forecasts are for finished gameweeks yet.")
        return
    weeks = sorted(df["gameweek"].unique())
    print(f"Tracking {CURRENT}: {len(weeks)} finished gameweek(s) "
          f"({weeks[0]} to {weeks[-1]})")

    by_horizon, _ = accuracy_report(df)
    rng = range_report(df)
    odds = odds_report(df)
    teams = team_report(df, args.team_id)

    df.to_csv(OUTPUT / f"tracking_players_{CURRENT}.csv", index=False)
    by_horizon.to_csv(OUTPUT / f"tracking_summary_{CURRENT}.csv")
    if teams is not None:
        teams.to_csv(OUTPUT / f"tracking_teams_{CURRENT}.csv")
    if odds is not None:
        odds.to_csv(OUTPUT / f"tracking_odds_{CURRENT}.csv")
    if rng is not None:
        rng.to_csv(OUTPUT / f"tracking_ranges_{CURRENT}.csv")
    print(f"\nDetails saved in {OUTPUT}/ (tracking_*.csv)")
    print("A few gameweeks is a small sample: one haul can move these "
          "numbers a lot, so look at the trend over 5-10 weeks.")


if __name__ == "__main__":
    main()
