"""
Squad optimiser: an integer programme solved with HiGHS via scipy.

Decisions: which 15 players to own, and for each gameweek in the horizon
which 11 start and who captains. Picking a separate XI each week means a
player with a blank or a tough fixture can sit on the bench that week
rather than dragging the whole plan down.

Two modes:
* from scratch (first squad, or a Wildcard): any 15 within £100.0m
* transfers: start from your squad, make exactly k changes, paying
  selling prices for players sold and current prices for players bought
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

POS = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
SQUAD_N = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
START_RANGE = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
HIT = 4  # points per transfer beyond your free ones


def solve(df, gw_cols, bench_weight=0.1, budget=1000,
          current=None, sell=None, bank=None, n_transfers=None):
    """df: one row per player with 'element', 'position', 'team', 'value'
    (price in tenths) and one predicted-points column per gameweek."""
    n, G = len(df), len(gw_cols)
    P = df[gw_cols].to_numpy()                 # n x G predicted points
    X = np.arange(n)                           # squad variable indices
    Y = n + np.arange(G * n).reshape(G, n)     # starter, per gameweek
    C = n + G * n + np.arange(G * n).reshape(G, n)  # captain, per gameweek
    N = n * (1 + 2 * G)

    # objective (milp minimises, so negate): starters score, the captain
    # scores again, the bench counts a little in case of injuries
    obj = np.zeros(N)
    obj[X] = bench_weight * P.sum(axis=1)
    for g in range(G):
        obj[Y[g]] = P[:, g] * (1 - bench_weight)
        obj[C[g]] = P[:, g]

    rows, cols, vals, lo, hi = [], [], [], [], []
    def add(idx, coefs, l, h):
        r = len(lo)
        idx = np.asarray(idx)
        coefs = np.broadcast_to(np.asarray(coefs, float), idx.shape)
        rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(coefs)
        lo.append(l); hi.append(h)

    pos = df["position"].to_numpy()
    team = df["team"].to_numpy()

    add(X, 1, 15, 15)
    for p, k in SQUAD_N.items():
        add(X[pos == p], 1, k, k)
    for t in np.unique(team):                  # max three per club
        add(X[team == t], 1, 0, 3)
    for g in range(G):
        add(Y[g], 1, 11, 11)
        add(C[g], 1, 1, 1)
        for p, (a, b) in START_RANGE.items():
            add(Y[g][pos == p], 1, a, b)
        for i in range(n):                     # start only if owned,
            add([Y[g][i], X[i]], [1, -1], -np.inf, 0)
            add([C[g][i], Y[g][i]], [1, -1], -np.inf, 0)  # captain a starter

    price = df["value"].to_numpy().astype(float)
    if current is None:
        add(X, price, 0, budget)
    else:
        owned = df["element"].isin(current).to_numpy()
        s = df["element"].map(sell).to_numpy()
        cost = np.where(owned, s, price)       # keeping a player "costs" his
        add(X, cost, 0, bank + np.nansum(s))   # sale value; cancels out
        add(X[owned], 1, 15 - n_transfers, 15 - n_transfers)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lo), N)).tocsr()
    res = milp(-obj, constraints=LinearConstraint(A, lo, hi),
               integrality=np.ones(N), bounds=Bounds(0, 1))
    if not res.success:
        return None

    x = res.x.round().astype(bool)
    squad = df[x[X]]
    lineups = []
    points = 0.0
    for g, col in enumerate(gw_cols):
        start, capt = x[Y[g]], x[C[g]]
        gw_points = P[start, g].sum() + P[capt, g].sum()
        points += gw_points
        lineups.append({"gw": col, "start": set(df["element"][start]),
                        "captain": df["element"][capt].iloc[0],
                        "captain_name": df["web_name"][capt].iloc[0],
                        "points": gw_points,
                        "bench_points": P[x[X] & ~start, g].sum()})
    return {"squad": squad, "lineups": lineups, "points": points}


def from_scratch(df, gw_cols, **kw):
    return solve(df, gw_cols, **kw)


def transfer_options(df, gw_cols, team, max_transfers=3, **kw):
    """Best plan for 0, 1, ... max_transfers transfers, with hit costs."""
    out = []
    for k in range(max_transfers + 1):
        r = solve(df, gw_cols, current=team["squad"], sell=team["sell"],
                  bank=team["bank"], n_transfers=k, **kw)
        if r is None:
            continue
        hits = max(0, k - team["free_transfers"]) * HIT
        r.update(k=k, hits=hits, net=r["points"] - hits)
        out.append(r)
    base = out[0]["net"]
    for r in out:
        r["gain"] = r["net"] - base
    return out


# ---------------------------------------------------------------- printing
def describe_lineup(df, result, gw_index=0):
    """Squad table for one gameweek: starters first, captain marked, with
    each player's predicted points per gameweek and over the horizon."""
    lu = result["lineups"][gw_index]
    sq = result["squad"].copy()
    starter = sq["element"].isin(lu["start"])
    sq["role"] = np.where(starter, "XI", "bench")
    sq.loc[sq["element"] == lu["captain"], "role"] = "XI (C)"
    sq["order"] = (~starter) * 10 + sq["position"].map(POS)
    sq["price"] = sq["value"] / 10
    gws = [c for c in df.columns if c.startswith("gw")]
    sq["total"] = sq[gws].sum(axis=1)
    extra = []
    g = lu["gw"]
    if f"low_{g}" in sq:   # ranges from ranges.py, for this gameweek
        sq[f"range {g}"] = [f"{int(a)}-{int(b)}" for a, b in
                            zip(sq[f"low_{g}"], sq[f"high_{g}"])]
        sq["haul %"] = (100 * sq[f"haul_{g}"]).round(0).astype(int)
        extra = [f"range {g}", "haul %"]
    return (sq.sort_values("order")
              [["web_name", "short_name", "position", "price", "role"]
               + gws + ["total"] + extra]
              .rename(columns={"web_name": "player", "short_name": "club"})
              .round(1))


def describe_points(result):
    """Predicted team points per gameweek, each week with its own XI and
    captain (the captain's points are counted twice, as in FPL)."""
    rows = [{"gameweek": lu["gw"][2:], "captain": lu["captain_name"],
             "team points": round(lu["points"], 1),
             "bench (not counted)": round(lu["bench_points"], 1)}
            for lu in result["lineups"]]
    return pd.DataFrame(rows)


def describe_transfers(df, team, r):
    names = df.set_index("element")
    out = [e for e in team["squad"] if e not in set(r["squad"]["element"])]
    inn = [e for e in r["squad"]["element"] if e not in team["squad"]]
    sold = ", ".join(f"{names.at[e, 'web_name']} (£{team['sell'][e] / 10:.1f}m)"
                     for e in out) or "none"
    bought = ", ".join(f"{names.at[e, 'web_name']} (£{names.at[e, 'value'] / 10:.1f}m)"
                       for e in inn) or "none"
    return sold, bought
