"""
Walk-forward tuning for fpl_model.py.

For each test gameweek, the model is trained only on matches played
before that gameweek, then scored on it, as it would be in real use.
Settings are chosen on one season and then checked on the next, so the
final numbers aren't flattered by the tuning itself.

python tune.py                  # with Understat
python tune.py --no-understat   # FPL data only (faster)
"""
import argparse
import time

import numpy as np
import pandas as pd

import fpl_model as fm

TUNE_SEASON, TEST_SEASON = "2024-25", "2025-26"
FOLD_GWS = [6, 12, 18, 24, 30, 36]   # gameweeks predicted in each season
POS_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

_tables = {}


def table(hist, cfg):
    """Feature table, cached because only form settings change it."""
    key = (cfg["form"], cfg["halflife"], cfg["window"], cfg["carry_over"])
    if key not in _tables:
        _tables[key] = fm.build_training_table(hist, cfg)
    return _tables[key]


def walk_forward(hist, cfg, season):
    df = table(hist, cfg)
    preds = []
    for gw in FOLD_GWS:
        test = df[(df["season"] == season) & (df["round"] == gw)]
        train = df[df["kickoff_time"] < test["kickoff_time"].min()]
        model = fm.PointsModel(cfg).fit(train)
        preds.append(test.assign(pred=model.predict(test)))
    return pd.concat(preds)


def score(p):
    """RMSE is the main measure: the model is trained to predict the mean
    (expected points), and RMSE rewards exactly that. MAE rewards the
    median instead, which for FPL is close to 2 points for almost every
    regular, so it can favour a model that is worse at spotting hauls."""
    err = p["pred"] - p["total_points"]
    reg = p["regular"]
    rmse = lambda e: float(np.sqrt((e ** 2).mean()))
    out = {"RMSE regulars": rmse(err[reg]), "MAE regulars": err[reg].abs().mean(),
           "RMSE all": rmse(err)}
    for k, name in POS_NAMES.items():
        out[f"{name} RMSE"] = rmse(err[reg & (p["pos"] == k)])
    return out


def run(hist, cfg, season, label, results):
    t = time.time()
    r = score(walk_forward(hist, cfg, season))
    results.append({"season": season, "config": label, **r})
    print(f"  {label:42s} RMSE regulars {r['RMSE regulars']:.3f}"
          f"   all {r['RMSE all']:.3f}   ({time.time() - t:.0f}s)", flush=True)
    return r["RMSE regulars"]


def baseline(hist, season):
    """Each player's average points over his previous five games."""
    cfg = {**fm.CONFIG, "form": "rolling", "window": 5, "carry_over": False}
    p = table(hist, cfg)
    p = p[(p["season"] == season) & p["round"].isin(FOLD_GWS)].copy()
    p["pred"] = p["r_total_points"].fillna(0)
    return score(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-understat", action="store_true")
    if ap.parse_args().no_understat:
        fm.USE_UNDERSTAT = False

    seasons = [s for s in fm.TRAIN_SEASONS if s <= TEST_SEASON]
    hist = fm.load_history(seasons)
    results = []
    original = {**fm.CONFIG, "form": "rolling", "window": 5,
                "carry_over": False, "sample_halflife": None,
                "per_position": False, "learning_rate": 0.05,
                "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 20}

    print(f"Step 1: how player form is measured ({TUNE_SEASON})")
    forms = {
        "rolling 5, reset each season": dict(form="rolling", window=5, carry_over=False),
        "rolling 5, carried over": dict(form="rolling", window=5, carry_over=True),
        "rolling 10, carried over": dict(form="rolling", window=10, carry_over=True),
        "ewm half-life 2, carried over": dict(form="ewm", halflife=2, carry_over=True),
        "ewm half-life 4, carried over": dict(form="ewm", halflife=4, carry_over=True),
        "ewm half-life 8, carried over": dict(form="ewm", halflife=8, carry_over=True),
    }
    best = min(forms, key=lambda k: run(hist, {**original, **forms[k]},
                                         TUNE_SEASON, k, results))
    cfg = {**original, **forms[best]}

    print(f"Step 2: down-weighting older training matches ({TUNE_SEASON})")
    opts = {"equal weight": None, "half-life 2 years": 730,
            "half-life 1 year": 365}
    scores = {k: run(hist, {**cfg, "sample_halflife": v}, TUNE_SEASON,
                     f"training rows: {k}", results) for k, v in opts.items()}
    cfg["sample_halflife"] = opts[min(scores, key=scores.get)]

    print(f"Step 3: shared model vs one per position ({TUNE_SEASON})")
    scores = {v: run(hist, {**cfg, "per_position": v}, TUNE_SEASON,
                     "one model per position" if v else "one shared model",
                     results) for v in (False, True)}
    cfg["per_position"] = min(scores, key=scores.get)

    print(f"Step 4: tree settings ({TUNE_SEASON})")
    grid = [dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20),
            dict(learning_rate=0.03, max_iter=600, max_leaf_nodes=15, min_samples_leaf=100),
            dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200),
            dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=100),
            dict(learning_rate=0.03, max_iter=600, max_leaf_nodes=63, min_samples_leaf=50)]
    scores = [run(hist, {**cfg, **g}, TUNE_SEASON,
                  f"lr {g['learning_rate']}, {g['max_iter']} trees, "
                  f"{g['max_leaf_nodes']} leaves, min leaf {g['min_samples_leaf']}",
                  results) for g in grid]
    cfg.update(grid[int(np.argmin(scores))])

    print(f"Check on unseen season ({TEST_SEASON})")
    run(hist, original, TEST_SEASON, "original settings", results)
    run(hist, cfg, TEST_SEASON, "tuned settings", results)
    flipped = {**cfg, "per_position": not cfg["per_position"]}
    run(hist, flipped, TEST_SEASON,
        "tuned, but " + ("shared model" if cfg["per_position"]
                         else "one model per position"), results)
    b = baseline(hist, TEST_SEASON)
    results.append({"season": TEST_SEASON, "config": "baseline: last-5 average", **b})

    res = pd.DataFrame(results).round(3)
    print("\nErrors on", TEST_SEASON, "(points per player per match; regulars = 60+ mins avg over previous 3)")
    print(res[res["season"] == TEST_SEASON].drop(columns="season")
             .to_string(index=False))
    res.to_csv("tuning_results.csv", index=False)
    print("\nBest settings, to paste into CONFIG in fpl_model.py:")
    print(cfg)


if __name__ == "__main__":
    main()
