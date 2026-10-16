"""
Understat loader for the FPL model.

Downloads per-player, per-match Premier League stats from understat.com
(shots, key passes, xG, xA, xGChain, xGBuildup) plus every shot, then
matches Understat players to FPL players so the two can be joined.

Understat has no tackle or save data. Those come from the FPL data,
which the main script already loads.

Responses are cached in CACHE_DIR, so each match is downloaded once.
A first run takes roughly 7 minutes per season because of the polite
delay between requests; later runs read from the cache.
"""
import json
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://understat.com"
HEADERS = {"X-Requested-With": "XMLHttpRequest",
           "User-Agent": "Mozilla/5.0 (personal FPL model)"}
CACHE_DIR = Path("understat_cache")
DELAY = 1.0  # seconds between uncached requests

_session = None


# ---------------------------------------------------------------- download
def _get(path, cache_file, refresh=False):
    """GET an Understat JSON endpoint, using the disk cache if present."""
    global _session
    cache_file = CACHE_DIR / cache_file
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    if _session is None:
        _session = requests.Session()
        _session.get(URL, headers=HEADERS, timeout=30)  # sets cookies
    time.sleep(DELAY)
    r = _session.get(URL + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def fpl_to_understat_season(season):
    """'2025-26' -> 2025 (Understat labels a season by its first year)."""
    return int(season[:4])


def load_understat_season(season):
    """Return (player_match, shots) DataFrames for one EPL season."""
    year = fpl_to_understat_season(season)
    path, name = f"/getLeagueData/EPL/{year}", f"league_EPL_{year}.json"
    league = _get(path, name)
    if not all(m.get("isResult") for m in league["dates"]):
        league = _get(path, name, refresh=True)  # season in progress
    played = [m for m in league["dates"] if m.get("isResult")]

    players, shots = [], []
    for m in played:
        gid = int(m["id"])
        date = pd.to_datetime(m["datetime"]).normalize()
        data = _get(f"/getMatchData/{gid}", f"match_{gid}.json")
        team_name = {"h": m["h"]["title"], "a": m["a"]["title"]}

        for side, roster in data["rosters"].items():
            for p in roster.values():
                players.append({
                    "us_game_id": gid, "date": date,
                    "us_team": team_name[side], "us_home": side == "h",
                    "us_player": p["player"], "us_player_id": int(p["player_id"]),
                    "us_position": p["position"], "us_minutes": int(p["time"]),
                    "us_shots": int(p["shots"]),
                    "us_key_passes": int(p["key_passes"]),
                    "us_xg": float(p["xG"]), "us_xa": float(p["xA"]),
                    "us_xg_chain": float(p["xGChain"]),
                    "us_xg_buildup": float(p["xGBuildup"]),
                })

        for side, side_shots in data["shots"].items():
            for s in side_shots:
                shots.append({
                    "us_game_id": gid, "date": date,
                    "us_player_id": int(s["player_id"]),
                    "xg": float(s["xG"]), "x": float(s["X"]), "y": float(s["Y"]),
                    "situation": s["situation"], "shot_type": s["shotType"],
                    "result": s["result"],
                })

    return pd.DataFrame(players), pd.DataFrame(shots)


# ---------------------------------------------------------------- shot features
def shot_features(shots):
    """Per player per match: non-penalty xG, penalty xG, shots in the box,
    headed shots and set-piece xG. Understat coordinates run 0 to 1, with
    the attacking goal at x = 1; the penalty area is roughly x > 0.83 and
    0.21 < y < 0.79."""
    if shots.empty:
        return pd.DataFrame(columns=["us_game_id", "us_player_id"])
    s = shots.copy()
    pen = s["situation"] == "Penalty"
    s["us_npxg"] = np.where(pen, 0.0, s["xg"])
    s["us_pen_xg"] = np.where(pen, s["xg"], 0.0)
    s["us_box_shots"] = ((s["x"] > 0.83) & s["y"].between(0.21, 0.79)
                         & ~pen).astype(int)
    s["us_headers"] = (s["shot_type"] == "Head").astype(int)
    s["us_setpiece_xg"] = np.where(
        s["situation"].isin(["FromCorner", "SetPiece", "DirectFreekick"]),
        s["xg"], 0.0)
    cols = ["us_npxg", "us_pen_xg", "us_box_shots", "us_headers",
            "us_setpiece_xg"]
    return s.groupby(["us_game_id", "us_player_id"], as_index=False)[cols].sum()


# ---------------------------------------------------------------- matching
def _norm(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.lower().replace("-", " ").replace("'", "").strip()


def name_similarity(a, b):
    """0 to 1. Handles 'Mohamed Salah' vs 'Mohamed Salah Hamed Ghaly'
    and single-name players such as 'Reinildo' vs 'Reinildo Mandava'."""
    a, b = _norm(a), _norm(b)
    ratio = SequenceMatcher(None, a, b).ratio()
    ta = {t for t in a.split() if len(t) > 2}
    tb = {t for t in b.split() if len(t) > 2}
    # tokens match if equal or one is a 4+ letter prefix ('rodri'/'rodrigo')
    shared = sum(any(x == y or (min(len(x), len(y)) >= 4 and
                                (x.startswith(y) or y.startswith(x)))
                     for y in tb) for x in ta)
    if not shared:
        return ratio
    overlap = min(shared / min(len(ta), len(tb)), 1)
    jaccard = shared / (len(ta) + len(tb) - shared)  # separates two "Gabriel"s
    return max(ratio, 0.6 + 0.3 * overlap + 0.1 * jaccard)


def _tokens(names):
    return {t for n in names for t in _norm(n).split() if len(t) > 2}


def map_teams(fpl, us):
    """FPL team name -> Understat team name, without a hand-written table.
    For every match, compare the name tokens of each FPL side's players
    with those of each Understat side that played on the same date at the
    same venue; the pairing with the most shared names across all matches
    wins. Dates alone are ambiguous because many games share a date."""
    f = (fpl[fpl["minutes"] > 0]
         .groupby(["season", "date", "team", "was_home"])["name"]
         .agg(_tokens).reset_index())
    u = (us.groupby(["season", "date", "us_team", "us_home"])["us_player"]
           .agg(_tokens).reset_index())
    m = f.merge(u, left_on=["season", "date", "was_home"],
                right_on=["season", "date", "us_home"])
    m["n"] = [len(a & b) for a, b in zip(m["name"], m["us_player"])]
    score = m.groupby(["team", "us_team"])["n"].sum().reset_index()
    best = score.sort_values("n").groupby("team").tail(1)
    return dict(zip(best["team"], best["us_team"]))


def map_players(fpl, us, min_similarity=0.55):
    """(season, FPL element) -> Understat player_id.

    For every match an FPL player played in, compare his name with the
    Understat players on the same team on the same date, then take the
    most frequent best match across his matches. Voting across matches
    makes one bad comparison harmless and copes with mid-season moves.
    """
    web = fpl["web_name"].fillna(fpl["name"]) if "web_name" in fpl else fpl["name"]
    fpl = fpl.assign(web_name=web)
    played = fpl[fpl["minutes"] > 0][["season", "element", "name", "web_name",
                                       "date", "us_team", "minutes"]]
    cand = played.merge(us[["season", "date", "us_team", "us_player",
                            "us_player_id", "us_minutes"]],
                        on=["season", "date", "us_team"])
    # compare both the full name and the FPL display name, so players
    # known by a nickname match ('Carlos Henrique Casimiro' / 'Casemiro');
    # cache per unique pair, then penalise minute gaps
    key = ["name", "web_name", "us_player"]
    pairs = cand[key].drop_duplicates()
    pairs["sim"] = [max(name_similarity(a, c), name_similarity(b, c))
                    for a, b, c in zip(pairs["name"], pairs["web_name"],
                                       pairs["us_player"])]
    cand = cand.merge(pairs, on=key)
    cand["score"] = cand["sim"] - 0.002 * (cand["minutes"]
                                           - cand["us_minutes"]).abs()
    # greedy one-to-one within each match: best-scoring pairs first
    best = (cand.sort_values("score", ascending=False)
                .drop_duplicates(["season", "date", "us_player_id"])
                .drop_duplicates(["season", "element", "date"]))
    best = best[best["sim"] >= min_similarity]
    votes = (best.groupby(["season", "element", "us_player_id"])
                 .size().reset_index(name="n")
                 .sort_values("n").groupby(["season", "element"]).tail(1))
    return votes[["season", "element", "us_player_id"]]


# ---------------------------------------------------------------- join
US_COLS = ["us_shots", "us_key_passes", "us_xg", "us_xa", "us_xg_chain",
           "us_xg_buildup", "us_npxg", "us_pen_xg", "us_box_shots",
           "us_headers", "us_setpiece_xg"]


def add_understat(fpl):
    """Add Understat columns to an FPL gameweek DataFrame (one or more
    seasons, as loaded by fpl_model.load_season). Players who didn't
    play get zeros; players who played but couldn't be matched get NaN,
    which the gradient-boosted model handles natively."""
    frames = []
    for season in fpl["season"].unique():
        pm, sh = load_understat_season(season)
        if pm.empty:
            continue
        pm = pm.merge(shot_features(sh), on=["us_game_id", "us_player_id"],
                      how="left")
        pm["season"] = season
        frames.append(pm)
    out = fpl.copy()
    out["date"] = out["kickoff_time"].dt.tz_localize(None).dt.normalize()
    if not frames:
        for c in US_COLS:
            out[c] = np.nan
        return out
    us = pd.concat(frames, ignore_index=True)
    shot_cols = ["us_npxg", "us_pen_xg", "us_box_shots", "us_headers",
                 "us_setpiece_xg"]
    us[shot_cols] = us[shot_cols].fillna(0)  # rostered but no shots

    out["us_team"] = out["team"].map(map_teams(out, us))
    ids = map_players(out, us)
    out = out.merge(ids, on=["season", "element"], how="left")
    out = out.merge(us[["season", "date", "us_player_id"] + US_COLS],
                    on=["season", "date", "us_player_id"], how="left")
    out.loc[out["minutes"] == 0, US_COLS] = 0

    played = out["minutes"] > 0
    rate = out.loc[played, "us_shots"].notna().mean()
    print(f"Understat matched {rate:.1%} of FPL appearances")
    unmatched = (out[played & out["us_player_id"].isna()]
                 .groupby(["season", "name"])["minutes"].sum()
                 .sort_values(ascending=False).head(10))
    if len(unmatched):
        print("Largest unmatched (season, player, minutes):")
        print(unmatched.to_string())
    return out.drop(columns=["us_team"])
