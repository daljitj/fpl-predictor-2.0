"""
Current-season data straight from the official FPL API.

The GitHub archive used for past seasons can lag weeks behind, so the
current season is rebuilt here in the same format: one row per player
per match, with the same column names as the archive's merged_gw.csv.
This mirrors how the archive itself is built from the same endpoints.

Responses are cached in CACHE_DIR and reused for MAX_AGE_HOURS, so
re-running the model on the same day doesn't download everything again.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://fantasy.premierleague.com/api"
CACHE_DIR = Path("fpl_api_cache")
MAX_AGE_HOURS = 6
DELAY = 0.2  # seconds between uncached player requests
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (personal FPL model)"


def _get_json(path, cache_name, max_age_hours=MAX_AGE_HOURS):
    f = CACHE_DIR / cache_name
    if f.exists() and time.time() - f.stat().st_mtime < max_age_hours * 3600:
        return json.loads(f.read_text(encoding="utf-8"))
    time.sleep(DELAY)
    r = _session.get(f"{API}/{path}", timeout=30)
    r.raise_for_status()
    data = r.json()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data), encoding="utf-8")
    return data


def bootstrap():
    return _get_json("bootstrap-static/", "bootstrap.json")


def players_raw():
    """Same role as the archive's players_raw.csv: prices, status, code."""
    return pd.DataFrame(bootstrap()["elements"])


def teams():
    return pd.DataFrame(bootstrap()["teams"])


def fixtures():
    fx = pd.DataFrame(_get_json("fixtures/", "fixtures.json"))
    return fx[fx["event"].notna()]  # postponed games have no gameweek yet


def gameweek_history():
    """Every player's match-by-match record this season, as merged_gw.csv."""
    players = players_raw()
    players = players[players["element_type"].isin(POSITIONS)]
    team_name = dict(zip(teams()["id"], teams()["name"]))
    fx = fixtures().set_index("id")

    rows = []
    n = len(players)
    for i, p in enumerate(players.itertuples(), 1):
        if i % 100 == 0 or i == n:
            print(f"  FPL API: player histories {i}/{n}", flush=True)
        hist = _get_json(f"element-summary/{p.id}/",
                         f"players/{p.id}.json")["history"]
        for h in hist:
            side = "team_h" if h["was_home"] else "team_a"
            h["team"] = team_name[fx.at[h["fixture"], side]]
            h["name"] = f"{p.first_name} {p.second_name}"
            h["position"] = POSITIONS[p.element_type]
            rows.append(h)

    gw = pd.DataFrame(rows)
    num = [c for c in gw.columns
           if c not in ("name", "position", "team", "kickoff_time", "was_home")]
    gw[num] = gw[num].apply(pd.to_numeric, errors="coerce")
    return gw


# ---------------------------------------------------------------- your team
# Public endpoints: no login needed, just your team ID (the number in the
# address bar on the FPL site's Points page). Cached for 15 minutes only.
TEAM_MAX_AGE = 0.25


def current_event():
    """Latest gameweek whose deadline has passed."""
    events = bootstrap()["events"]
    started = [e["id"] for e in events if e.get("is_current") or e.get("finished")]
    return max(started) if started else 0


def entry_picks(team_id, event):
    return _get_json(f"entry/{team_id}/event/{event}/picks/",
                     f"entry/{team_id}_picks_{event}.json", TEAM_MAX_AGE)


def entry_history(team_id):
    return _get_json(f"entry/{team_id}/history/",
                     f"entry/{team_id}_history.json", TEAM_MAX_AGE)


def entry_transfers(team_id):
    return _get_json(f"entry/{team_id}/transfers/",
                     f"entry/{team_id}_transfers.json", TEAM_MAX_AGE)


def finished_events():
    """Gameweeks FPL has finished and checked."""
    return sorted(e["id"] for e in bootstrap()["events"]
                  if e.get("finished") and e.get("data_checked", True))


def event_live(event):
    """Every player's actual points and minutes in a finished gameweek.
    Results don't change once checked, so the cache is kept for a year."""
    data = _get_json(f"event/{event}/live/", f"live/event_{event}.json",
                     max_age_hours=24 * 365)
    return pd.DataFrame([{"element": e["id"],
                          "minutes": e["stats"]["minutes"],
                          "actual": e["stats"]["total_points"]}
                         for e in data["elements"]])
