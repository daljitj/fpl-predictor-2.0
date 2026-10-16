"""
Your current FPL squad, bank, selling prices and free transfers, from
the public FPL API (team ID only, no login).

Two things the public API doesn't show directly, so they're estimated:

* Selling prices. FPL pays the purchase price plus half of any rise,
  rounded down to £0.1m (a fall is passed on in full). Purchase prices
  come from your transfer history, or the start-of-season price for
  players you've had since your first gameweek.
* Free transfers. Rebuilt gameweek by gameweek from your history: one
  added per gameweek, up to five banked, unused ones carried over, and
  saved ones kept when a Wildcard or Free Hit is played.

Both are printed so you can check them against the FPL app, and both
can be overridden from the command line.
"""
import pandas as pd

import fpl_api

MAX_FREE_TRANSFERS = 5
CHIPS_KEEPING_TRANSFERS = {"wildcard", "freehit"}


def selling_price(bought, now):
    """In tenths of £1m, as the API uses."""
    if now <= bought:
        return now
    return bought + (now - bought) // 2


def estimate_free_transfers(history):
    """Replay the season's transfers to estimate this week's free ones."""
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    weeks = sorted(history["current"], key=lambda w: w["event"])
    ft = 1  # after your first gameweek, whose squad was picked for free
    for w in weeks[1:]:
        if chips.get(w["event"]) in CHIPS_KEEPING_TRANSFERS:
            ft = min(ft + 1, MAX_FREE_TRANSFERS)   # saved ones kept
        else:
            ft = min(max(ft - w["event_transfers"], 0) + 1, MAX_FREE_TRANSFERS)
    return ft


def load_my_team(team_id, players):
    """players: the bootstrap player table (id, now_cost, cost_change_start,
    web_name). Returns a dict describing your squad."""
    event = fpl_api.current_event()
    if event == 0:
        raise SystemExit("The season hasn't started, so there's no squad to "
                         "read yet. Run without --team-id to pick one.")
    history = fpl_api.entry_history(team_id)
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}

    # A Free Hit squad reverts afterwards, so read the gameweek before it
    pick_event = event
    while chips.get(pick_event) == "freehit" and pick_event > 1:
        pick_event -= 1
    picks = fpl_api.entry_picks(team_id, pick_event)
    squad = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"]

    now = dict(zip(players["id"], players["now_cost"]))
    start = dict(zip(players["id"],
                     players["now_cost"] - players["cost_change_start"]))

    # latest purchase of each player, ignoring Free Hit weeks
    tr = pd.DataFrame(fpl_api.entry_transfers(team_id))
    bought = {}
    if len(tr):
        tr = tr[~tr["event"].map(chips).eq("freehit")]
        tr = tr.sort_values("time")
        bought = dict(zip(tr["element_in"], tr["element_in_cost"]))

    sell, notes = {}, []
    for e in squad:
        if e in bought:
            sell[e] = selling_price(bought[e], now[e])
        else:
            sell[e] = selling_price(start[e], now[e])
    if history["current"] and history["current"][0]["event"] > 1:
        notes.append("You joined after gameweek 1, so purchase prices for "
                     "your original players use start-of-season prices.")
    if any(c in CHIPS_KEEPING_TRANSFERS for c in chips.values()):
        notes.append("You've played a Wildcard or Free Hit, which makes the "
                     "free transfer estimate less certain.")

    return {"event": event, "squad": squad, "bank": bank, "sell": sell,
            "free_transfers": estimate_free_transfers(history),
            "notes": notes}
