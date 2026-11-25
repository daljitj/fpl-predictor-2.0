"""
Is it time to send this gameweek's suggestions?

Used by the GitHub Actions workflow, which checks twice a day. Due when
the next deadline is between MIN_HOURS and MAX_HOURS away and no plan
for that gameweek was saved before its deadline (so a manual run, or an
earlier automatic one, isn't repeated).

Writes due=true/false and gameweek=N to $GITHUB_OUTPUT when run in
Actions, and always prints the reason.

poetry run python due.py           # check
poetry run python due.py --force   # due regardless of the window
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import fpl_api
from fpl_model import CURRENT, OUTPUT

MIN_HOURS = 3    # GitHub's scheduled runs can start up to an hour late
MAX_HOURS = 20   # a Saturday 11:00 deadline is sent the evening before


def next_deadline(events):
    """(gameweek, deadline) of the first gameweek whose deadline is ahead."""
    now = datetime.now(timezone.utc)
    for e in sorted(events, key=lambda e: e["id"]):
        d = datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
        if d > now:
            return e["id"], d
    return None, None


def already_planned(gw, deadline):
    f = OUTPUT / f"plan_{CURRENT}_gw{gw}.json"
    if not f.exists():
        return False
    made = json.loads(f.read_text(encoding="utf-8"))["made_at"]
    return datetime.fromisoformat(made) < deadline


def check(force=False):
    gw, deadline = next_deadline(fpl_api.bootstrap()["events"])
    if gw is None:
        return False, None, "No deadlines left this season."
    hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600
    when = f"GW{gw} deadline in {hours:.1f} hours"
    if force:
        return True, gw, f"{when}; forced."
    if already_planned(gw, deadline):
        return False, gw, f"{when}; plan already saved for GW{gw}."
    if not MIN_HOURS <= hours <= MAX_HOURS:
        return False, gw, (f"{when}; sends between {MIN_HOURS} and "
                           f"{MAX_HOURS} hours before.")
    return True, gw, f"{when}; running."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    due, gw, reason = check(ap.parse_args().force)
    print(reason)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with Path(out).open("a", encoding="utf-8") as f:
            f.write(f"due={'true' if due else 'false'}\ngameweek={gw or ''}\n")


if __name__ == "__main__":
    main()
