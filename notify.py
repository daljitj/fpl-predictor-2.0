"""
Send this gameweek's suggestions to your phone with ntfy (https://ntfy.sh).

Reads the plan and predictions fpl_model.py saved in output/, builds a
short message (transfers, captain, XI, bench, flags, last week's score)
and posts it to your private ntfy topic. Install the ntfy app on your
phone and subscribe to the same topic name.

The topic name comes from the NTFY_TOPIC environment variable (a GitHub
secret in the workflow). Anyone who knows it can read the messages, so
use a long random name.

poetry run python notify.py --dry-run      # print the message, send nothing
poetry run python notify.py                # send (needs NTFY_TOPIC)
poetry run python notify.py --gameweek 12  # a particular gameweek
"""
import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from fpl_model import CURRENT, OUTPUT

NTFY = "https://ntfy.sh"
UK = ZoneInfo("Europe/London")
POS_ORDER = ["GK", "DEF", "MID", "FWD"]


def latest_gameweek():
    gws = [int(f.stem.split("_gw")[1])
           for f in OUTPUT.glob(f"plan_{CURRENT}_gw*.json")]
    if not gws:
        raise SystemExit("No saved plan in output/. Run fpl_model.py first.")
    return max(gws)


def load(gw):
    plan = json.loads((OUTPUT / f"plan_{CURRENT}_gw{gw}.json")
                      .read_text(encoding="utf-8"))
    preds = pd.read_csv(OUTPUT / f"predictions_{CURRENT}_gw{gw}.csv")
    return plan, preds.set_index("element")


def money(tenths):
    return f"£{tenths / 10:.1f}m"


def uk_time(iso):
    if not iso:
        return "unknown"
    t = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(UK)
    return t.strftime("%a %d %b %H:%M")


def name(preds, e):
    return preds.at[e, "player"] if e in preds.index else f"player {e}"


def player_line(preds, e, gw):
    """'Haaland 6.1 pts (2-13, haul 14%)'"""
    p = preds.loc[e]
    line = f"{p['player']} {p[f'gw{gw}']:.1f} pts"
    if f"low_gw{gw}" in preds:
        line += (f" ({int(p[f'low_gw{gw}'])}-{int(p[f'high_gw{gw}'])}, "
                 f"haul {100 * p[f'haul_gw{gw}']:.0f}%)")
    return line


def availability_flags(elements):
    """Players with an injury, suspension or doubt flag, from the FPL API.
    Skipped quietly if the API can't be reached."""
    try:
        import fpl_api
        raw = fpl_api.players_raw().set_index("id")
    except Exception:
        return []
    out = []
    for e in elements:
        if e not in raw.index:
            continue
        r = raw.loc[e]
        chance = r.get("chance_of_playing_next_round")
        if r.get("status", "a") != "a" or (pd.notna(chance) and chance < 100):
            pct = "" if pd.isna(chance) else f" {int(chance)}%"
            news = f" ({r['news']})" if r.get("news") else ""
            out.append(f"{r['web_name']}{pct}{news}")
    return out


def last_week_line():
    f = OUTPUT / f"tracking_teams_{CURRENT}.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    t = t[t["gameweek"].astype(str) != "total"]
    if t.empty:
        return None
    r = t.iloc[-1]
    line = (f"Last scored GW{r['gameweek']}: suggested XI "
            f"{r['suggested: actual']:.0f} pts (predicted "
            f"{r['suggested: predicted']:.0f})")
    if "yours: actual" in t and pd.notna(r["yours: actual"]):
        line += f", your team {r['yours: actual']:.0f}"
    return line


def build_message(plan, preds):
    gw = plan["gameweek"]
    lu = plan["lineups"][0]
    gws = [x["gameweek"] for x in plan["lineups"]]
    span = f"GW{gws[0]}-{gws[-1]}" if len(gws) > 1 else f"GW{gws[0]}"
    lines = []

    head = f"Deadline {uk_time(plan.get('deadline'))}"
    if "free_transfers" in plan:
        head += (f". {plan['free_transfers']} free transfer(s), "
                 f"{money(plan['bank'])} in bank (estimated)")
    lines.append(head + ".")
    lines.append("")

    n_tr = len(plan.get("transfers_in", []))
    sell = plan.get("sell_prices", {})
    pos = lambda e: (POS_ORDER.index(preds.at[e, "position"])
                     if e in preds.index else 9)

    def swap_lines(outs, ins):
        """'OUT a (£x) > IN b (£y)', pairing players of the same position."""
        out = []
        for o, i in zip(sorted(outs, key=pos), sorted(ins, key=pos)):
            out_p = f" ({money(sell[str(o)])})" if str(o) in sell else ""
            in_p = (f" (£{preds.at[i, 'price']:.1f}m)"
                    if i in preds.index else "")
            out.append(f"  OUT {name(preds, o)}{out_p} > "
                       f"IN {name(preds, i)}{in_p}")
        return out

    options = plan.get("options", [])
    if plan.get("mode") != "transfers":
        lines.append(f"Best squad from scratch for {span}.")
    elif options and "lineup" in options[0]:
        # one block per option, each with its own GW line-up
        for o in options:
            k = o["transfers"]
            mark = "RECOMMENDED: " if k == n_tr else ""
            hit = f", hit -{o['hits']}" if o.get("hits") else ""
            what = "no transfers" if k == 0 else (
                f"{k} transfer{'s' if k > 1 else ''}")
            lines.append(f"{mark}{what}: {o['net']:.1f} pts over {span} "
                         f"(gain {o['gain']:+.1f}{hit})")
            lines += swap_lines(o["out"], o["in"])
            lo = o["lineup"]
            lines.append(f"  GW{gw} XI: {lo['xi']}")
            lines.append(f"  Bench: {lo['bench']}")
            lines.append("")
    else:
        # plans saved before per-option line-ups were added
        hits = f", hit -{plan['hits']}" if plan.get("hits") else ""
        lines.append(f"Suggested: {n_tr} transfer(s){hits}")
        lines += swap_lines(plan.get("transfers_out", []),
                            plan.get("transfers_in", []))
        lines.append("Options (transfers: pts over "
                     f"{span}): " + ", ".join(
                         f"{o['transfers']}: {o['net']:.1f}" for o in options))
        lines.append("")

    start = [e for e in lu["start"] if e in preds.index]
    cap = lu["captain"]
    vice = max((e for e in start if e != cap),
               key=lambda e: preds.at[e, f"gw{gw}"], default=None)
    lines.append(f"Captain (recommended XI): {player_line(preds, cap, gw)}")
    if vice is not None:
        lines.append(f"Vice: {player_line(preds, vice, gw)}")
    if not (options and "lineup" in options[0]):
        xi = preds.loc[start]
        for p_ in POS_ORDER:
            names = xi[xi["position"] == p_].sort_values(
                f"gw{gw}", ascending=False)["player"].tolist()
            if names:
                lines.append(f"{p_}: " + ", ".join(names))
    lines.append(f"Predicted GW{gw} points: {lu['predicted']:.0f} "
                 "(captain counted twice)")

    flags = availability_flags(start)
    inputs = plan.get("inputs", {})
    if inputs.get("understat") is False:
        flags.append("run without Understat")
    elif inputs.get("understat_stale"):
        flags.append("Understat down, latest matches without its data")
    if flags:
        lines.append("")
        lines.append("Check: " + "; ".join(flags))

    last = last_week_line()
    if last:
        lines.append("")
        lines.append(last)

    if n_tr == 0 or plan.get("mode") != "transfers":
        title = f"FPL GW{gw}: no transfers suggested" if plan.get(
            "mode") == "transfers" else f"FPL GW{gw}: wildcard squad"
    else:
        title = f"FPL GW{gw}: {n_tr} transfer{'s' if n_tr > 1 else ''} suggested"
    return title, "\n".join(lines)


def send(topic, title, body, click=None):
    headers = {"Title": title, "Tags": "soccer"}
    if click:
        headers["Click"] = click
    r = requests.post(f"{NTFY}/{topic}", data=body.encode("utf-8"),
                      headers=headers, timeout=30)
    r.raise_for_status()


def run_link():
    s, repo, run = (os.environ.get(k) for k in
                    ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{s}/{repo}/actions/runs/{run}" if s and repo and run else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gameweek", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan, preds = load(args.gameweek or latest_gameweek())
    title, body = build_message(plan, preds)
    if args.dry_run:
        print(title)
        print(body)
        return
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        sys.exit("Set NTFY_TOPIC to your ntfy topic name, or use --dry-run.")
    send(topic, title, body, run_link())
    print(f"Sent: {title}")


if __name__ == "__main__":
    main()
