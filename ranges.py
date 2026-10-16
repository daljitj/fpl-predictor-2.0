"""
Ranges of likely points, not just the average.

The main model predicts each player's expected (average) points. FPL
scores are lumpy: a regular midfielder might average 4.5 but score 2
most weeks and 13 now and then. This module predicts the whole spread.

How: a gradient-boosted classifier, on the same inputs as the main
model, gives the chance of each band of points in a match (0 or less,
1, 2, 3, 4-5, 6-7, 8-9, 10-12, 13+). Each band is split into single
scores using how often each score occurred in training (for example
how often 10, 11 and 12 occurred within 10-12). Then, per gameweek:

* a flagged player (e.g. 75% chance of playing) is a mix: zero with
  the remaining chance, the forecast otherwise, as in the main model
* a double gameweek adds the two matches' scores (treated as
  independent, which is approximate)

From that distribution come the low and high ends of the 80% range
(the 10th and 90th percentiles: he should score below the low end about
1 week in 10 and above the high end about 1 week in 10) and the haul
chance (10 or more points).

Negative scores count as 0 here, which barely moves the range.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

EDGES = [0, 1, 2, 3, 4, 6, 8, 10, 13]   # band starts; first band is <= 0
MAX_POINTS = 30                         # scores above this count as 30
LOW_Q, HIGH_Q = 0.1, 0.9
HAUL = 10


def band_of(points):
    p = np.clip(np.asarray(points, float), 0, MAX_POINTS)
    return np.searchsorted(EDGES, p, side="right") - 1


class PointsDistribution:
    """Chance of each score 0..MAX_POINTS for a player in one match."""

    def __init__(self, cols, max_iter=150, learning_rate=0.1,
                 max_leaf_nodes=15, min_samples_leaf=200):
        self.cols = cols
        self.clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            categorical_features=[c == "pos" for c in cols],
            early_stopping=False, random_state=0)

    def fit(self, df):
        y = np.clip(df["total_points"].to_numpy(float), 0, MAX_POINTS).round()
        bands = band_of(y)
        self.clf.fit(df[self.cols], bands)
        # within each band, how often each single score occurred
        counts = np.bincount(y.astype(int), minlength=MAX_POINTS + 1).astype(float)
        self.split = np.zeros((len(EDGES), MAX_POINTS + 1))
        bands_of_score = band_of(np.arange(MAX_POINTS + 1))
        for b in range(len(EDGES)):
            inside = bands_of_score == b
            c = counts * inside
            self.split[b] = c / c.sum() if c.sum() else inside / inside.sum()
        return self

    def predict(self, df):
        """n x (MAX_POINTS + 1) array of score probabilities."""
        pb = np.zeros((len(df), len(EDGES)))
        pb[:, self.clf.classes_] = self.clf.predict_proba(df[self.cols])
        return pb @ self.split


def with_availability(dist, avail):
    """Zero points with chance (1 - avail), the forecast otherwise."""
    a = np.asarray(avail, float)[:, None]
    out = dist * a
    out[:, 0] += 1 - a[:, 0]
    return out


def add_up(dists):
    """Distribution of the total over several matches (double gameweek)."""
    total = dists[0]
    for d in dists[1:]:
        total = np.convolve(total, d)
    return total


def summarise(dist):
    """(low, high, haul chance) from one distribution."""
    cdf = np.cumsum(dist)
    low = int(np.searchsorted(cdf, LOW_Q - 1e-9))
    high = int(np.searchsorted(cdf, HIGH_Q - 1e-9))
    return low, high, float(dist[HAUL:].sum())


def per_gameweek(rows, dist, events):
    """rows: one per player per fixture with element and event; dist: the
    matching score distributions (availability already applied). Returns
    one row per element with low_gwN, high_gwN and haul_gwN columns.
    Players with no fixture in a gameweek get 0, 0 and 0."""
    out = {}
    rows = rows.reset_index(drop=True)
    for (e, ev), idx in rows.groupby(["element", "event"]).groups.items():
        out[(e, ev)] = summarise(add_up([dist[i] for i in idx]))
    elements = rows["element"].unique()
    res = pd.DataFrame({"element": elements})
    for ev in events:
        vals = [out.get((e, ev), (0, 0, 0.0)) for e in elements]
        res[f"low_gw{ev}"] = [v[0] for v in vals]
        res[f"high_gw{ev}"] = [v[1] for v in vals]
        res[f"haul_gw{ev}"] = [v[2] for v in vals]
    return res
