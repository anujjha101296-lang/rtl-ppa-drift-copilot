#!/usr/bin/env python3
"""Pairwise ranking accuracy on leave-one-family-out predictions: for random cross-family pairs with a large true gap,
how often does the model order them correctly? Also reports the size-only (n_in, n_out) model for comparison."""
import json, os, warnings, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from features import extract, FEATURE_NAMES
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
data = [d for d in json.load(open(os.path.join(HERE, "dataset.json"))) if d["ok"]]
X = np.array([extract(os.path.join(HERE, "graphs", d["name"] + ".json"))[0] for d in data])
fam = np.array([d["family"] for d in data]); yd = np.array([d["depth"] for d in data], float); yc = np.array([d["cells"] for d in data], float)
size_cols = [FEATURE_NAMES.index("n_in"), FEATURE_NAMES.index("n_out")]
def lofo(cols, y, log):
    p = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, fam):
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=0).fit(X[tr][:, cols], np.log1p(y[tr]) if log else y[tr])
        q = m.predict(X[te][:, cols]); p[te] = np.expm1(q) if log else q
    return p
def rank_acc(y, p, gap):
    rng = np.random.default_rng(0); ok = tot = 0
    for _ in range(20000):
        i, j = rng.integers(0, len(y), 2)
        if fam[i] == fam[j] or not gap(y[i], y[j]): continue
        tot += 1; ok += (y[i] > y[j]) == (p[i] > p[j])
    return round(ok / tot, 3), tot
allc = list(range(X.shape[1])); out = {}
for name, cols in (("full", allc), ("size_only", size_cols)):
    out[f"depth_{name}"] = rank_acc(yd, lofo(cols, yd, False), lambda a, b: abs(a - b) >= 3)
    out[f"cells_{name}"] = rank_acc(yc, lofo(cols, yc, True), lambda a, b: abs(a - b) / max(a, b) >= .2)
print(out); json.dump(out, open(os.path.join(HERE, "pairwise_results.json"), "w"), indent=1)
