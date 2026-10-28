#!/usr/bin/env python3
"""Control experiment: how much of the model's skill is just design size?
Same GBR, same folds as train_eval.py, different feature subsets."""
import json, os, warnings, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
from features import extract, FEATURE_NAMES
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
data = [d for d in json.load(open(os.path.join(HERE, "dataset.json"))) if d["ok"]]
X = np.array([extract(os.path.join(HERE, "graphs", d["name"] + ".json"))[0] for d in data])
fam = np.array([d["family"] for d in data]); yd = np.array([d["depth"] for d in data], float); yc = np.array([d["cells"] for d in data], float)
ix = {n: i for i, n in enumerate(FEATURE_NAMES)}
SIZE = ["n_in", "n_out"]; SIZE_WIDE = SIZE + ["n_cells", "n_edges", "sum_w", "log_sum_w", "max_w", "mean_w"]
subsets = {"size_only (n_in, n_out)": [ix[n] for n in SIZE],
           "size_wide (ports, cells, edges, widths)": [ix[n] for n in SIZE_WIDE],
           "no_ports (everything except n_in, n_out)": [i for i, n in enumerate(FEATURE_NAMES) if n not in SIZE],
           "structure_only (type counts + path features)": [i for i, n in enumerate(FEATURE_NAMES) if n not in SIZE_WIDE],
           "full": list(range(len(FEATURE_NAMES)))}
def gbr(): return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
def cv(cols, y, sp, groups, log):
    p = np.zeros_like(y)
    for tr, te in sp.split(X, y, groups):
        m = gbr().fit(X[tr][:, cols], np.log1p(y[tr]) if log else y[tr]); q = m.predict(X[te][:, cols]); p[te] = np.expm1(q) if log else q
    return p
out = {}
print(f"{'features':48s} {'depth R2 rnd':>12s} {'depth R2 LOFO':>14s} {'MAE LOFO':>9s} {'cells R2 LOFO':>14s}")
for name, cols in subsets.items():
    r = {}
    for tgt, y, log in (("depth", yd, False), ("cells", yc, True)):
        for pn, sp, g in (("rnd", KFold(5, shuffle=True, random_state=1), None), ("lofo", LeaveOneGroupOut(), fam)):
            p = cv(cols, y, sp, g, log); r[f"{tgt}_{pn}_R2"] = round(r2_score(y, p), 3); r[f"{tgt}_{pn}_MAE"] = round(mean_absolute_error(y, p), 2)
    out[name] = r
    print(f"{name:48s} {r['depth_rnd_R2']:12.3f} {r['depth_lofo_R2']:14.3f} {r['depth_lofo_MAE']:9.2f} {r['cells_lofo_R2']:14.3f}")
json.dump(out, open(os.path.join(HERE, "size_control_results.json"), "w"), indent=1)
