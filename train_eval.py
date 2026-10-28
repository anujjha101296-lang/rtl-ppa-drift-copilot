#!/usr/bin/env python3
import json, os, pickle, warnings
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import extract, FEATURE_NAMES

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
data = [d for d in json.load(open(os.path.join(HERE, "dataset.json"))) if d["ok"]]
X, G = [], []
for d in data:
    x, _ = extract(os.path.join(HERE, "graphs", d["name"] + ".json")); X.append(x)
X = np.array(X); fam = np.array([d["family"] for d in data])
ydep = np.array([d["depth"] for d in data], float)
ycell = np.array([d["cells"] for d in data], float)
print(f"N={len(data)} designs, {len(set(fam))} families, {X.shape[1]} features")

def gbr(): return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, random_state=0)
def ridge():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=3.0))

def cv_predict(make, X, y, splitter, groups=None, log=False):
    pred = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        m = make(); yt = np.log1p(y[tr]) if log else y[tr]
        m.fit(X[tr], yt); p = m.predict(X[te]); pred[te] = np.expm1(p) if log else p
    return pred

def metrics(y, p):
    return {"R2": round(r2_score(y, p), 3), "MAE": round(mean_absolute_error(y, p), 2),
            "within_pm2": round(float(np.mean(np.abs(p - y) <= 2)), 3),
            "within_10pct": round(float(np.mean(np.abs(p - y) <= 0.10 * np.maximum(y, 1))), 3)}

res = {}
protocols = {
    "random_5fold": (KFold(5, shuffle=True, random_state=1), None),
    "leave_one_family_out": (LeaveOneGroupOut(), fam),
}
for tname, y, log in (("depth", ydep, False), ("cells", ycell, True)):
    for pname, (sp, grp) in protocols.items():
        for mname, mk, lg in (("mean_baseline", lambda: DummyRegressor(), False),
                              ("ridge", ridge, log), ("gbr", gbr, log)):
            p = cv_predict(mk, X, y, sp, grp, lg)
            res[f"{tname}|{pname}|{mname}"] = metrics(y, p)
            if mname == "gbr": res[f"_pred|{tname}|{pname}"] = (y.tolist(), p.tolist())
for k, v in res.items():
    if not k.startswith("_"): print(f"{k:42s} {v}")

# ---- final model on all data + plots
final = {}
for tname, y, log in (("depth", ydep, False), ("cells", ycell, True)):
    m = gbr(); m.fit(X, np.log1p(y) if log else y); final[tname] = (m, log)
pickle.dump({"models": final, "features": FEATURE_NAMES}, open(os.path.join(HERE, "model.pkl"), "wb"))

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, (t, lab) in zip(ax, (("depth", "Logic depth (ltp levels)"), ("cells", "Gate count (post-ABC)"))):
    y, p = res[f"_pred|{t}|leave_one_family_out"]
    a.scatter(y, p, s=12, alpha=.6); lo, hi = min(y + p), max(y + p)
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    r = res[f"{t}|leave_one_family_out|gbr"]
    a.set_xlabel(f"Yosys/ABC actual — {lab}"); a.set_ylabel("Predicted from pre-synthesis RTL graph")
    a.set_title(f"{t}: leave-one-family-out  R²={r['R2']}"); a.set_xscale("log" if t == "cells" else "linear")
    a.set_yscale("log" if t == "cells" else "linear")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "parity.png"), dpi=130)

imp = sorted(zip(final["depth"][0].feature_importances_, FEATURE_NAMES), reverse=True)[:8]
print("top depth features:", [(n, round(i, 3)) for i, n in imp])
out = {k: v for k, v in res.items() if not k.startswith("_")}
out["n_designs"] = len(data); out["n_families"] = int(len(set(fam))); out["n_features"] = int(X.shape[1])
out["top_depth_features"] = [(n, round(float(i), 3)) for i, n in imp]
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
