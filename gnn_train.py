#!/usr/bin/env python3
"""GNN over the pre-synthesis RTL cell graph vs the gradient-boosting feature model, same family-grouped folds.
Message passing: h' = relu(W_s h + W_i mean_in(h) + W_o mean_out(h)); readout = [mean, max] pooling (+ optional 4 scalar path features)."""
import json, os, sys, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from features import build_graph, TYPES, extract
torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
data = [d for d in json.load(open(os.path.join(HERE, "dataset.json"))) if d["ok"]]
ti = {t: i for i, t in enumerate(TYPES)}
graphs, scal = [], []
for d in data:
    p = os.path.join(HERE, "graphs", d["name"] + ".json")
    G, nin, nout = build_graph(p); nodes = list(G.nodes); idx = {n: i for i, n in enumerate(nodes)}
    x = np.zeros((len(nodes), len(TYPES) + 3), np.float32)
    for n, a in G.nodes(data=True):
        x[idx[n], ti.get(a["t"], 0)] = 1; x[idx[n], len(TYPES)] = np.log1p(a["w"]); x[idx[n], len(TYPES) + 1] = float(a["seq"])
        x[idx[n], len(TYPES) + 2] = 1.0
    e = np.array([[idx[u], idx[v]] for u, v in G.edges()], np.int64).reshape(-1, 2)
    graphs.append((x, e)); f, _ = extract(p); scal.append(f[[8, 9, 10, 11]])   # 4 longest-path scalars
scal = np.array(scal, np.float32); scal = np.log1p(scal)
fam = np.array([d["family"] for d in data]); yd = np.array([d["depth"] for d in data], np.float32)
yc = np.log1p(np.array([d["cells"] for d in data], np.float32))

def batch(ids):
    xs, es, gi, off = [], [], [], 0
    for k, i in enumerate(ids):
        x, e = graphs[i]; xs.append(x); es.append(e + off); gi += [k] * len(x); off += len(x)
    return (torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(es)), torch.tensor(gi), len(ids))

class GNN(nn.Module):
    def __init__(s, din, h=48, L=4, extra=0):
        super().__init__(); s.extra = extra
        s.inp = nn.Linear(din, h)
        s.ws = nn.ModuleList([nn.Linear(h, h) for _ in range(L)]); s.wi = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(L)])
        s.wo = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(L)])
        s.head = nn.Sequential(nn.Linear(2 * h + extra, h), nn.ReLU(), nn.Linear(h, 2))
    def forward(s, x, e, gi, B, sc=None):
        h = torch.relu(s.inp(x)); n = h.shape[0]; src, dst = e[:, 0], e[:, 1]
        indeg = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1).unsqueeze(1)
        outdeg = torch.zeros(n).index_add_(0, src, torch.ones(len(src))).clamp(min=1).unsqueeze(1)
        for ws, wi, wo in zip(s.ws, s.wi, s.wo):
            mi = torch.zeros_like(h).index_add_(0, dst, h[src]) / indeg
            mo = torch.zeros_like(h).index_add_(0, src, h[dst]) / outdeg
            h = torch.relu(ws(h) + wi(mi) + wo(mo)) + h
        cnt = torch.zeros(B).index_add_(0, gi, torch.ones(n)).unsqueeze(1)
        mean = torch.zeros(B, h.shape[1]).index_add_(0, gi, h) / cnt
        mx = torch.full((B, h.shape[1]), -1e9).scatter_reduce(0, gi.unsqueeze(1).expand_as(h), h, "amax")
        z = torch.cat([mean, mx] + ([sc] if sc is not None else []), 1)
        return s.head(z)

def fit_predict(tr, te, extra, epochs=160):
    m = GNN(len(TYPES) + 3, extra=4 if extra else 0); opt = torch.optim.Adam(m.parameters(), 3e-3, weight_decay=1e-4)
    ym, ys = yd[tr].mean(), yd[tr].std() + 1e-6; cm, cs = yc[tr].mean(), yc[tr].std() + 1e-6
    T = torch.tensor(np.stack([(yd[tr] - ym) / ys, (yc[tr] - cm) / cs], 1))
    for ep in range(epochs):
        perm = np.random.permutation(len(tr)); m.train()
        for b in range(0, len(tr), 32):
            ids = [tr[i] for i in perm[b:b + 32]]; xb, eb, gb, B = batch(ids)
            sc = torch.tensor(scal[ids]) if extra else None
            loss = nn.functional.smooth_l1_loss(m(xb, eb, gb, B, sc), T[perm[b:b + 32]]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xb, eb, gb, B = batch(list(te)); out = m(xb, eb, gb, B, torch.tensor(scal[te]) if extra else None).numpy()
    return out[:, 0] * ys + ym, np.expm1(out[:, 1] * cs + cm)

folds = list(GroupKFold(6).split(np.zeros(len(data)), groups=fam))
X = np.array([extract(os.path.join(HERE, "graphs", d["name"] + ".json"))[0] for d in data])
P = {k: (np.zeros(len(data)), np.zeros(len(data))) for k in ("gbr", "gnn", "gnn+path")}
for f, (tr, te) in enumerate(folds):
    for kind, ex in (("gnn", False), ("gnn+path", True)):
        pd_, pc_ = fit_predict(tr, te, ex); P[kind][0][te] = pd_; P[kind][1][te] = pc_
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=0).fit(X[tr], yd[tr]); P["gbr"][0][te] = g.predict(X[te])
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, random_state=0).fit(X[tr], yc[tr]); P["gbr"][1][te] = np.expm1(g.predict(X[te]))
    print("fold", f, "done", flush=True)
yc_raw = np.expm1(yc); out = {}
print("family-grouped 6-fold CV (whole families held out)")
for k, (pdp, pcp) in P.items():
    r = {"depth_R2": round(r2_score(yd, pdp), 3), "depth_MAE": round(mean_absolute_error(yd, pdp), 2),
         "depth_within2": round(float(np.mean(abs(pdp - yd) <= 2)), 3), "cells_R2": round(r2_score(yc_raw, pcp), 3)}
    out[k] = r; print(f"{k:9s}", r)
json.dump(out, open(os.path.join(HERE, "gnn_results.json"), "w"), indent=1)
