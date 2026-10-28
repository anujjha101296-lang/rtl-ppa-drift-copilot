#!/usr/bin/env python3
"""Structural features from a PRE-synthesis RTL graph (Yosys proc/flatten/opt, coarse cells).
No post-synthesis information is used. Nodes = coarse cells ($add, $mux, $dff, ...),
edges = bit-level connectivity driver-cell -> sink-cell."""
import json, math, collections
import networkx as nx
import numpy as np

SEQ = ("$dff", "$adff", "$sdff", "$dffe", "$sdffe", "$adffe", "$dlatch", "$aldff", "$dffsr")
TYPES = ["$add", "$sub", "$mul", "$and", "$or", "$xor", "$xnor", "$not", "$mux", "$pmux", "$eq", "$ne",
         "$lt", "$le", "$gt", "$ge", "$shl", "$shr", "$sshr", "$shift", "$shiftx", "$reduce_and",
         "$reduce_or", "$reduce_xor", "$logic_not", "$logic_and", "$logic_or", "$neg", "$pos", "$dff"]


def norm_type(t):
    if t in SEQ: return "$dff"
    if t == "$sdff": return "$dff"
    return t


def cell_width(c):
    p = c.get("parameters", {})
    ws = []
    for k in ("Y_WIDTH", "A_WIDTH", "B_WIDTH", "WIDTH", "S_WIDTH"):
        if k in p:
            try: ws.append(int(str(p[k]), 2))
            except Exception: pass
    return max(ws) if ws else 1


def build_graph(path):
    j = json.load(open(path))
    mod = j["modules"]["top"]
    G = nx.DiGraph()
    driver = {}  # bit -> cell
    for name, c in mod["cells"].items():
        t = norm_type(c["type"]); w = cell_width(c)
        G.add_node(name, t=t, w=w, seq=c["type"] in SEQ)
        for port, bits in c["connections"].items():
            if c["port_directions"][port] == "output":
                for b in bits:
                    if isinstance(b, int): driver[b] = name
    for name, c in mod["cells"].items():
        for port, bits in c["connections"].items():
            if c["port_directions"][port] == "input":
                for b in bits:
                    if isinstance(b, int) and b in driver and driver[b] != name:
                        G.add_edge(driver[b], name)
    n_in = sum(len(p["bits"]) for p in mod["ports"].values() if p["direction"] == "input")
    n_out = sum(len(p["bits"]) for p in mod["ports"].values() if p["direction"] == "output")
    return G, n_in, n_out


def longest_path(G, weight_fn):
    # break at sequential cells: cut edges INTO flops (path ends at flop D) and treat flop as source
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in G.edges():
        H.add_edge(u, v)
    # sequential feedback => cut edges leaving flops so graph is a DAG per cycle
    cut = [(u, v) for u, v in H.edges() if H.nodes[u]["seq"]]
    H.remove_edges_from(cut)
    try:
        order = list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible:
        return 0.0
    dist = {}
    best = 0.0
    for n in order:
        w = 0.0 if H.nodes[n]["seq"] else weight_fn(H.nodes[n])
        d = max((dist[p] for p in H.predecessors(n)), default=0.0) + w
        dist[n] = d
        best = max(best, d)
    return best


def wl_hist(G, rounds=2):
    """Weisfeiler-Lehman style neighbourhood-type histogram (message passing, no learned weights)."""
    lab = {n: d["t"] for n, d in G.nodes(data=True)}
    feats = collections.Counter()
    for r in range(rounds):
        new = {}
        for n in G.nodes:
            ins = sorted(lab[p] for p in G.predecessors(n))
            new[n] = lab[n] + "<" + ",".join(ins[:3])
        lab = new
    return lab


FEATURE_NAMES = (["n_cells", "n_edges", "n_in", "n_out", "sum_w", "log_sum_w", "max_w", "mean_w",
                  "lp_unit", "lp_logw", "lp_w", "lp_sqrtw", "max_fanout", "mean_fanout",
                  "n_seq", "frac_seq", "mean_indeg"]
                 + [f"cnt{t}" for t in TYPES] + [f"w{t}" for t in TYPES])


def extract(path):
    G, n_in, n_out = build_graph(path)
    n = G.number_of_nodes(); e = G.number_of_edges()
    ws = [d["w"] for _, d in G.nodes(data=True)] or [1]
    cnt = collections.Counter(d["t"] for _, d in G.nodes(data=True))
    wsum = collections.defaultdict(float)
    for _, d in G.nodes(data=True): wsum[d["t"]] += d["w"]
    fo = [G.out_degree(x) for x in G.nodes] or [0]
    f = [n, e, n_in, n_out, sum(ws), math.log1p(sum(ws)), max(ws), float(np.mean(ws)),
         longest_path(G, lambda d: 1.0),
         longest_path(G, lambda d: math.log2(d["w"] + 1)),
         longest_path(G, lambda d: float(d["w"])),
         longest_path(G, lambda d: math.sqrt(d["w"])),
         max(fo), float(np.mean(fo)),
         sum(1 for _, d in G.nodes(data=True) if d["seq"]),
         (sum(1 for _, d in G.nodes(data=True) if d["seq"]) / n) if n else 0,
         float(np.mean([G.in_degree(x) for x in G.nodes] or [0]))]
    f += [cnt.get(t, 0) for t in TYPES] + [wsum.get(t, 0.0) for t in TYPES]
    return np.array(f, dtype=float), G


if __name__ == "__main__":
    import sys
    x, G = extract(sys.argv[1])
    for k, v in zip(FEATURE_NAMES, x):
        if v: print(f"{k:16s} {v:g}")
