#!/usr/bin/env python3
"""For each design: (1) pre-synthesis RTL graph (Yosys proc/flatten/opt, coarse cells,
exported as JSON), (2) post-synthesis ground truth (synth + abc gate map: cell count,
longest topological path = logic depth). Writes dataset.json."""
import json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "designs")
GRAPHS = os.path.join(HERE, "graphs"); os.makedirs(GRAPHS, exist_ok=True)
GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT"


def ys(script):
    r = subprocess.run(["yosys", "-q", "-p", script], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def process(m):
    name = m["name"]; f = os.path.join(D, name + ".v")
    gj = os.path.join(GRAPHS, name + ".json")
    # 1. pre-synthesis coarse graph (RTL structure, no logic optimization beyond proc/opt_clean)
    rc, out = ys(f"read_verilog -sv {f}; hierarchy -top top; proc; flatten; opt_expr; opt_clean; write_json {gj}")
    if rc != 0:
        return {**m, "ok": False, "err": out[-300:]}
    # 2. post-synthesis ground truth
    with tempfile.TemporaryDirectory() as t:
        ltp = os.path.join(t, "ltp.txt"); st = os.path.join(t, "st.txt")
        rc, out = ys(f"read_verilog -sv {f}; synth -top top -flatten; abc -g {GATES}; opt_clean; "
                     f"tee -o {ltp} ltp -noff; tee -o {st} stat")
        if rc != 0:
            return {**m, "ok": False, "err": out[-300:]}
        L = open(ltp).read(); S = open(st).read()
    mm = re.search(r"length=(\d+)", L)
    depth = int(mm.group(1)) if mm else 0
    cells = re.search(r"Number of cells:\s+(\d+)", S)
    cells = int(cells.group(1)) if cells else 0
    ffs = sum(int(x) for x in re.findall(r"\$_\w*DFF\w*_\s+(\d+)", S))
    # depth = number of logic cells on longest path (ltp counts nodes incl. PI start => cells = length-1... keep raw)
    return {**m, "ok": True, "depth": depth, "cells": cells, "ffs": ffs}


if __name__ == "__main__":
    man = json.load(open(os.path.join(D, "manifest.json")))
    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(process, man))
    bad = [r for r in res if not r["ok"]]
    print(f"{len(res)-len(bad)} ok, {len(bad)} failed")
    for b in bad[:10]:
        print(b["name"], b["err"][-200:].replace("\n", " "))
    json.dump(res, open(os.path.join(HERE, "dataset.json"), "w"), indent=1)
