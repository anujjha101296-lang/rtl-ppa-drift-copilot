#!/usr/bin/env python3
"""RTL-PPA drift copilot CLI.
  copilot.py predict a.v            -> predicted depth/cells from pre-synthesis graph
  copilot.py drift old.v new.v      -> predicted delta, flags regression; --verify runs Yosys/ABC to check
"""
import sys, os, pickle, subprocess, re, tempfile
import numpy as np
from features import extract
HERE = os.path.dirname(os.path.abspath(__file__))
M = pickle.load(open(os.path.join(HERE, "model.pkl"), "rb"))["models"]

def graph_json(v):
    t = tempfile.mktemp(suffix=".json")
    subprocess.run(["yosys","-q","-p",f"read_verilog -sv {v}; hierarchy -top top; proc; flatten; opt_expr; opt_clean; write_json {t}"],check=True,capture_output=True)
    return t
def predict(v):
    x,_ = extract(graph_json(v)); out={}
    for k,(m,log) in M.items():
        p=m.predict(x.reshape(1,-1))[0]; out[k]=float(np.expm1(p) if log else p)
    return out
def truth(v):
    with tempfile.TemporaryDirectory() as t:
        subprocess.run(["yosys","-q","-p",f"read_verilog -sv {v}; synth -top top -flatten; abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT; opt_clean; tee -o {t}/l ltp -noff; tee -o {t}/s stat"],check=True,capture_output=True)
        d=int(re.search(r"length=(\d+)",open(f"{t}/l").read()).group(1))
        c=int(re.search(r"Number of cells:\s+(\d+)",open(f"{t}/s").read()).group(1))
    return {"depth":d,"cells":c}
if __name__=="__main__":
    a=sys.argv[1:]
    if a[0]=="predict":
        print({k:round(v,1) for k,v in predict(a[1]).items()})
    elif a[0]=="drift":
        o,n=predict(a[1]),predict(a[2])
        dd,dc=n["depth"]-o["depth"],(n["cells"]/o["cells"]-1)*100
        print(f"predicted depth {o['depth']:.0f} -> {n['depth']:.0f} ({dd:+.0f}); cells {o['cells']:.0f} -> {n['cells']:.0f} ({dc:+.0f}%)")
        if dd>2 or dc>10: print("FLAG: likely timing/area regression")
        if "--verify" in a:
            to,tn=truth(a[1]),truth(a[2]); print("actual (Yosys/ABC):",to,"->",tn)
