# RTL-PPA Drift Copilot
Predict post-synthesis logic depth and gate count from the *pre-synthesis* RTL graph, and flag regressions between RTL revisions.

Pipeline: `gen_designs.py` (315 parameterized designs, 29 families) -> `run_synth.py` (Yosys `synth` + ABC gate map; ground truth = `ltp` depth, cell count; coarse-cell graph exported as JSON) -> `features.py` (77 graph features: cell-type counts/widths, 4 weighted longest paths, fanout, ports ...) -> `train_eval.py` (GradientBoosting, baselines, 2 CV protocols) -> `size_control.py` (feature-subset control) -> `pairwise_rank.py` -> `copilot.py` (`predict`, `drift old.v new.v [--verify]`).

Run: `python3 gen_designs.py && python3 run_synth.py && python3 train_eval.py && python3 size_control.py && python3 pairwise_rank.py`  (needs yosys, scikit-learn, networkx)

| target | protocol | R2 | MAE | within +-2 levels |
|---|---|---|---|---|
| depth | random 5-fold | 0.930 | 3.4 | 49.2% |
| depth | leave-one-family-out | 0.651 | 8.1 | 23.5% |
| cells (log target) | random 5-fold | 0.763 | 36 | - |
| cells | leave-one-family-out | 0.404 | 72 | - |
Mean-predictor baseline: R2 ~ 0 (depth MAE ~16).

## Control: how much is just design size?
Same model and folds, different feature subsets (`size_control_results.json`):
| features | depth R2 random | depth R2 family-held-out | gate-count R2 family-held-out |
|---|---|---|---|
| input + output bit counts only | 0.717 | 0.645 | 0.286 |
| ports + cells + edges + widths | 0.789 | 0.433 | 0.226 |
| everything except ports | 0.848 | 0.353 | 0.186 |
| structure only (type counts + path features) | 0.857 | 0.223 | 0.103 |
| all 77 features | 0.930 | 0.651 | 0.404 |

**Reading:** for depth on unseen families, port width alone matches the full model (0.645 vs 0.651), so the structural features add nothing there. They do help in-distribution (0.72 -> 0.93), for gate count (0.29 -> 0.40), and for cross-family pairwise ranking (depth 78.9% -> 84.4%, gate count 75.1% -> 86.8%).

Limits: synthetic parametric corpus; depth is not delay; the family-held-out numbers are the honest ones. Ridge diverges under family hold-out (not reported). The model is gradient boosting on graph features, not a GNN. The `drift` demo designs were in the training set.

## GNN variant (UNTESTED - needs a machine with working PyTorch)
`gnn_train.py` trains a 4-layer message-passing GNN on the RTL cell graphs and compares it with gradient boosting on identical family-grouped 6-fold CV. Syntax-checked only; no results yet. Run after `run_synth.py`: `pip install torch && python3 gnn_train.py`.
