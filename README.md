# RTL PPA Drift Copilot

Predict post-synthesis logic depth and mapped gate count from a **pre-synthesis Yosys RTL graph**, then compare two RTL revisions for likely structural regressions.

> **Reproducibility status:** The checked-in `dataset.json` and result JSON files are historical result snapshots. The generated RTL corpus, pre-synthesis graph JSON files, and trained `model.pkl` are intentionally generated locally/CI and are not included in a fresh clone. Run the full pipeline below before using the prediction CLI. Reported metrics are historical until reproduced in your own toolchain.

## Pipeline

1. `gen_designs.py` generates 315 parameterized RTL designs across 29 synthetic design families and writes `designs/manifest.json`.
2. `run_synth.py` uses Yosys to export pre-synthesis coarse-cell graphs and collect post-synthesis ABC gate-count / logic-depth labels.
3. `train_eval.py` extracts structural features, evaluates baselines and models using random 5-fold and leave-one-family-out splits, then writes `model.pkl`, `results.json`, and `parity.png`.
4. `copilot.py` predicts for a new RTL module or compares two revisions. Input RTL must define a top module named `top`.

## Requirements

- Python 3.10+
- Yosys with ABC support
- Python packages in `requirements.txt`

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Yosys using your OS package manager. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y yosys
```

## Rebuild the dataset and model from scratch

Run from this repository directory:

```bash
python gen_designs.py
python run_synth.py
python train_eval.py
```

The pipeline creates `designs/`, `graphs/`, `model.pkl`, `results.json`, and `parity.png`. The corpus is synthetic and designed for controlled experiments; it is not a substitute for validation on independent industrial RTL or physical timing data.

## Use the CLI

Create a file such as `example.v` with a module named `top`, then:

```bash
python copilot.py predict example.v
python copilot.py drift old.v new.v
python copilot.py drift old.v new.v --verify
```

`--verify` runs Yosys/ABC to compare predictions with actual synthesis results for the supplied files.

## Historical benchmark snapshot

The checked-in `results.json` reports the following prior run:

| Target | Evaluation protocol | R² | MAE |
|---|---|---:|---:|
| Logic depth | Random 5-fold | 0.930 | 3.4 |
| Logic depth | Leave-one-family-out | 0.651 | 8.1 |
| Gate count (log target) | Random 5-fold | 0.763 | 36 |
| Gate count | Leave-one-family-out | 0.404 | 72 |

The leave-one-family-out results are the more meaningful generalization estimate for this synthetic corpus. Port-width-only controls perform nearly as well as the full feature set for depth on unseen families; see `size_control_results.json`. These results do **not** establish accuracy on real-world RTL or predict routed timing.

## Known limitations

- Synthetic parameterized corpus; external RTL validation is still required.
- Logic depth is a mapped logic-level proxy, not static timing analysis or routed delay.
- `gnn_train.py` is experimental and requires PyTorch; it is not part of the validated baseline.
- The generated model and graph files are not checked in. Rebuild them using the documented pipeline.

## License

See `LICENSE`.
