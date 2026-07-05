# GrokkingMetrics

**➡ Live site: <https://benmeyersusc.github.io/GrokkingMetrics/>** — the interactive
dashboard: the circle algorithm taught interactively, the trained model running it live,
and the full training instrumentation.


Using my own [TTTN](https://github.com/benmeyersUSC/TTNN) C++ Machine Learning library, I
reproduce Neel Nanda's grokking transformer results, profiling the model's learning of its
algorithm through the lens of several different metrics — movement metrics (output-space
accord, loss-space work), state metrics (spectral crystallization, predictive emergence),
and leverage metrics (realized Jacobian influence vs. architecture-level structural
potential).

The **instrumentation lives upstream in TTTN** (`FunctionalInfluence.hpp`, trajectory
metrics on `Param`, `AdamState` weight decay, positional leverage). This repo holds the
**experiment**: the Nanda-style network, the training runner, the leverage analyzer, and
the visualization/analysis pipeline.

## Layout

```
nanda_net.hpp                  shared network definition (task, blocks, dataset)
nanda_grokking.cpp             training runner — full instrumentation, snapshots, resumable
nanda_leverage.cpp             structural potential (Metric III) + realized leverage +
                               top-param Jacobian columns per snapshot
tools/
  grokking_crystallization.py  effective rank, embedding Fourier share, emergence R²
  grokking_report.py           single-run report.html (all instruments, one time axis)
  grokking_ensemble.py         N-seed mean ± σ bands, step- and grok-aligned
  grokking_site.py             the dashboard: glossary + ensemble + per-seed deep dives
  leverage_heatmap.py          static leverage figures (matplotlib)
  nanda_grokking_dashboard.py  quick 5-panel PNG from one run's metrics.csv
circle_algorithm.html          interactive explainer: pegs & winding, dials & shadows,
                               the DFT (freeze vs spin), interference scoring
nanda_grokking_experiment.tex  concise experiment write-up (metric derivations)
```

## Build

TTTN is pulled as a dependency into `deps/TTTN`:

```bash
./build.sh                # clone TTTN from GitHub if missing, compile both binaries
./build.sh --local        # snapshot ../TTTN working tree instead (for unpushed TTTN changes)
./fetch_deps.sh --update  # fast-forward the dep to latest remote
```

## Run

```bash
# training: [dir] [seed] [jacobian-every] [per-param-accumulators 0/1]
./nanda_grokking runs/seed_1 1 10 0

# leverage analysis over a run's retained snapshots
./nanda_leverage runs/seed_1

# dashboards
python3 tools/grokking_report.py runs/seed_1        # single-run report.html
python3 tools/grokking_ensemble.py runs             # ensemble.html over runs/seed_*
python3 tools/grokking_site.py runs                 # dashboard.html (the flagship)
```

Structural potential is architecture-level (an expectation over random inits) and therefore
seed-independent: compute it once and copy `structural_potential.bin` between run dirs.

## Data

Run artifacts (`checkpoints_*/`, `ens_runs/`, `runs/`) are git-ignored — reproduce with the
commands above, or see `nanda_grokking_experiment.tex` for the configuration used in the
write-up.
