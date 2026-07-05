#!/usr/bin/env python3
"""Ensemble aggregation across N grokking runs (different seeds/inits).

Reads every <runs_dir>/seed_*/ produced by nanda_grokking_v5. For each run,
finds its own grok moment (peak smoothed val-accuracy slope), then aggregates
the instrument curves two ways:

  step-aligned  — raw training step on x (transitions smear: seeds grok at
                  different steps)
  grok-aligned  — x = t − t_grok per run, so the transition stays sharp and the
                  mean ± band shows the *shape* of the choreography

Per-param analyses are deliberately absent: parameter indices don't correspond
across inits. Ensemble = aggregate statistics only.

Usage:  python3 tools/grokking_ensemble.py [runs_dir]     (default: ens_runs)
Output: <runs_dir>/ensemble.html (self-contained)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from grokking_report import load_csv, smooth, snap_metrics  # noqa: E402

SERIES = [
    ("val_acc",        "val accuracy (%)",            False),
    ("__slope__",      "val-acc slope (Δ%/100)",      False),
    ("jva_w500",       "output accord w500 @val refs", False),
    ("va_shape_w500",  "SHAPE accord w500",            False),
    ("cos_gtr_gva",    "cos(∇L_train, ∇L_val)",        True),
    ("lva_accord",     "val loss-work accord",         True),
]
SNAP_SERIES = [
    ("effrank_acts", "effective rank — readout acts"),
    ("r2_fourier",   "R² Fourier basis"),
    ("r2_group",     "R² 8 group-means"),
    ("fourier_share","embedding Fourier share"),
]


def run_curves(d: Path):
    cols = load_csv(d / "metrics.csv")
    steps = cols["step"]
    dt = float(np.median(np.diff(steps)))
    w100 = max(int(100 / dt), 1)
    acc_s = smooth(cols["val_acc"], w100)
    slope = smooth(np.gradient(acc_s, steps) * 100.0, w100)
    grok = float(steps[int(np.argmax(slope))])
    cols["__slope__"] = slope
    return steps, cols, grok, w100


def band(fig, row, col, x, Y, name, color, showlegend=True):
    m = np.nanmean(Y, axis=0)
    s = np.nanstd(Y, axis=0)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([m + s, (m - s)[::-1]]),
                             fill="toself", fillcolor=color.replace("1)", "0.18)"),
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=m, name=name, line=dict(color=color, width=2),
                             showlegend=showlegend), row=row, col=col)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ens_runs")
    dirs = sorted(p for p in root.glob("seed_*") if (p / "metrics.csv").exists())
    if len(dirs) < 2:
        print(f"need ≥2 completed runs under {root}", file=sys.stderr)
        return 1
    print(f"{len(dirs)} runs: {[d.name for d in dirs]}")

    runs = [run_curves(d) for d in dirs]
    groks = [g for _, _, g, _ in runs]
    print("grok steps:", [int(g) for g in groks])

    # Common grids.
    step_grid = np.arange(0, min(r[0][-1] for r in runs) + 1, 10)
    tau_grid = np.arange(-1500, 4001, 10)

    n_rows = len(SERIES) + len(SNAP_SERIES)
    figs = make_subplots(rows=n_rows, cols=2, shared_xaxes=False,
                         vertical_spacing=0.7 / n_rows, horizontal_spacing=0.07,
                         column_titles=("step-aligned", "grok-aligned (τ = t − t_grok)"),
                         subplot_titles=[t for _, t, _ in SERIES for _ in (0, 1)] +
                                        [t for _, t in SNAP_SERIES for _ in (0, 1)])

    colors = ["rgba(31,119,180,1)", "rgba(214,39,40,1)", "rgba(44,160,44,1)",
              "rgba(255,127,14,1)", "rgba(148,103,189,1)", "rgba(140,86,75,1)"]

    for r, (key, title, smooth_it) in enumerate(SERIES, start=1):
        Ys, Yt = [], []
        for steps, cols, grok, w100 in runs:
            y = cols[key]
            if smooth_it:
                y = smooth(y, w100)
            Ys.append(np.interp(step_grid, steps, y))
            Yt.append(np.interp(tau_grid, steps - grok, y, left=np.nan, right=np.nan))
        c = colors[(r - 1) % len(colors)]
        band(figs, r, 1, step_grid, np.stack(Ys), title, c, showlegend=False)
        band(figs, r, 2, tau_grid, np.stack(Yt), title, c, showlegend=False)
        figs.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5,
                       row=r, col=2)

    for k, (key, title) in enumerate(SNAP_SERIES):
        r = len(SERIES) + 1 + k
        Ys, Yt = [], []
        for (steps, cols, grok, _), d in zip(runs, dirs):
            sm = snap_metrics(d)
            ss = sm["steps"]
            Ys.append(np.interp(step_grid, ss, sm[key]))
            Yt.append(np.interp(tau_grid, ss - grok, sm[key], left=np.nan, right=np.nan))
        c = colors[k % len(colors)]
        band(figs, r, 1, step_grid, np.stack(Ys), title, c, showlegend=False)
        band(figs, r, 2, tau_grid, np.stack(Yt), title, c, showlegend=False)
        figs.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5,
                       row=r, col=2)

    figs.update_layout(height=280 * n_rows, width=1250,
                       hoverlabel=dict(namelength=-1), showlegend=False,
                       title=f"Ensemble over {len(dirs)} seeds — mean ± 1σ. "
                             f"Grok steps: {[int(g) for g in groks]}")
    out = root / "ensemble.html"
    out.write_text("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                   "<title>Grokking ensemble</title></head><body>"
                   + figs.to_html(include_plotlyjs="inline", full_html=False)
                   + "</body></html>")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
