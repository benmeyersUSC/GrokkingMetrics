#!/usr/bin/env python3
"""The grokking dashboard site: glossary, ensemble bands, per-seed deep dives.

One self-contained HTML with three sections:
  Glossary  — precise definitions of every instrument on the page
  Ensemble  — mean ± 1σ bands over all seeds, step-aligned (cropped) and
              grok-aligned (τ = t − t_grok per run, tight window)
  Per-seed  — dropdown-selected deep dive: crystallization + emergence curves,
              leverage twin heatmaps (realized vs structural potential),
              concentration, stepping-up/back traces, leaderboard, and the
              collaboration (off-diagonal Fisher) matrices.

Seeds appear in the dropdown if their run dir exists; leverage figures appear
for seeds where `nanda_leverage` outputs are present (rerun this script after
the backfill completes to pick up more).

Usage:  python3 tools/grokking_site.py [runs_dir] [extra_dir ...]
        (default: ens_runs checkpoints_nanda_grokking_v3)
Output: <runs_dir>/dashboard.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from grokking_report import (  # noqa: E402
    load_csv, smooth, snap_metrics, read_sp, read_realized, read_jcols,
    pct_rank, gini, TENSOR_NAMES,
)
from grokking_crystallization import (  # noqa: E402
    read_snap_weights, embedding_freq_power, unembedding_freq_power,
)
import re as _re
import struct as _struct  # noqa: F401


TAU_WIN = (-1250, 1250)
STEP_WIN = (0, 4000)

GLOSSARY = [
    ("val accuracy / slope",
     "Accuracy on the held-out 70% of (a,b) pairs. The slope (smoothed d(acc)/dt) is the "
     "grokking-intensity trace; each run's <b>grok moment is the peak of its own slope</b> — "
     "that per-run peak is what τ = 0 means in the grok-aligned column."),
    ("output accord (w500) @ val refs",
     "Over the last 500 steps: ‖Σ J·Δθ‖ / ‖Σ |J·Δθ|‖ measured at fixed validation inputs. "
     "J·Δθ is each step's first-order change to the model's output; the ratio is the fraction "
     "of output-space movement that survives cancellation across parameters and time. "
     "1 = every push agrees; → 0 = churn."),
    ("shape accord (w500)",
     "Same ratio, but computed only on the component of output movement <b>perpendicular</b> "
     "to the current logit direction F̂. Immune to uniform logit scaling (weight decay's "
     "melt), so it isolates <i>direction change</i> — the answers themselves changing."),
    ("melt rate ⟨net, F̂⟩",
     "The parallel component: signed output movement along the current logit direction. "
     "Negative while weight decay sheds magnitude; the confidence/parsimony channel."),
    ("cos(∇L_train, ∇L_val)",
     "Cosine between the loss gradient computed on a training batch and on a held-out val "
     "batch, at the same weights. It asks: does the direction that improves training also "
     "improve validation? ~0 during memorization; → 1 when one algorithm serves both."),
    ("val loss-work accord",
     "Each step, each parameter does val-loss-work Δθ_i · g_i^val (the sum over params ≈ the "
     "step's first-order change to val loss). The accord |Σ_i work| / Σ_i |work| is the "
     "fraction of that per-parameter work surviving cross-parameter cancellation."),
    ("effective rank",
     "Of the <b>readout activations</b> (the 128-dim vector feeding the unembedding), "
     "collected across all 12,769 inputs — and separately of the logits. Covariance "
     "eigenvalues λ → distribution p = λ/Σλ → e^{entropy(p)}: the soft count of dimensions "
     "carrying the representation. Activations, not weights."),
    ("R² Fourier / one-hot / group-means (emergence)",
     "Predictive emergence: ridge-regression test R² on held-out inputs. Fourier basis = "
     "ground-truth macro description (cos/sin of ka, kb, k(a+b) at 8 key frequencies, 48 "
     "dims); additive one-hot = 226-dim micro basis; 8 group-means = macro variables "
     "<i>discovered blind</i> from neuron correlations. Macro beating a 5×-wider micro basis "
     "= the layer's state answers to macro variables."),
    ("embedding Fourier share",
     "FFT the embedding matrix along the token axis (113 tokens); fraction of spectral power "
     "in the top-8 key frequencies. Chance level ≈ 8/56 ≈ 0.14; → 1 as the embedding "
     "concentrates onto the Fourier circle."),
    ("realized leverage |J_i| (Metric II)",
     "L2 norm of parameter i's output-shaped influence vector ∂F/∂θ_i at the trained "
     "weights — equal to √F_ii, the Fisher diagonal. Value-entangled: this run's geometry."),
    ("structural potential (Metric III)",
     "The same quantity averaged over 128 random Xavier inits — the architecture's influence "
     "prior, before any training. <b>Seed-independent by construction</b>: computed once, "
     "shared by every run of this architecture."),
    ("collaboration matrices cos(J_i, J_j)",
     "Pairwise cosine between the top-50 params' influence vectors — the off-diagonal Fisher "
     "structure the per-param norms cannot show. Red blocks = params pushing the output the "
     "same way (a circuit); blue = anti-aligned."),
]

SERIES = [
    ("val_acc",       "val accuracy (%)",             False),
    ("__slope__",     "val-acc slope (Δ%/100)",       False),
    ("jva_w500",      "output accord w500 @val refs", False),
    ("va_shape_w500", "shape accord w500",            False),
    ("cos_gtr_gva",   "cos(∇L_train, ∇L_val)",        True),
    ("lva_accord",    "val loss-work accord",         True),
]
SNAP_SERIES = [
    ("effrank_acts",  "effective rank — readout acts"),
    ("r2_fourier",    "R² Fourier basis"),
    ("r2_group",      "R² 8 group-means"),
    ("fourier_share", "embedding Fourier share"),
]


def run_curves(d: Path):
    cols = load_csv(d / "metrics.csv")
    steps = cols["step"]
    dt = float(np.median(np.diff(steps)))
    w100 = max(int(100 / dt), 1)
    acc_s = smooth(cols["val_acc"], w100)
    slope = smooth(np.gradient(acc_s, steps) * 100.0, w100)
    cols["__slope__"] = slope
    return steps, cols, float(steps[int(np.argmax(slope))]), w100


def band(fig, row, col, x, Y, color):
    m, s = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([m + s, (m - s)[::-1]]),
                             fill="toself", fillcolor=color.replace("1)", "0.16)"),
                             line=dict(width=0), hoverinfo="skip", showlegend=False),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=m, line=dict(color=color, width=2.2),
                             showlegend=False, hovertemplate="%{y:.4g}<extra></extra>"),
                  row=row, col=col)


def build_ensemble(dirs: list[Path]) -> tuple[str, list[int]]:
    runs = [run_curves(d) for d in dirs]
    groks = [int(g) for _, _, g, _ in runs]
    step_grid = np.arange(0, min(r[0][-1] for r in runs) + 1, 10)
    tau_grid = np.arange(TAU_WIN[0], TAU_WIN[1] + 1, 10)

    rows = len(SERIES) + len(SNAP_SERIES)
    titles = []
    for _, t, _ in SERIES:
        titles += [f"{t} — raw step", f"{t} — grok-aligned"]
    for _, t in SNAP_SERIES:
        titles += [f"{t} — raw step", f"{t} — grok-aligned"]
    fig = make_subplots(rows=rows, cols=2, vertical_spacing=0.55 / rows,
                        horizontal_spacing=0.06, subplot_titles=titles)
    colors = ["rgba(31,119,180,1)", "rgba(214,39,40,1)", "rgba(255,127,14,1)",
              "rgba(44,160,44,1)", "rgba(148,103,189,1)", "rgba(23,130,140,1)"]

    for r, (key, _, sm_it) in enumerate(SERIES, start=1):
        Ys, Yt = [], []
        for steps, cols, grok, w100 in runs:
            y = smooth(cols[key], w100) if sm_it else cols[key]
            Ys.append(np.interp(step_grid, steps, y))
            Yt.append(np.interp(tau_grid, steps - grok, y, left=np.nan, right=np.nan))
        c = colors[(r - 1) % len(colors)]
        band(fig, r, 1, step_grid, np.stack(Ys), c)
        band(fig, r, 2, tau_grid, np.stack(Yt), c)
        fig.add_vline(x=0, line_dash="dash", line_color="#333", opacity=0.6, row=r, col=2)
        fig.update_xaxes(range=list(STEP_WIN), row=r, col=1)
        fig.update_xaxes(range=list(TAU_WIN), row=r, col=2)

    for k, (key, _) in enumerate(SNAP_SERIES):
        r = len(SERIES) + 1 + k
        Ys, Yt = [], []
        for (steps, cols, grok, _), d in zip(runs, dirs):
            sm = snap_metrics(d)
            Ys.append(np.interp(step_grid, sm["steps"], sm[key]))
            Yt.append(np.interp(tau_grid, sm["steps"] - grok, sm[key],
                                left=np.nan, right=np.nan))
        c = colors[k % len(colors)]
        band(fig, r, 1, step_grid, np.stack(Ys), c)
        band(fig, r, 2, tau_grid, np.stack(Yt), c)
        fig.add_vline(x=0, line_dash="dash", line_color="#333", opacity=0.6, row=r, col=2)
        fig.update_xaxes(range=list(STEP_WIN), row=r, col=1)
        fig.update_xaxes(range=list(TAU_WIN), row=r, col=2)

    fig.update_xaxes(title_text="training step", row=rows, col=1)
    fig.update_xaxes(title_text="τ = t − t_grok", row=rows, col=2)
    fig.update_layout(height=310 * rows, width=1280, hoverlabel=dict(namelength=-1),
                      margin=dict(t=60, r=30), plot_bgcolor="#f7f8fa")
    fig.update_annotations(font_size=13)
    return fig.to_html(include_plotlyjs=False, full_html=False), groks


def build_seed_state(d: Path) -> str:
    sm = snap_metrics(d)
    ss = sm["steps"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        specs=[[{"secondary_y": True}, {}]],
                        subplot_titles=("crystallization", "emergence (test R²)"))
    fig.add_trace(go.Scatter(x=ss, y=sm["effrank_acts"], name="effrank acts",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ss, y=sm["effrank_logits"], name="effrank logits",
                             line=dict(color="#d62728")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ss, y=sm["fourier_share"], name="Fourier share",
                             line=dict(color="#2ca02c", width=2)),
                  row=1, col=1, secondary_y=True)
    for key, name, color in [("r2_fourier", "Fourier basis", "#2ca02c"),
                             ("r2_onehot", "one-hot", "#1f77b4"),
                             ("r2_group", "8 group-means", "#ff7f0e"),
                             ("r2_pca", "top-8 PCs", "#9467bd"),
                             ("r2_randn", "8 random neurons", "#7f7f7f")]:
        fig.add_trace(go.Scatter(x=ss, y=sm[key], name=name, line=dict(color=color)),
                      row=1, col=2)
    fig.update_yaxes(title_text="dims", row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[-0.05, 1.02], row=1, col=2)
    fig.update_layout(height=380, width=1280, hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
                      margin=dict(t=40), plot_bgcolor="#f7f8fa")
    return fig.to_html(include_plotlyjs=False, full_html=False)



def build_seed_spectra(d: Path) -> str:
    """Embedding vs unembedding frequency spectra over training + agreement curve."""
    mpath = d / "param_manifest.txt"
    snaps = sorted(d.glob("snaps/snap_*.bin"),
                   key=lambda p: int(_re.search(r"snap_(\d+)", p.name).group(1)))
    if not mpath.exists() or len(snaps) < 2:
        return ""
    sizes = [int(x) for x in mpath.read_text().split()]
    steps, Espec, Uspec, agree, top8, hshake = [], [], [], [], [], []
    for sp in snaps:
        step = int(_re.search(r"snap_(\d+)", sp.name).group(1))
        emb, unemb = read_snap_weights(sp, sizes)
        pe = embedding_freq_power(emb);   pe = pe / pe.sum()
        pu = unembedding_freq_power(unemb); pu = pu / pu.sum()
        steps.append(step); Espec.append(pe); Uspec.append(pu)
        # centered cosine: raw shares are trivially parallel at init (both near-flat)
        ce, cu = pe - pe.mean(), pu - pu.mean()
        agree.append(float(ce @ cu / (np.linalg.norm(ce) * np.linalg.norm(cu) + 1e-30)))
        # handshake: raw dot of the share spectra, in units of the chance floor
        # (uniform x uniform = 1/56) — rises only if BOTH concentrate on the SAME ks
        hshake.append(float(pe @ pu * 56.0))
        top8.append(len(set(np.argsort(pe)[-8:]) & set(np.argsort(pu)[-8:])))
    E = np.stack(Espec).T   # (56, n_snaps)
    U = np.stack(Uspec).T
    ks = list(range(1, 57))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.38, 0.38, 0.24],
                        specs=[[{}], [{}], [{"secondary_y": True}]],
                        subplot_titles=("embedding spectrum (share per frequency)",
                                        "unembedding spectrum (share per frequency)",
                                        "spectral agreement"))
    fig.add_trace(go.Heatmap(z=E, x=steps, y=ks, colorscale="Magma",
                             zmin=0, zmax=float(max(E.max(), U.max())),
                             colorbar=dict(title="share", len=0.55, y=0.78)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=U, x=steps, y=ks, colorscale="Magma",
                             zmin=0, zmax=float(max(E.max(), U.max())),
                             showscale=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=agree, name="centered cosine (alignment)",
                             line=dict(color="#2ca02c", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=hshake, name="handshake ⟨E,U⟩ ×56 (1 = chance)",
                             line=dict(color="#d62728", width=2.2)),
                  row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=steps, y=[t / 8 for t in top8], name="top-8 overlap (fraction)",
                             line=dict(color="#ff7f0e", width=1.6, dash="dot")), row=3, col=1)
    fig.update_yaxes(title_text="k", row=1, col=1)
    fig.update_yaxes(title_text="k", row=2, col=1)
    fig.update_yaxes(range=[-0.3, 1.05], title_text="alignment / overlap", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="handshake (× chance)", type="log", row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="training step", row=3, col=1)
    fig.update_layout(height=760, width=1280, hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=-0.09, font=dict(size=10)),
                      margin=dict(t=40), plot_bgcolor="#f7f8fa")
    return ("<h4>Embedding vs unembedding spectra — the two ends agreeing on the circles</h4>"
            "<p class='sub'>FFT of the embedding along the token axis and of the unembedding "
            "along the candidate axis, per snapshot, normalized to share. The dot-product "
            "scoring identity only works if both matrices commit to the same frequencies — "
            "noise first, then the same few bright rows in both panels. The agreement curve "
            "is the cosine between the two spectra; the dotted line counts shared top-8 "
            "frequencies.</p>"
            + fig.to_html(include_plotlyjs=False, full_html=False))

def build_seed_leverage(d: Path) -> str:
    sp = read_sp(d / "structural_potential.bin")
    lsteps, realized = read_realized(d / "leverage_realized.bin")
    sizes = [int(x) for x in (d / "param_manifest.txt").read_text().split()]
    names = TENSOR_NAMES if len(TENSOR_NAMES) == len(sizes) else \
        [f"t{i}" for i in range(len(sizes))]
    bounds = np.cumsum([0] + sizes)
    tensor_of = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    order = np.argsort(realized[-1])[::-1]
    sel = order[np.linspace(0, len(order) - 1, 500).astype(int)]

    html = []

    fig = make_subplots(rows=1, cols=2, column_widths=[0.92, 0.08],
                        horizontal_spacing=0.015,
                        subplot_titles=("realized |J_i| percentile × time",
                                        "potential"))
    fig.add_trace(go.Heatmap(z=pct_rank(realized)[:, sel].T, x=lsteps,
                             colorscale="Magma", zmin=0, zmax=1,
                             colorbar=dict(title="pct")), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pct_rank(sp[None, :])[0][sel][:, None], x=[0],
                             colorscale="Magma", zmin=0, zmax=1, showscale=False),
                  row=1, col=2)
    fig.update_yaxes(autorange="reversed", title_text="params (by final realized)",
                     row=1, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(height=480, width=1280, margin=dict(t=40),
                      title_text=None, plot_bgcolor="#f7f8fa")
    html.append(fig.to_html(include_plotlyjs=False, full_html=False))

    ginis = np.array([gini(r) for r in realized])
    lam = realized ** 2
    partf = (lam.sum(1) ** 2 / (lam ** 2).sum(1)) / realized.shape[1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=lsteps, y=ginis, name="Gini(realized)",
                             line=dict(color="#d62728", width=2)))
    fig.add_trace(go.Scatter(x=lsteps, y=partf, name="participation fraction",
                             line=dict(color="#1f77b4", width=2)), secondary_y=True)
    fig.update_yaxes(title_text="Gini", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="participating fraction", type="log", secondary_y=True)
    fig.update_layout(height=300, width=1280, hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=-0.3), margin=dict(t=25),
                      plot_bgcolor="#f7f8fa")
    html.append(fig.to_html(include_plotlyjs=False, full_html=False))

    def label_of(i: int) -> str:
        t = tensor_of[i]
        return f"{names[t]}[{i - bounds[t]}]"

    eps = 1e-12
    ratio = realized / (sp[None, :] + eps)
    early_strong = np.where(realized[0] >= np.quantile(realized[0], 0.9))[0]
    decline = (ratio[-1, early_strong] + eps) / (ratio[0, early_strong] + eps)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("stepping up — top-10 final realized",
                                        "stepping back — early strong, biggest decline"))
    for i in order[:10]:
        fig.add_trace(go.Scatter(x=lsteps, y=realized[:, i], name=label_of(i),
                                 legend="legend"), row=1, col=1)
    for i in early_strong[np.argsort(decline)[:10]]:
        fig.add_trace(go.Scatter(x=lsteps, y=realized[:, i], name=label_of(i),
                                 legend="legend2"), row=1, col=2)
    fig.update_yaxes(type="log", title_text="|J_i|", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_layout(height=430, width=1280, hoverlabel=dict(namelength=-1),
                      legend=dict(x=0.0, y=-0.32, orientation="h", font=dict(size=9)),
                      legend2=dict(x=0.55, y=-0.32, orientation="h", font=dict(size=9)),
                      margin=dict(t=40, b=130), plot_bgcolor="#f7f8fa")
    html.append(fig.to_html(include_plotlyjs=False, full_html=False))

    top = order[:25]
    fig = go.Figure(go.Bar(x=realized[-1, top][::-1],
                           y=[label_of(i) for i in top][::-1], orientation="h",
                           marker=dict(color=sp[top][::-1], colorscale="Viridis",
                                       colorbar=dict(title="potential"))))
    fig.update_xaxes(type="log", title_text="final realized |J_i|")
    fig.update_layout(height=520, width=1280, margin=dict(t=30),
                      plot_bgcolor="#f7f8fa")
    html.append("<h4>Leading realized-leverage params (bar color = architectural prior)</h4>"
                + fig.to_html(include_plotlyjs=False, full_html=False))

    jc_path = d / "j_columns.bin"
    if jc_path.exists():
        jsteps, jcols = read_jcols(jc_path)
        _, _, grok, _ = run_curves(d)
        picks = sorted({0, int(np.searchsorted(jsteps, grok * 0.6)),
                        int(np.searchsorted(jsteps, grok)), len(jsteps) - 1})
        picks = [min(p, len(jsteps) - 1) for p in picks]
        fig = make_subplots(rows=1, cols=len(picks),
                            subplot_titles=[f"step {int(jsteps[p])}" for p in picks],
                            horizontal_spacing=0.03)
        for c, p in enumerate(picks, start=1):
            M = jcols[p]
            nrm = np.linalg.norm(M, axis=1, keepdims=True)
            C = (M / (nrm + 1e-12)) @ (M / (nrm + 1e-12)).T
            fig.add_trace(go.Heatmap(z=C, colorscale="RdBu", zmin=-1, zmax=1,
                                     showscale=(c == len(picks))), row=1, col=c)
            fig.update_yaxes(autorange="reversed", row=1, col=c,
                             showticklabels=(c == 1))
        fig.update_layout(height=400, width=1280, margin=dict(t=40),
                          plot_bgcolor="#f7f8fa")
        html.append("<h4>Collaboration — cos(J_i, J_j) among top-50 params "
                    "(off-diagonal Fisher)</h4>"
                    + fig.to_html(include_plotlyjs=False, full_html=False))
    return "\n".join(html)


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;background:#eef0f4;color:#1c1e21}
.wrap{max-width:1340px;margin:0 auto;padding:28px 20px 80px}
header{background:#141a26;color:#fff;padding:34px 20px 26px}
header h1{margin:0;font-size:26px;font-weight:650}
header p{margin:8px 0 0;color:#9fb0c9;font-size:14px;max-width:900px}
nav{position:sticky;top:0;background:#141a26;padding:10px 20px;z-index:10;
    border-top:1px solid #2a3550}
nav a{color:#cfe0ff;text-decoration:none;margin-right:22px;font-size:14px;font-weight:500}
nav a:hover{color:#fff}
.card{background:#fff;border-radius:14px;box-shadow:0 1px 5px rgba(20,26,38,.09);
      padding:26px 28px;margin:26px 0;overflow-x:auto}
h2{font-size:20px;margin:0 0 6px;font-weight:650}
.sub{color:#5a6472;font-size:13.5px;margin:0 0 18px;max-width:1000px;line-height:1.5}
dl.gloss dt{font-weight:640;margin-top:14px;font-size:14.5px}
dl.gloss dd{margin:3px 0 0 0;color:#4d5765;font-size:13.5px;line-height:1.55;max-width:1050px}
select{font-size:15px;padding:7px 12px;border-radius:8px;border:1px solid #c6ccd8;
       background:#fff;margin-bottom:14px}
.badge{display:inline-block;background:#e8eefc;color:#2b4a9b;border-radius:6px;
       padding:2px 9px;font-size:12.5px;margin-left:8px}
h4{margin:26px 0 4px;font-size:15px;font-weight:640}
"""


def main() -> int:
    args = sys.argv[1:] or ["ens_runs", "checkpoints_nanda_grokking_v3"]
    root = Path(args[0])
    seed_dirs = sorted((p for p in root.glob("seed_*") if (p / "metrics.csv").exists()),
                       key=lambda p: int(p.name.split("_")[1]))
    extra = [Path(a) for a in args[1:] if (Path(a) / "metrics.csv").exists()]
    all_dirs = [(d.name, d) for d in seed_dirs] + [(d.name, d) for d in extra]

    ens_html, groks = build_ensemble(seed_dirs)

    seed_blocks, options = [], []
    for name, d in all_dirs:
        _, _, grok, _ = run_curves(d)
        has_lev = (d / "leverage_realized.bin").exists()
        options.append(f"<option value='blk-{name}'>{name} — grok @ {int(grok)}"
                       f"{' · leverage ✓' if has_lev else ''}</option>")
        parts = [f"<h3>{name} <span class='badge'>grok @ {int(grok)}</span></h3>",
                 build_seed_state(d), build_seed_spectra(d)]
        lev_html = None
        if has_lev:
            try:
                lev_html = build_seed_leverage(d)
            except Exception as e:  # in-progress backfill writes partial files
                print(f"  {name}: leverage files incomplete ({e}); skipping")
        if lev_html:
            parts.append(lev_html)
        else:
            parts.append("<p class='sub'>leverage pass not yet computed for this seed — "
                         "rerun this script after the backfill completes.</p>")
        seed_blocks.append(f"<div class='seed-block' id='blk-{name}' "
                           f"style='display:none'>{''.join(parts)}</div>")

    gloss = "".join(f"<dt>{t}</dt><dd>{b}</dd>" for t, b in GLOSSARY)
    import plotly.offline as po
    plotlyjs = po.get_plotlyjs()

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Grokking Instrument Panel</title><style>{CSS}</style>
<script>{plotlyjs}</script></head><body>
<header><h1>Grokking Instrument Panel</h1>
<p>Nanda-style modular addition (mod 113), 1-layer transformer, AdamW wd=1.0 — trained,
instrumented, and replicated in TTTN. {len(seed_dirs)} independent seeds
(grok steps {min(groks)}–{max(groks)}, mean {int(np.mean(groks))}), plus the canonical
fully-instrumented run. Movement metrics (accord, loss-work), state metrics
(crystallization, emergence), and leverage (realized vs architectural potential).</p>
</header>
<nav><a href='#gloss'>Glossary</a><a href='#ens'>Ensemble</a><a href='#seed'>Per-seed</a></nav>
<div class='wrap'>
<div class='card' id='gloss'><h2>Glossary</h2>
<p class='sub'>Every instrument on this page, defined once.</p>
<dl class='gloss'>{gloss}</dl></div>
<div class='card' id='ens'><h2>Ensemble — {len(seed_dirs)} seeds, mean ± 1σ</h2>
<p class='sub'>Left: raw training step (cropped to {STEP_WIN[1]} — the action is early;
zoom out for the full tail). Right: grok-aligned — each run is shifted so its own
grok moment (peak val-accuracy slope) sits at τ = 0, then averaged. Alignment is per-run,
which is why the transition stays sharp even though grok steps span
{min(groks)}–{max(groks)}.</p>
{ens_html}</div>
<div class='card' id='seed'><h2>Per-seed deep dive</h2>
<p class='sub'>Crystallization and emergence for the selected network, plus — where the
leverage pass has run — realized-vs-potential heatmaps (within-snapshot percentiles,
one colorscale), leverage concentration, per-param trajectories, the leaderboard, and
the collaboration matrices. Per-param views are single-network by nature: parameter
indices do not correspond across inits.</p>
<select id='seedsel' onchange="document.querySelectorAll('.seed-block').forEach(e=>e.style.display='none');document.getElementById(this.value).style.display='block';">
{''.join(options)}</select>
{''.join(seed_blocks)}
<script>document.getElementById('seedsel').dispatchEvent(new Event('change'));</script>
</div></div></body></html>"""

    out = root / "dashboard.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
