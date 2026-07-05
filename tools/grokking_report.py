#!/usr/bin/env python3
"""Single-file interactive report for a grokking run: every instrument on one page.

Section A — time series (shared x, one legend per row, grok line = peak val slope):
  1. Losses (log) + accuracies
  2. Val-accuracy slope — grokking as intensity, not a point
  3. Finding vs honing — output-space accord @ val refs (total vs SHAPE) + melt rate
  4. Loss-space work accords + cos(∇L_train, ∇L_val)
  5. Spectral crystallization (effective rank, Fourier share)      [snapshots]
  6. Causal emergence (macro-basis R², discovered-grouping R²)     [snapshots]

Section B — leverage (Metric II vs Metric III), if nanda_leverage outputs exist:
  Twin heatmaps (same colorscale, within-snapshot percentile, unsigned):
  realized |J_i| over time beside the architectural prior; concentration curves;
  stepping-up / stepping-back traces; top-30 leaderboard.

Section C — collaboration (off-diagonal Fisher), if j_columns.bin exists:
  Pairwise cosine of the top-50 params' Jacobian columns at four moments.

Usage:  python3 tools/grokking_report.py [ckpt_dir]
Output: <ckpt_dir>/report.html (self-contained)
"""

from __future__ import annotations

import csv
import re
import struct
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from grokking_crystallization import (  # noqa: E402
    P, N_GROUPS, KEY_K, TRAIN_FRAC, SPLIT_SEED,
    read_acts, spectral_stats, embedding_freq_power,
    fourier_features, ridge_r2, group_means,
)

TENSOR_NAMES = [
    "embed.W", "embed.b", "pos_emb",
    "ln1.gamma", "ln1.beta",
    "attn.Q", "attn.K", "attn.V", "attn.O",
    "ln2.gamma", "ln2.beta",
    "ffn_in.W", "ffn_in.b", "ffn_out.W", "ffn_out.b",
    "unembed.W", "unembed.b",
]

CAPTIONS = {
    "A": "All time-series instruments on one axis. The dashed vertical line is the "
         "grokking moment defined as the <b>peak of the validation-accuracy slope</b>.",
    "cos": "cos(∇L_train, ∇L_val): alignment between the gradient computed on a training "
           "batch and on a held-out validation batch at the same weights. Near 0 during "
           "memorization — the direction that helps train does nothing for val. Rising "
           "toward 1 as one algorithm starts serving both distributions.",
    "B": "Metric II (realized |J_i|, this run's values) against Metric III (structural "
         "potential, averaged over random inits — the architecture's prior). Heatmap "
         "values are <b>within-snapshot percentiles</b> of leverage — unsigned, "
         "scale-free (weight decay's global shrink cancels), and robust to the heavy "
         "tail (a softmax would saturate on the top param). Both panels share one "
         "colorscale; rows are ordered identically, so a mismatch between the left "
         "gradient and the right strip is exactly the divergence of trained routing "
         "from architectural prescription.",
    "C": "Off-diagonal structure the per-param diagonal cannot show: pairwise cosine of "
         "the top-50 params' Jacobian columns (their influence directions in output "
         "space). Red blocks = params pushing the output the same way (a circuit); "
         "blue = anti-aligned. Crystallization should appear as block structure "
         "emerging by the grok moment.",
}


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def smooth(y: np.ndarray, rows: int) -> np.ndarray:
    if rows <= 1:
        return y
    k = np.ones(rows) / rows
    pad = rows // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, k, mode="same")[pad:pad + len(y)]


def read_sp(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        (pc,) = struct.unpack("<Q", f.read(8))
        return np.frombuffer(f.read(4 * pc), dtype=np.float32).astype(np.float64)


def read_realized(path: Path):
    with path.open("rb") as f:
        n, pc = struct.unpack("<2Q", f.read(16))
        steps, mats = [], []
        for _ in range(n):
            (step,) = struct.unpack("<Q", f.read(8))
            steps.append(step)
            mats.append(np.frombuffer(f.read(4 * pc), dtype=np.float32).astype(np.float64))
    return np.array(steps), np.stack(mats)


def read_jcols(path: Path):
    with path.open("rb") as f:
        n, topk, outsize = struct.unpack("<3Q", f.read(24))
        steps, mats = [], []
        for _ in range(n):
            (step,) = struct.unpack("<Q", f.read(8))
            steps.append(step)
            m = np.frombuffer(f.read(4 * topk * outsize), dtype=np.float32)
            mats.append(m.reshape(topk, outsize).astype(np.float64))
    return np.array(steps), np.stack(mats)          # (n,), (n, topk, out)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x))
    n, s = len(x), x.sum()
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s)) if s > 0 else 0.0


def pct_rank(m: np.ndarray) -> np.ndarray:
    """Within-row percentile rank (rows = snapshots), 0..1."""
    r = np.argsort(np.argsort(m, axis=1), axis=1).astype(float)
    return r / (m.shape[1] - 1)


def snap_metrics(ckpt: Path) -> dict[str, np.ndarray]:
    snaps = sorted(ckpt.glob("snaps/acts_*.bin"),
                   key=lambda p: int(re.search(r"acts_(\d+)", p.name).group(1)))
    cache_path = ckpt / "crystal_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}
    done = set(cache.get("steps", np.array([])).astype(int).tolist())
    todo = [p for p in snaps if int(re.search(r"acts_(\d+)", p.name).group(1)) not in done]
    if todo:
        n = P * P
        idx = np.arange(n)
        a, b = idx // P, idx % P
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(n)
        ntr = int(TRAIN_FRAC * n)
        tr_i, te_i = perm[:ntr], perm[ntr:]
        _, emb_final, _, _ = read_acts(snaps[-1])
        key_ks = np.sort(np.argsort(embedding_freq_power(emb_final))[::-1][:KEY_K] + 1)
        Xf = fourier_features(a, b, key_ks)
        Xoh = np.zeros((n, 2 * P)); Xoh[idx, a] = 1.0; Xoh[idx, P + b] = 1.0
        Rp = rng.standard_normal((2 * P, Xf.shape[1])) / np.sqrt(2 * P)
        Xrand = Xoh @ Rp
        new: dict[str, list[float]] = {k: [] for k in
            ("steps", "effrank_acts", "effrank_logits", "fourier_share",
             "r2_fourier", "r2_onehot", "r2_random", "r2_group", "r2_pca", "r2_randn")}
        for path in todo:
            step, emb, acts, logits = read_acts(path)
            new["steps"].append(step)
            _, er = spectral_stats(acts);   new["effrank_acts"].append(er)
            _, er = spectral_stats(logits); new["effrank_logits"].append(er)
            fp = embedding_freq_power(emb)
            new["fourier_share"].append(fp[key_ks - 1].sum() / max(fp.sum(), 1e-30))
            new["r2_fourier"].append(ridge_r2(Xf, acts, tr_i, te_i))
            new["r2_onehot"].append(ridge_r2(Xoh, acts, tr_i, te_i))
            new["r2_random"].append(ridge_r2(Xrand, acts, tr_i, te_i))
            G = group_means(acts, N_GROUPS)
            new["r2_group"].append(ridge_r2(G, acts, tr_i, te_i))
            Ac = acts - acts.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Ac, full_matrices=False)
            new["r2_pca"].append(ridge_r2(U[:, :N_GROUPS] * S[:N_GROUPS], acts, tr_i, te_i))
            rn = rng.choice(acts.shape[1], N_GROUPS, replace=False)
            new["r2_randn"].append(ridge_r2(acts[:, rn], acts, tr_i, te_i))
            print(f"  snap {step}: effrank={new['effrank_acts'][-1]:.1f}")
        for k in new:
            cache[k] = np.concatenate([cache.get(k, np.array([])), np.array(new[k], float)])
        order = np.argsort(cache["steps"])
        cache = {k: v[order] for k, v in cache.items()}
        np.savez(cache_path, **cache)
    return cache


def best_lag_corr(x_steps, x, ref_steps, ref, lags, smooth_rows: int):
    xi = np.interp(ref_steps, x_steps, x)
    dxi = smooth(np.gradient(xi, ref_steps), smooth_rows)
    best = (0, 0.0)
    for lag in lags:
        shifted = np.interp(ref_steps, ref_steps - lag, dxi)
        c = np.corrcoef(shifted, ref)[0, 1]
        if np.isfinite(c) and c > best[1]:
            best = (int(lag), float(c))
    return best


def row_centers(n: int, vs: float) -> list[float]:
    h = (1 - (n - 1) * vs) / n
    return [1 - (i - 1) * (h + vs) - h / 2 for i in range(1, n + 1)]


def main() -> int:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_nanda_grokking_v3")
    cols = load_csv(ckpt / "metrics.csv")
    steps = cols["step"]
    row_dt = float(np.median(np.diff(steps)))
    w500 = max(int(500 / row_dt), 1)
    w100 = max(int(100 / row_dt), 1)

    acc_s = smooth(cols["val_acc"], w100)
    slope_s = smooth(np.gradient(acc_s, steps) * 100.0, w100)
    grok_step = float(steps[int(np.argmax(slope_s))])

    sm = snap_metrics(ckpt)

    # ── Section A: time series with per-row legends ──────────────────────────
    NROWS, VS = 6, 0.045
    figA = make_subplots(rows=NROWS, cols=1, shared_xaxes=True, vertical_spacing=VS,
                         specs=[[{"secondary_y": True}]] * NROWS,
                         subplot_titles=(
                             "Grokking signature",
                             "Val-accuracy slope",
                             "Finding vs honing — output-space accord @ val refs",
                             "Loss-space work + gradient alignment",
                             "Spectral crystallization",
                             "Causal emergence — macro description power"))
    centers = row_centers(NROWS, VS)

    def tr(row, x, y, name, secondary=False, **kw):
        lg = "legend" if row == 1 else f"legend{row}"
        figA.add_trace(go.Scatter(x=x, y=y, name=name, legend=lg, **kw),
                       row=row, col=1, secondary_y=secondary)

    tr(1, steps, cols["train_loss"], "train loss", line=dict(color="#1f77b4", width=1.5))
    tr(1, steps, cols["val_loss"], "val loss", line=dict(color="#d62728", width=1.5))
    tr(1, steps, cols["train_acc"], "train acc", True, line=dict(color="#1f77b4", dash="dot"))
    tr(1, steps, cols["val_acc"], "val acc", True, line=dict(color="#2ca02c", width=2))
    figA.update_yaxes(type="log", title_text="loss", row=1, col=1, secondary_y=False)
    figA.update_yaxes(title_text="acc %", range=[0, 102], row=1, col=1, secondary_y=True)

    tr(2, steps, slope_s, "val-acc slope (Δ%/100 steps)", line=dict(color="#2ca02c", width=2))
    figA.update_yaxes(title_text="slope", row=2, col=1)

    tr(3, steps, cols["jva_w500"], "total accord w500", line=dict(color="#ff7f0e", width=2))
    tr(3, steps, cols["va_shape_w500"], "SHAPE accord w500 — finding",
       line=dict(color="#2ca02c", width=2))
    tr(3, steps, smooth(cols["va_net_par"], w500), "melt rate ⟨net,F̂⟩ — honing", True,
       line=dict(color="#9467bd", width=1.5))
    figA.update_yaxes(title_text="accord", row=3, col=1, secondary_y=False)
    figA.update_yaxes(title_text="signed ∥ movement", row=3, col=1, secondary_y=True)

    tr(4, steps, smooth(cols["ltr_accord"], w100), "train loss-work accord (w≈100)",
       line=dict(color="#1f77b4", width=1.5))
    tr(4, steps, smooth(cols["lva_accord"], w100), "val loss-work accord (w≈100)",
       line=dict(color="#d62728", width=1.5))
    tr(4, steps, smooth(cols["cos_gtr_gva"], w100), "cos(∇L_train, ∇L_val)", True,
       line=dict(color="#2ca02c", width=1.5))
    figA.update_yaxes(title_text="loss-work accord", row=4, col=1, secondary_y=False)
    figA.update_yaxes(title_text="cosine", range=[-1.05, 1.05], row=4, col=1, secondary_y=True)

    if len(sm.get("steps", [])) >= 2:
        ss = sm["steps"]
        tr(5, ss, sm["effrank_acts"], "effective rank — readout acts",
           line=dict(color="#1f77b4", width=2), mode="lines+markers")
        tr(5, ss, sm["effrank_logits"], "effective rank — logits",
           line=dict(color="#d62728", width=1.5), mode="lines+markers")
        tr(5, ss, sm["fourier_share"], "embedding Fourier share", True,
           line=dict(color="#2ca02c", width=2), mode="lines+markers")
        figA.update_yaxes(title_text="dims", row=5, col=1, secondary_y=False)
        figA.update_yaxes(title_text="share", range=[0, 1.02], row=5, col=1, secondary_y=True)

        tr(6, ss, sm["r2_fourier"], "R² Fourier basis (ground truth)",
           line=dict(color="#2ca02c", width=2), mode="lines+markers")
        tr(6, ss, sm["r2_onehot"], "R² additive one-hot (micro)",
           line=dict(color="#1f77b4", width=1.5), mode="lines+markers")
        tr(6, ss, sm["r2_random"], "R² random projection",
           line=dict(color="#7f7f7f", dash="dot"), mode="lines+markers")
        tr(6, ss, sm["r2_group"], "R² 8 group-means (discovered)",
           line=dict(color="#ff7f0e", width=2), mode="lines+markers")
        tr(6, ss, sm["r2_pca"], "R² top-8 PCs", line=dict(color="#9467bd"),
           mode="lines+markers")
        tr(6, ss, sm["r2_randn"], "R² 8 random neurons",
           line=dict(color="#7f7f7f", dash="dash"), mode="lines+markers")
        figA.update_yaxes(title_text="test R²", range=[-0.05, 1.02], row=6, col=1)

    figA.add_vline(x=grok_step, line_dash="dash", line_color="black", opacity=0.6)
    legends = {("legend" if i == 1 else f"legend{i}"):
               dict(x=1.01, y=centers[i - 1], xanchor="left", yanchor="middle",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.7)")
               for i in range(1, NROWS + 1)}
    figA.update_layout(height=2100, width=1250, margin=dict(r=290),
                       hoverlabel=dict(namelength=-1), hovermode="x unified",
                       title=f"Grokking instrument panel — {ckpt.name} "
                             f"(grok = peak val slope @ step {int(grok_step)})",
                       **legends)
    figA.update_xaxes(title_text="training step", row=NROWS, col=1)

    # Lead/lag table.
    lags = np.arange(-800, 801, 40)
    cands = [("shape accord w500", steps, cols["va_shape_w500"]),
             ("net accord w500", steps, cols["jva_w500"]),
             ("train loss-work accord", steps, smooth(cols["ltr_accord"], w500)),
             ("val loss-work accord", steps, smooth(cols["lva_accord"], w500)),
             ("cos(g_train, g_val)", steps, smooth(cols["cos_gtr_gva"], w500))]
    if len(sm.get("steps", [])) >= 4:
        cands += [("R² Fourier basis", sm["steps"], sm["r2_fourier"]),
                  ("R² 8 group-means", sm["steps"], sm["r2_group"]),
                  ("−effective rank", sm["steps"], -sm["effrank_acts"]),
                  ("Fourier share", sm["steps"], sm["fourier_share"])]
    table = [(n, *best_lag_corr(xs, xv, steps, slope_s, lags, w100)) for n, xs, xv in cands]
    for n, lag, c in table:
        print(f"{n:26s} lag {lag:+5d}  corr {c:+.3f}")

    parts = [figA.to_html(include_plotlyjs="inline", full_html=False)]
    parts.insert(0, f"<p style='max-width:1100px;font-family:sans-serif'>{CAPTIONS['A']} "
                    f"{CAPTIONS['cos']}</p>")

    # ── Section B: leverage ──────────────────────────────────────────────────
    sp_path = ckpt / "structural_potential.bin"
    rl_path = ckpt / "leverage_realized.bin"
    if sp_path.exists() and rl_path.exists():
        sp = read_sp(sp_path)
        lsteps, realized = read_realized(rl_path)
        sizes = [int(x) for x in (ckpt / "param_manifest.txt").read_text().split()]
        names = TENSOR_NAMES if len(TENSOR_NAMES) == len(sizes) else \
            [f"t{i}" for i in range(len(sizes))]
        bounds = np.cumsum([0] + sizes)
        tensor_of = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])

        order = np.argsort(realized[-1])[::-1]
        n_rows = 600
        sel = order[np.linspace(0, len(order) - 1, n_rows).astype(int)]

        real_pct = pct_rank(realized)[:, sel].T                 # (rows, snaps)
        pot_pct = pct_rank(sp[None, :])[0][sel][:, None]        # (rows, 1)

        figB = make_subplots(
            rows=2, cols=2, column_widths=[0.9, 0.1], row_heights=[0.62, 0.38],
            horizontal_spacing=0.02, vertical_spacing=0.14,
            specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, None]],
            subplot_titles=("realized |J_i| — within-snapshot percentile",
                            "potential (arch prior)",
                            "leverage concentration"))
        figB.add_trace(go.Heatmap(z=real_pct, x=lsteps, colorscale="Magma",
                                  zmin=0, zmax=1, colorbar=dict(title="pct", x=1.06)),
                       row=1, col=1)
        figB.add_trace(go.Heatmap(z=pot_pct, x=[0], colorscale="Magma",
                                  zmin=0, zmax=1, showscale=False), row=1, col=2)
        figB.update_yaxes(autorange="reversed", title_text="params (sorted by final realized)",
                          row=1, col=1)
        figB.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
        figB.update_xaxes(showticklabels=False, row=1, col=2)
        figB.update_xaxes(title_text="training step", row=1, col=1)

        ginis = np.array([gini(r) for r in realized])
        figB.add_trace(go.Scatter(x=lsteps, y=ginis, name="Gini(realized leverage)",
                                  line=dict(color="#d62728", width=2)), row=2, col=1)
        lam = realized ** 2
        partf = (lam.sum(1) ** 2 / (lam ** 2).sum(1)) / realized.shape[1]
        figB.add_trace(go.Scatter(x=lsteps, y=partf, name="participation fraction",
                                  line=dict(color="#1f77b4", width=2)),
                       row=2, col=1, secondary_y=True)
        figB.update_yaxes(title_text="Gini", range=[0, 1], row=2, col=1, secondary_y=False)
        figB.update_yaxes(title_text="participating fraction", type="log",
                          row=2, col=1, secondary_y=True)
        figB.add_vline(x=grok_step, line_dash="dash", line_color="white",
                       opacity=0.8, row=1, col=1)
        figB.add_vline(x=grok_step, line_dash="dash", line_color="black",
                       opacity=0.6, row=2, col=1)
        figB.update_layout(height=850, width=1250, hoverlabel=dict(namelength=-1),
                           title="Section B — realized leverage vs structural potential",
                           legend=dict(x=1.01, y=0.15, xanchor="left"),
                           margin=dict(r=290))

        # Traces + leaderboard.
        def label_of(i: int) -> str:
            t = tensor_of[i]
            return f"{names[t]}[{i - bounds[t]}]"

        eps = 1e-12
        ratio = realized / (sp[None, :] + eps)
        risers = order[:10]
        early_strong = np.where(realized[0] >= np.quantile(realized[0], 0.9))[0]
        decline = (ratio[-1, early_strong] + eps) / (ratio[0, early_strong] + eps)
        fallers = early_strong[np.argsort(decline)[:10]]

        figT = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                             subplot_titles=("stepping up — |J_i|(t), top-10 final realized",
                                             "stepping back — early strong, largest decline"))
        for i in risers:
            figT.add_trace(go.Scatter(x=lsteps, y=realized[:, i], name=label_of(i),
                                      legend="legend"), row=1, col=1)
        for i in fallers:
            figT.add_trace(go.Scatter(x=lsteps, y=realized[:, i], name=label_of(i),
                                      legend="legend2"), row=1, col=2)
        figT.update_yaxes(type="log", title_text="|J_i|", row=1, col=1)
        figT.update_yaxes(type="log", row=1, col=2)
        figT.add_vline(x=grok_step, line_dash="dash", line_color="black", opacity=0.6)
        figT.update_layout(height=460, width=1250, hoverlabel=dict(namelength=-1),
                           legend=dict(x=0.0, y=-0.35, orientation="h", font=dict(size=9)),
                           legend2=dict(x=0.55, y=-0.35, orientation="h", font=dict(size=9)),
                           margin=dict(b=160),
                           title="Per-param Fisher-metric diagonal over time")

        TOP = 30
        top_idx = order[:TOP]
        labels = [label_of(i) for i in top_idx]
        figL = go.Figure(go.Bar(
            x=realized[-1, top_idx][::-1], y=labels[::-1], orientation="h",
            marker=dict(color=sp[top_idx][::-1], colorscale="Viridis",
                        colorbar=dict(title="potential")),
        ))
        figL.update_xaxes(type="log", title_text="final realized |J_i| (log)")
        figL.update_layout(height=650, width=1250,
                           title=f"Leading realized-leverage params (top {TOP}) — "
                                 "bar color = architectural potential")

        parts.append(f"<h2 style='font-family:sans-serif'>Section B — leverage</h2>"
                     f"<p style='max-width:1100px;font-family:sans-serif'>{CAPTIONS['B']}</p>")
        parts.append(figB.to_html(include_plotlyjs=False, full_html=False))
        parts.append(figT.to_html(include_plotlyjs=False, full_html=False))
        parts.append(figL.to_html(include_plotlyjs=False, full_html=False))

    # ── Section C: collaboration matrices ────────────────────────────────────
    jc_path = ckpt / "j_columns.bin"
    if jc_path.exists():
        jsteps, jcols = read_jcols(jc_path)
        picks = sorted({0, int(np.searchsorted(jsteps, grok_step * 0.6)),
                        int(np.searchsorted(jsteps, grok_step)), len(jsteps) - 1})
        picks = [min(p, len(jsteps) - 1) for p in picks]
        figC = make_subplots(rows=1, cols=len(picks),
                             subplot_titles=[f"step {int(jsteps[p])}" for p in picks],
                             horizontal_spacing=0.03)
        for c, p in enumerate(picks, start=1):
            M = jcols[p]
            nrm = np.linalg.norm(M, axis=1, keepdims=True)
            C = (M / (nrm + 1e-12)) @ (M / (nrm + 1e-12)).T
            figC.add_trace(go.Heatmap(z=C, colorscale="RdBu", zmin=-1, zmax=1,
                                      showscale=(c == len(picks))), row=1, col=c)
            figC.update_yaxes(autorange="reversed", row=1, col=c,
                              showticklabels=(c == 1))
        figC.update_layout(height=420, width=1250,
                           title="Section C — collaboration: cos(J_i, J_j) among top-50 params")
        parts.append(f"<h2 style='font-family:sans-serif'>Section C — collaboration</h2>"
                     f"<p style='max-width:1100px;font-family:sans-serif'>{CAPTIONS['C']}</p>")
        parts.append(figC.to_html(include_plotlyjs=False, full_html=False))

    # Lead/lag table.
    rows_html = "".join(f"<tr><td>{n}</td><td style='text-align:right'>{lag:+d}</td>"
                        f"<td style='text-align:right'>{c:+.3f}</td></tr>"
                        for n, lag, c in table)
    parts.append(
        "<div style='max-width:700px;margin:1em auto;font-family:sans-serif'>"
        "<h3>Lead/lag vs val-accuracy slope</h3>"
        "<p>Best cross-correlation of each metric's <i>rate of change</i> against the "
        "val-accuracy slope, lags ±800 steps. Negative lag = leading indicator.</p>"
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
        "<tr><th>metric</th><th>best lag</th><th>corr</th></tr>"
        f"{rows_html}</table></div>")

    out = ckpt / "report.html"
    out.write_text("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                   "<title>Grokking instrument panel</title></head><body>"
                   + "\n".join(parts) + "</body></html>")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
