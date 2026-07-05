#!/usr/bin/env python3
"""Realized vs potential leverage over training — where the circuit lives.

Reads (produced by ./nanda_leverage <ckpt_dir>):
    <ckpt_dir>/structural_potential.bin   Metric III — E_init ||∂F/∂θ_i||
    <ckpt_dir>/leverage_realized.bin      Metric II per snapshot — ||∂F/∂θ_i|| at trained weights
    <ckpt_dir>/param_manifest.txt         flat Param tensor sizes (all_params order)
    <ckpt_dir>/metrics.csv                (optional) grok marker

Writes <ckpt_dir>/leverage.png:
    1. Heatmap: per-tensor mean realization ratio (realized/potential) × time
    2. Heatmap: per-param realization ratio × time — params sorted by final ratio,
       downsampled rows (the "which params orchestrate the algorithm" picture)
    3. Concentration of realized leverage over time: participation ratio + Gini
       (scale-invariant, so weight decay's global shrink doesn't fake the signal)

Usage: python3 tools/leverage_heatmap.py [ckpt_dir]
"""

from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Param tensor names for NandaNet, in all_params() order (see nanda_net.hpp).
# LayerNorm blocks expose (gamma, beta); MapDense/Dense expose (W, b);
# attention exposes (Q, K, V, O); ResidualBlock/BlockSequence concatenate inner params.
TENSOR_NAMES = [
    "embed.W", "embed.b",
    "pos_emb",
    "ln1.gamma", "ln1.beta",
    "attn.Q", "attn.K", "attn.V", "attn.O",
    "ln2.gamma", "ln2.beta",
    "ffn_in.W", "ffn_in.b",
    "ffn_out.W", "ffn_out.b",
    "unembed.W", "unembed.b",
]


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
    return np.array(steps), np.stack(mats)          # (n_snaps,), (n_snaps, P)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x))
    n = len(x)
    s = x.sum()
    if s <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def participation(x: np.ndarray) -> float:
    lam = x * x
    s = lam.sum()
    return float(s * s / (lam * lam).sum() / len(x)) if s > 0 else 0.0  # fraction of params


def main() -> int:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_nanda_grokking_v3")
    sp = read_sp(ckpt / "structural_potential.bin")
    steps, realized = read_realized(ckpt / "leverage_realized.bin")
    sizes = [int(l) for l in (ckpt / "param_manifest.txt").read_text().split()]
    assert sum(sizes) == len(sp), f"manifest {sum(sizes)} != P {len(sp)}"
    names = TENSOR_NAMES if len(TENSOR_NAMES) == len(sizes) else [f"t{i}" for i in range(len(sizes))]

    grok_step = None
    mcsv = ckpt / "metrics.csv"
    if mcsv.exists():
        with mcsv.open() as f:
            for row in csv.DictReader(f):
                if float(row["val_acc"]) >= 90.0:
                    grok_step = float(row["step"])
                    break

    eps = 1e-12
    ratio = realized / (sp[None, :] + eps)           # (n_snaps, P)

    # 1 — per-tensor mean ratio heatmap.
    bounds = np.cumsum([0] + sizes)
    tensor_ratio = np.stack([
        ratio[:, bounds[i]:bounds[i + 1]].mean(axis=1) for i in range(len(sizes))])

    # 2 — per-param heatmap, sorted by final ratio, downsampled to ~800 rows.
    order = np.argsort(ratio[-1])[::-1]
    n_rows = min(800, ratio.shape[1])
    sel = order[np.linspace(0, len(order) - 1, n_rows).astype(int)]
    param_ratio = ratio[:, sel].T                     # (n_rows, n_snaps)

    # 3 — concentration of realized leverage (scale-invariant).
    ginis = np.array([gini(r) for r in realized])
    parts = np.array([participation(r) for r in realized])

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.6, 1.0], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(np.log10(tensor_ratio + eps), aspect="auto", cmap="magma",
                   extent=[steps[0], steps[-1], len(sizes) - 0.5, -0.5])
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("log10 realization ratio (realized / structural potential) — per tensor")
    fig.colorbar(im, ax=ax, shrink=0.9)
    if grok_step: ax.axvline(grok_step, color="w", ls="--", lw=1)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(np.log10(param_ratio + eps), aspect="auto", cmap="magma",
                   extent=[steps[0], steps[-1], n_rows, 0])
    ax.set_ylabel("params (sorted by final ratio)")
    ax.set_title("log10 realization ratio — per parameter (the circuit lights up at the top)")
    fig.colorbar(im, ax=ax, shrink=0.9)
    if grok_step: ax.axvline(grok_step, color="w", ls="--", lw=1)

    ax = fig.add_subplot(gs[2])
    ax.plot(steps, ginis, color="C3", lw=2, marker="o", ms=3, label="Gini(realized leverage)")
    ax.set_ylabel("Gini", color="C3")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(steps, parts, color="C0", lw=2, marker="o", ms=3,
             label="participation fraction")
    ax2.set_ylabel("participating fraction of params", color="C0")
    ax2.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_title("Leverage concentration — functional weight collapsing onto the circuit")
    ax.grid(alpha=0.3)
    if grok_step: ax.axvline(grok_step, color="k", ls="--", lw=1, alpha=0.6)
    ln1, l1 = ax.get_legend_handles_labels()
    ln2, l2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, l1 + l2, loc="center right", fontsize=9)

    out = ckpt / "leverage.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # ── Twin view: realization ratio × time beside the architectural prior ────
    # Same row order in both panels, so the eye can ask directly: are the params
    # that end up carrying the function the ones the architecture favored?
    tensor_of = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    sp_norm = sp / sp.max()                                    # 0-1 normalized potential

    fig2 = plt.figure(figsize=(13, 12))
    gs2 = fig2.add_gridspec(2, 2, width_ratios=[8, 1], height_ratios=[1.6, 1.0],
                            hspace=0.3, wspace=0.06)

    ratio_norm = ratio / ratio.max()                           # global 0-1
    ax = fig2.add_subplot(gs2[0, 0])
    im = ax.imshow(np.log10(ratio_norm[:, sel].T + eps), aspect="auto", cmap="magma",
                   extent=[steps[0], steps[-1], n_rows, 0])
    ax.set_ylabel("params (sorted by final realization ratio)")
    ax.set_title("log10 normalized realization ratio (realized / potential) × time")
    ax.set_xlabel("training step")
    fig2.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    if grok_step: ax.axvline(grok_step, color="w", ls="--", lw=1)

    axs = fig2.add_subplot(gs2[0, 1], sharey=ax)
    strip = sp_norm[sel][:, None]
    ims = axs.imshow(np.log10(strip + eps), aspect="auto", cmap="viridis",
                     extent=[0, 1, n_rows, 0])
    axs.set_xticks([])
    axs.set_title("potential\n(arch prior)", fontsize=9)
    plt.setp(axs.get_yticklabels(), visible=False)
    fig2.colorbar(ims, ax=axs, shrink=0.85, pad=0.15)

    # Leaderboard: the potential-maximizers — top params by final realization ratio.
    TOP = 30
    top_idx = order[:TOP]
    within = top_idx - np.array([bounds[tensor_of[i]] for i in top_idx])
    labels = [f"{names[tensor_of[i]]}[{w}]" for i, w in zip(top_idx, within)]
    axb = fig2.add_subplot(gs2[1, :])
    colors = plt.cm.viridis(np.log10(sp_norm[top_idx] + eps) /
                            np.log10(sp_norm[top_idx] + eps).min())
    axb.barh(range(TOP), ratio[-1, top_idx], color=colors)
    axb.set_yticks(range(TOP))
    axb.set_yticklabels(labels, fontsize=7)
    axb.invert_yaxis()
    axb.set_xscale("log")
    axb.set_xlabel("final realization ratio (log)")
    axb.set_title(f"Leading potential-maximizers — top {TOP} params by realized/potential "
                  "(bar color = architectural potential, darker = higher prior)")
    axb.grid(alpha=0.3, axis="x")

    out2 = ckpt / "leverage_twin.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"wrote {out2}")

    # ── Per-param Jacobian trajectories: steppers-up and steppers-back ────────
    # |J_i|(t) is the diagonal of the Fisher metric evolving — influence-space
    # bending under this network's values. Top panel: the 10 params with the
    # highest final realization ratio. Bottom: params that were important early
    # (top-decile realized leverage at the first snapshot) and stepped back most.
    def label_of(i: int) -> str:
        t = tensor_of[i]
        return f"{names[t]}[{i - bounds[t]}]"

    risers = order[:10]
    early_strong = np.where(realized[0] >= np.quantile(realized[0], 0.9))[0]
    decline = (ratio[-1, early_strong] + eps) / (ratio[0, early_strong] + eps)
    fallers = early_strong[np.argsort(decline)[:10]]

    fig3, axes3 = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cmap = plt.cm.tab10
    for k, i in enumerate(risers):
        axes3[0].plot(steps, realized[:, i] + eps, lw=1.6, color=cmap(k % 10),
                      label=label_of(i))
    axes3[0].set_yscale("log")
    axes3[0].set_ylabel("|J_i| (realized leverage)")
    axes3[0].set_title("Stepping up — |J_i|(t) for the top-10 final realization ratios")
    axes3[0].legend(fontsize=7, ncol=2)
    axes3[0].grid(alpha=0.3)

    for k, i in enumerate(fallers):
        axes3[1].plot(steps, realized[:, i] + eps, lw=1.6, color=cmap(k % 10),
                      label=label_of(i))
    axes3[1].set_yscale("log")
    axes3[1].set_ylabel("|J_i| (realized leverage)")
    axes3[1].set_xlabel("training step")
    axes3[1].set_title("Stepping back — early top-decile params with the largest ratio decline")
    axes3[1].legend(fontsize=7, ncol=2)
    axes3[1].grid(alpha=0.3)

    if grok_step:
        for a in axes3:
            a.axvline(grok_step, color="k", ls="--", lw=1, alpha=0.6)

    out3 = ckpt / "leverage_traces.png"
    plt.savefig(out3, dpi=140, bbox_inches="tight")
    print(f"wrote {out3}")

    print(f"\nfinal snapshot (step {steps[-1]}):")
    print(f"  Gini(realized) = {ginis[-1]:.3f}   (init: {ginis[0]:.3f})")
    print(f"  participation  = {parts[-1]*100:.1f}% of params (init: {parts[0]*100:.1f}%)")
    for i, nm in enumerate(names):
        r = ratio[-1, bounds[i]:bounds[i + 1]]
        print(f"  {nm:12s} median ratio {np.median(r):8.3f}   max {r.max():8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
