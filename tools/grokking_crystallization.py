#!/usr/bin/env python3
"""Spectral crystallization + causal-emergence analysis over grokking snapshots.

Consumes the retained per-SnapEvery artifacts written by nanda_grokking (v3):
    <ckpt_dir>/snaps/acts_<step>.bin   header, embedding weights, readout acts, logits
    <ckpt_dir>/metrics.csv             (optional) for the grok-step marker

Produces:
    <ckpt_dir>/crystallization.png
        1. effective rank / participation ratio of readout activations + logits vs step
           (the low-entropy manifold crystallizing)
        2. embedding Fourier concentration vs step (top-K frequency power share)
    <ckpt_dir>/emergence.png
        1. ground-truth macro basis: test R² predicting readout acts from
           Fourier features vs additive one-hot vs random projection
        2. discovered macro variables: test R² predicting all 128 neurons from
           8 group-means (correlation clusters) vs top-8 PCs vs 8 random neurons

Usage:
    python3 tools/grokking_crystallization.py [ckpt_dir]
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = 113
RIDGE_ALPHA = 1e-2
KEY_K = 8          # number of key frequencies for the Fourier macro basis
N_GROUPS = 8       # discovered macro variables
SPLIT_SEED = 7
TRAIN_FRAC = 0.8


def read_acts(path: Path):
    with path.open("rb") as f:
        step, n, adim, ldim, vocab, edim = struct.unpack("<6Q", f.read(48))
        emb = np.frombuffer(f.read(4 * edim * vocab), dtype=np.float32).reshape(edim, vocab)
        acts = np.frombuffer(f.read(4 * n * adim), dtype=np.float32).reshape(n, adim)
        logits = np.frombuffer(f.read(4 * n * ldim), dtype=np.float32).reshape(n, ldim)
    return int(step), emb, acts.astype(np.float64), logits.astype(np.float64)


def spectral_stats(X: np.ndarray) -> tuple[float, float]:
    """(participation ratio, effective rank e^H) of the covariance spectrum."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s * s
    tot = lam.sum()
    if tot <= 0:
        return 0.0, 0.0
    p = lam / tot
    pr = tot * tot / (lam * lam).sum()
    h = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(pr), float(np.exp(h))


def embedding_freq_power(emb: np.ndarray) -> np.ndarray:
    """Power per frequency k=1..56 of the token->embedding map (EQ token excluded)."""
    tokens = emb[:, :P].T                     # (113, edim)
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    F = np.fft.rfft(tokens, axis=0)           # (57, edim)
    power = (np.abs(F) ** 2).sum(axis=1)
    return power[1:]                           # drop DC → k = 1..56


def read_snap_weights(snap_path: Path, manifest_sizes: list[int]):
    """Extract (embed_W [128,114], unembed_W [113,128]) from a raw weights-only
    snapshot (flat float32 in all_params order; embed.W first, unembed.W second-to-last)."""
    flat = np.fromfile(snap_path, dtype=np.float32)
    assert flat.size == sum(manifest_sizes), f"{snap_path}: {flat.size} != {sum(manifest_sizes)}"
    emb = flat[:manifest_sizes[0]].reshape(128, P + 1).astype(np.float64)
    off = sum(manifest_sizes[:-2])
    unemb = flat[off:off + manifest_sizes[-2]].reshape(P, 128).astype(np.float64)
    return emb, unemb


def unembedding_freq_power(unemb: np.ndarray) -> np.ndarray:
    """Power per frequency k=1..56 of the candidate->unembedding-row map.
    Rows are indexed by answer c = 0..112; FFT along the candidate axis mirrors
    embedding_freq_power along the token axis."""
    rows = unemb - unemb.mean(axis=0, keepdims=True)   # (113, 128)
    F = np.fft.rfft(rows, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    return power[1:]                                    # k = 1..56


def fourier_features(a: np.ndarray, b: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Ground-truth macro basis: additive features per input plus sum features
    cos/sin(2πk(a+b)/p) — the representation the grokked circuit actually carries."""
    cols = []
    for k in ks:
        wa = 2 * np.pi * k * a / P
        wb = 2 * np.pi * k * b / P
        ws = 2 * np.pi * k * (a + b) / P
        cols += [np.cos(wa), np.sin(wa), np.cos(wb), np.sin(wb), np.cos(ws), np.sin(ws)]
    return np.stack(cols, axis=1)


def ridge_r2(X: np.ndarray, Y: np.ndarray, train_idx, test_idx, alpha=RIDGE_ALPHA) -> float:
    """Mean test R² over targets for closed-form ridge (with intercept)."""
    Xtr, Xte = X[train_idx], X[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]
    xm, ym = Xtr.mean(axis=0), Ytr.mean(axis=0)
    Xtr = Xtr - xm; Xte = Xte - xm
    Ytr = Ytr - ym; Yte = Yte - ym
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr)
    resid = Yte - Xte @ W
    sse = (resid ** 2).sum(axis=0)
    sst = (Yte ** 2).sum(axis=0)
    ok = sst > 1e-12
    return float(np.mean(1.0 - sse[ok] / sst[ok])) if ok.any() else 0.0


def group_means(acts: np.ndarray, n_groups: int) -> np.ndarray:
    """Discovered macro variables: cluster neurons by top-PC loading, take group means."""
    Ac = acts - acts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    assign = np.abs(Vt[:n_groups]).argmax(axis=0)   # neuron -> strongest component
    G = np.zeros((acts.shape[0], n_groups))
    for g in range(n_groups):
        members = assign == g
        G[:, g] = Ac[:, members].mean(axis=1) if members.any() else 0.0
    return G


def main() -> int:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_nanda_grokking_v3")
    snaps = sorted(ckpt.glob("snaps/acts_*.bin"),
                   key=lambda p: int(re.search(r"acts_(\d+)", p.name).group(1)))
    if not snaps:
        print(f"no snapshots under {ckpt}/snaps", file=sys.stderr)
        return 1

    # Grok marker from metrics.csv if available.
    grok_step = None
    mcsv = ckpt / "metrics.csv"
    if mcsv.exists():
        import csv as _csv
        with mcsv.open() as f:
            rd = _csv.DictReader(f)
            for row in rd:
                if float(row["val_acc"]) >= 90.0:
                    grok_step = float(row["step"])
                    break

    # Input coordinates in canonical order i = a*P + b.
    n = P * P
    idx = np.arange(n)
    a, b = idx // P, idx % P

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    ntr = int(TRAIN_FRAC * n)
    train_idx, test_idx = perm[:ntr], perm[ntr:]

    # Fixed bases across snapshots for comparability.
    # Key frequencies from the FINAL snapshot's embedding (the formed circuit).
    _, emb_final, _, _ = read_acts(snaps[-1])
    freq_power_final = embedding_freq_power(emb_final)
    key_ks = np.sort(np.argsort(freq_power_final)[::-1][:KEY_K] + 1)
    print(f"key frequencies (final snapshot): {key_ks}")

    X_fourier = fourier_features(a, b, key_ks)                       # (n, 6K)
    X_onehot = np.zeros((n, 2 * P))
    X_onehot[idx, a] = 1.0
    X_onehot[idx, P + b] = 1.0                                        # additive micro basis
    R = rng.standard_normal((2 * P, X_fourier.shape[1])) / np.sqrt(2 * P)
    X_random = X_onehot @ R                                           # matched-width control

    steps, pr_acts, er_acts, pr_log, er_log, topk_share = [], [], [], [], [], []
    r2_fourier, r2_onehot, r2_random = [], [], []
    r2_group, r2_pca, r2_randn = [], [], []

    for path in snaps:
        step, emb, acts, logits = read_acts(path)
        steps.append(step)

        pr, er = spectral_stats(acts)
        pr_acts.append(pr); er_acts.append(er)
        pr, er = spectral_stats(logits)
        pr_log.append(pr); er_log.append(er)

        fp = embedding_freq_power(emb)
        topk_share.append(fp[key_ks - 1].sum() / max(fp.sum(), 1e-30))

        # Ground-truth macro vs micro vs random.
        r2_fourier.append(ridge_r2(X_fourier, acts, train_idx, test_idx))
        r2_onehot.append(ridge_r2(X_onehot, acts, train_idx, test_idx))
        r2_random.append(ridge_r2(X_random, acts, train_idx, test_idx))

        # Discovered macro variables: predict all neurons from few summaries.
        G = group_means(acts, N_GROUPS)
        r2_group.append(ridge_r2(G, acts, train_idx, test_idx))
        Ac = acts - acts.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Ac, full_matrices=False)
        r2_pca.append(ridge_r2(U[:, :N_GROUPS] * S[:N_GROUPS], acts, train_idx, test_idx))
        rand_neurons = rng.choice(acts.shape[1], N_GROUPS, replace=False)
        r2_randn.append(ridge_r2(acts[:, rand_neurons], acts, train_idx, test_idx))

        print(f"step {step:6d}  effrank(acts)={er_acts[-1]:7.2f}  "
              f"fourier_share={topk_share[-1]:.3f}  "
              f"R2[fourier]={r2_fourier[-1]:.3f}  R2[group8]={r2_group[-1]:.3f}")

    def mark(ax):
        if grok_step is not None:
            ax.axvline(grok_step, color="k", ls="--", lw=1.0, alpha=0.6)

    # ── crystallization.png ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(steps, er_acts, color="C0", lw=1.6, label="effective rank — readout acts (128d)")
    ax.plot(steps, pr_acts, color="C0", lw=1.2, ls=":", label="participation ratio — acts")
    ax.plot(steps, er_log, color="C3", lw=1.6, label="effective rank — logits (113d)")
    ax.plot(steps, pr_log, color="C3", lw=1.2, ls=":", label="participation ratio — logits")
    ax.set_ylabel("dimensionality")
    ax.set_title("Spectral crystallization — the manifold collapsing")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); mark(ax)

    ax = axes[1]
    ax.plot(steps, topk_share, color="C2", lw=1.8)
    ax.set_ylabel(f"top-{KEY_K} frequency power share")
    ax.set_xlabel("training step")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Embedding Fourier concentration (key ks = {list(key_ks)})")
    ax.grid(alpha=0.3); mark(ax)
    plt.tight_layout()
    out1 = ckpt / "crystallization.png"
    plt.savefig(out1, dpi=140); plt.close(fig)

    # ── emergence.png ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(steps, r2_fourier, color="C2", lw=1.8,
            label=f"Fourier macro basis ({X_fourier.shape[1]}d, ground truth)")
    ax.plot(steps, r2_onehot, color="C0", lw=1.6,
            label="additive one-hot (226d, micro)")
    ax.plot(steps, r2_random, color="C7", lw=1.2, ls=":",
            label=f"random projection ({X_fourier.shape[1]}d, control)")
    ax.set_ylabel("test R² on readout acts")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Causal emergence, ground-truth basis: macro description beats micro")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); mark(ax)

    ax = axes[1]
    ax.plot(steps, r2_group, color="C1", lw=1.8, label=f"{N_GROUPS} group-means (discovered circuits)")
    ax.plot(steps, r2_pca, color="C4", lw=1.4, label=f"top-{N_GROUPS} PCs (optimal linear summary)")
    ax.plot(steps, r2_randn, color="C7", lw=1.2, ls=":", label=f"{N_GROUPS} random neurons")
    ax.set_ylabel("test R² on all 128 neurons")
    ax.set_xlabel("training step")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Causal emergence, discovered: few macro variables profile the layer")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); mark(ax)
    plt.tight_layout()
    out2 = ckpt / "emergence.png"
    plt.savefig(out2, dpi=140); plt.close(fig)

    print(f"wrote {out1}")
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
