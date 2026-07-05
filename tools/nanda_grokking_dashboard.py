#!/usr/bin/env python3
"""Plot Nanda grokking metrics: canonical grokking curve alongside movement metrics.

Panels:
  1. Grokking signature   — train/val loss (log) + train/val accuracy
  2. Output-space accord @ VAL refs   — inst, w100, w500, w2000, cumulative
  3. Output-space accord @ TRAIN refs — inst, w100, w500, w2000, cumulative
  4. Loss-space           — train/val loss-work accords + cos(g_train, g_val)
  5. Parameter-space      — cumulative efficiency + cumulative output-space GROSS

Reads:
    checkpoints_nanda_grokking/metrics.csv

Writes:
    checkpoints_nanda_grokking/dashboard.png

Usage:
    python3 tools/nanda_grokking_dashboard.py [checkpoints_dir]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with csv_path.open() as f:
        r = csv.reader(f)
        header = next(r)
        for h in header:
            cols[h] = []
        for row in r:
            for h, v in zip(header, row):
                cols[h].append(float(v))
    return cols


def main() -> int:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_nanda_grokking")
    csv_path = ckpt / "metrics.csv"
    if not csv_path.exists():
        print(f"no metrics.csv at {csv_path}", file=sys.stderr)
        return 1
    cols = load(csv_path)
    if not cols["step"]:
        print("empty CSV", file=sys.stderr)
        return 1

    steps = cols["step"]

    # Grokking moment: first step where val accuracy crosses the threshold.
    GROK_ACC = 90.0
    grok_step = next((s for s, a in zip(steps, cols["val_acc"]) if a >= GROK_ACC), None)

    v2 = "jva_w500" in cols  # runner v2 columns present

    n_panels = 5 if v2 else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels), sharex=True)

    # 1) Grokking signature — train/val loss (log) + train/val accuracy.
    ax = axes[0]
    ax.set_yscale("log")
    ax.plot(steps, cols["train_loss"], label="train loss", color="C0", lw=1.4)
    ax.plot(steps, cols["val_loss"],   label="val loss",   color="C3", lw=1.4)
    ax.set_ylabel("loss (log)")
    ax.set_title("Grokking signature")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    if "train_acc" in cols:
        ax2.plot(steps, cols["train_acc"], label="train acc", color="C0",
                 lw=1.6, ls=":", alpha=0.85)
    ax2.plot(steps, cols["val_acc"], label="val acc", color="C2", lw=1.6, alpha=0.85)
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 102)
    ln1, l1 = ax.get_legend_handles_labels()
    ln2, l2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, l1 + l2, loc="center right", fontsize=9)

    if v2:
        # 2) Output-space accord at VAL reference inputs — the hypothesis panel.
        ax = axes[1]
        ax.plot(steps, cols["jva_inst"],  color="C7", lw=0.6, alpha=0.5, label="instantaneous")
        ax.plot(steps, cols["jva_w100"],  color="C1", lw=1.4, label="w=100")
        ax.plot(steps, cols["jva_w500"],  color="C3", lw=1.6, label="w=500")
        ax.plot(steps, cols["jva_w2000"], color="C4", lw=1.6, label="w=2000")
        ax.plot(steps, cols["jva_cum"],   color="k",  lw=1.2, ls="--", label="cumulative")
        if "va_shape_w500" in cols:
            ax.plot(steps, cols["va_shape_w500"], color="C2", lw=1.8,
                    label="SHAPE accord w=500 (⊥ to F̂ — algorithm channel)")
        ax.set_ylabel("accord ‖NET‖/‖GROSS‖")
        ax.set_title("Output-space accord @ validation refs (where the function moves during grokking)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # 3) Output-space accord at TRAIN reference inputs.
        ax = axes[2]
        ax.plot(steps, cols["jtr_inst"],  color="C7", lw=0.6, alpha=0.5, label="instantaneous")
        ax.plot(steps, cols["jtr_w100"],  color="C1", lw=1.4, label="w=100")
        ax.plot(steps, cols["jtr_w500"],  color="C3", lw=1.6, label="w=500")
        ax.plot(steps, cols["jtr_w2000"], color="C4", lw=1.6, label="w=2000")
        ax.plot(steps, cols["jtr_cum"],   color="k",  lw=1.2, ls="--", label="cumulative")
        ax.set_ylabel("accord ‖NET‖/‖GROSS‖")
        ax.set_title("Output-space accord @ train refs (function pinned here post-memorization)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # 4) Loss-space work accords + train/val gradient alignment.
        ax = axes[3]
        ax.plot(steps, cols["ltr_accord"], color="C0", lw=1.0, alpha=0.8,
                label="train loss-work accord (inst)")
        ax.plot(steps, cols["lva_accord"], color="C3", lw=1.0, alpha=0.8,
                label="val loss-work accord (inst)")
        ax.plot(steps, cols["ltr_cum_accord"], color="C0", lw=1.2, ls="--",
                label="train (cumulative)")
        ax.plot(steps, cols["lva_cum_accord"], color="C3", lw=1.2, ls="--",
                label="val (cumulative)")
        ax.set_ylabel("loss-work accord")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax3 = ax.twinx()
        ax3.plot(steps, cols["cos_gtr_gva"], color="C2", lw=1.4, alpha=0.9,
                 label="cos(∇L_train, ∇L_val)")
        ax3.axhline(0.0, color="C2", lw=0.5, alpha=0.4)
        ax3.set_ylabel("cos(g_train, g_val)")
        ax3.set_ylim(-1.05, 1.05)
        ax3.legend(loc="lower right", fontsize=9)
        ax.set_title("Loss-space movement + train/val gradient alignment")

        # 5) Parameter-space efficiency + cumulative output-space GROSS.
        ax = axes[4]
        ax.plot(steps, cols["param_eff"], color="C0", lw=1.4, label="param-space efficiency (cum)")
        ax.set_ylabel("efficiency")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax4 = ax.twinx()
        ax4.set_yscale("log")
        ax4.plot(steps, cols["jtr_gross_l1"], color="C4", lw=1.2, alpha=0.8,
                 label="output GROSS L1 (train refs)")
        ax4.plot(steps, cols["jva_gross_l1"], color="C5", lw=1.2, alpha=0.8,
                 label="output GROSS L1 (val refs)")
        ax4.set_ylabel("cumulative output-space GROSS (log)")
        ax4.legend(loc="lower right", fontsize=9)
        ax.set_title("Parameter-space efficiency + cumulative output-space movement")
        ax.set_xlabel("training step")
    else:
        # v1 CSV fallback.
        ax = axes[1]
        ax.plot(steps, cols["param_eff"], label="param-space efficiency", color="C0", lw=1.4)
        ax.plot(steps, cols["out_accord"], label="output-space accord (cum)", color="C1", lw=1.6)
        ax.set_ylabel("efficiency / accord")
        ax.grid(alpha=0.3)
        ax.legend(loc="center right", fontsize=9)

        ax = axes[2]
        ax.plot(steps, cols["out_gross_l1"], color="C4", lw=1.4, label="output GROSS L1")
        ax.plot(steps, cols["out_net_l2"],   color="C5", lw=1.4, label="output NET L2")
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=9)

    # Mark the grokking moment on every panel so movement curves align with it visually.
    if grok_step is not None:
        for a in axes:
            a.axvline(grok_step, color="k", ls="--", lw=1.0, alpha=0.6)
        axes[0].annotate(f"grok @ {int(grok_step)}",
                         xy=(grok_step, 0.95), xycoords=("data", "axes fraction"),
                         fontsize=9, ha="left", va="top",
                         xytext=(6, 0), textcoords="offset points")

    plt.tight_layout()
    out = ckpt / "dashboard.png"
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
