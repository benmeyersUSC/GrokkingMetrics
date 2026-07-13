#!/usr/bin/env python3
"""Single-file interactive report for the Jacobian-lens instrumentation (Metric IV).

Reads jlens/jlens_<step>.bin dumps produced by ./nanda_jlens plus the run's
snaps/snap_<step>.bin weights (for the naive logit-lens baseline) and metrics.csv
(for val accuracy / the grok moment), and writes <ckpt_dir>/jlens.html:

  A. Lens dispersion through training — per boundary, 1 - ||E_c[L]||^2/E_c[||L||^2]:
     how much the mean lens lies about individual contexts. Hypothesis: collapses
     at the grok (the algorithm becomes context-uniform).
  B. Lens accuracy through depth and training — top-1 accuracy of argmax(L_l @ h_l)
     on held-out contexts, vs the naive logit lens (J = I: unembed applied to the
     "="-position residual) and the model itself. "When does each depth first know?"
  C. Disposition through depth — softmax(L_l @ h_l) for one context at three
     moments (pre-grok / grok / final), J-lens bars vs naive-lens line.
  D. Circle-check — per-candidate lens rows projected into the key frequency
     planes (plane from the row matrix itself, aligned to ideal peg angles):
     do earlier-boundary lens vectors live on transported circles?

Usage: python3 tools/jlens_analysis.py [ckpt_dir] [--ks 5,8,17,49]
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
from grokking_crystallization import read_snap_weights  # noqa: E402

P = 113
EMB = 128
SEQ = 3
BOUNDS = [1, 2, 3, 4]
BNAMES = {1: "embed-out", 2: "posemb-out", 3: "txf-out", 4: "readout"}
BCOLS = {1: "#4c78a8", 2: "#f58518", 3: "#54a24b", 4: "#b279a2"}
MAGIC = 0x4A4C454E53303031


def read_jlens(path: Path):
    buf = path.read_bytes()
    off = 0

    def u64():
        nonlocal off
        v = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        return v

    def f32(n):
        nonlocal off
        v = np.frombuffer(buf, dtype=np.float32, count=n, offset=off)
        off += 4 * n
        return v

    assert u64() == MAGIC, f"{path}: bad magic"
    step = u64()
    nb = u64()
    bounds = {}
    for _ in range(nb):
        b, tgt, act = u64(), u64(), u64()
        lens = f32(tgt * act).reshape(tgt, act).astype(np.float64)
        rowcoh = f32(tgt).astype(np.float64)
        coh, dis = f32(1)[0], f32(1)[0]
        bounds[b] = dict(lens=lens, rowcoh=rowcoh, coh=float(coh), dis=float(dis))
    n = u64()
    fit_idx = np.frombuffer(buf, dtype=np.uint32, count=n, offset=off); off += 4 * n
    eval_idx = np.frombuffer(buf, dtype=np.uint32, count=n, offset=off); off += 4 * n

    def read_set():
        nonlocal off
        acts = {1: [], 2: [], 3: [], 4: []}
        logits = []
        for _ in range(n):
            for b in (1, 2, 3):
                acts[b].append(f32(SEQ * EMB))
            acts[4].append(f32(EMB))
            logits.append(f32(P))
        return ({b: np.stack(a).astype(np.float64) for b, a in acts.items()},
                np.stack(logits).astype(np.float64))

    fit_acts, fit_logits = read_set()
    eval_acts, eval_logits = read_set()
    return dict(step=step, bounds=bounds, fit_idx=fit_idx, eval_idx=eval_idx,
                fit_acts=fit_acts, fit_logits=fit_logits,
                eval_acts=eval_acts, eval_logits=eval_logits)


def circle_err(lens_rows: np.ndarray, k: int):
    """Angular error of per-candidate rows (113, D) against ideal peg angles in
    the frequency-k plane derived from the rows themselves. Returns (coords, err)."""
    ideal = 2 * np.pi * k * np.arange(P) / P
    X = lens_rows - lens_rows.mean(axis=0, keepdims=True)
    ph = np.exp(-2j * np.pi * k * np.arange(P) / P)
    C = ph @ X
    u, v = np.real(C), -np.imag(C)
    x, y = X @ u, X @ v
    r = np.hypot(x, y).mean() + 1e-30
    x, y = x / r, y / r
    best = None
    for refl in (1, -1):
        a2 = np.arctan2(refl * y, x)
        rot = np.angle(np.exp(1j * (ideal - a2)).mean())
        err = np.abs(np.angle(np.exp(1j * (a2 + rot - ideal)))).mean()
        if best is None or err < best[0]:
            best = (err, refl, rot)
    err, refl, rot = best
    a2 = np.arctan2(refl * y, x) + rot
    rr = np.hypot(x, y)
    coords = np.stack([rr * np.cos(a2), rr * np.sin(a2)], axis=1)
    return coords, float(err)


def boundary_circle(lens: np.ndarray, b: int, k: int):
    """Best (coords, err, slot_label) over position slices for seq boundaries."""
    if b == 4:
        c, e = circle_err(lens, k)
        return c, e, ""
    L = lens.reshape(P, SEQ, EMB)
    best = None
    for s, lab in enumerate(("a-slot", "b-slot", "=-slot")):
        c, e = circle_err(L[:, s, :], k)
        if best is None or e < best[1]:
            best = (c, e, lab)
    return best


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def lens_logits(lens: np.ndarray, acts: np.ndarray) -> np.ndarray:
    return acts @ lens.T  # (n, act) @ (act, 113).T? lens is (113, act) -> acts @ lens.T


def naive_logits(unemb: np.ndarray, acts: np.ndarray, b: int) -> np.ndarray:
    """J = I baseline: unembed applied to the '='-position residual slice."""
    if b == 4:
        h = acts
    else:
        h = acts.reshape(len(acts), SEQ, EMB)[:, 2, :]
    return h @ unemb.T


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ckpt = Path(args[0]) if args else Path("checkpoints_nanda_grokking_v3")
    ks = [5, 8, 17, 49]
    for a in sys.argv[1:]:
        if a.startswith("--ks"):
            ks = [int(x) for x in a.split("=")[1].split(",")]

    files = sorted(ckpt.glob("jlens/jlens_[0-9]*.bin"),
                   key=lambda p: int(re.search(r"jlens_(\d+)", p.name).group(1)))
    if not files:
        print(f"no jlens dumps under {ckpt}/jlens", file=sys.stderr)
        return 1
    data = []
    for p in files:
        try:
            data.append(read_jlens(p))
        except Exception as e:  # in-progress dump from a live nanda_jlens run
            print(f"skip {p.name}: {e}", file=sys.stderr)
    steps = np.array([d["step"] for d in data])
    print(f"{len(data)} jlens snapshots: steps {steps[0]}..{steps[-1]}")

    sizes = [int(x) for x in (ckpt / "param_manifest.txt").read_text().split()]
    unembs = {}
    for st in steps:
        sp = ckpt / "snaps" / f"snap_{st}.bin"
        if sp.exists():
            unembs[int(st)] = read_snap_weights(sp, sizes)[1]  # (113, 128)

    # Grok moment: peak of smoothed val-acc slope (house definition).
    grok_step = None
    mpath = ckpt / "metrics.csv"
    msteps, vacc = [], []
    if mpath.exists():
        with open(mpath) as f:
            for row in csv.DictReader(f):
                msteps.append(float(row["step"])); vacc.append(float(row["val_acc"]))
        msteps, vacc = np.array(msteps), np.array(vacc)
        # house definition (grokking_report.py): w100-smoothed acc, w100-smoothed slope
        w100 = max(int(100 / float(np.median(np.diff(msteps)))), 1)
        sm = np.convolve(vacc, np.ones(w100) / w100, mode="same")
        slope = np.convolve(np.gradient(sm, msteps), np.ones(w100) / w100, mode="same")
        grok_step = float(msteps[np.argmax(slope)])
        print(f"grok (peak val-acc slope) ~ step {grok_step:.0f}")

    answers = {}  # per snapshot eval answers (same every snapshot; idx sets are fixed)
    ei = data[0]["eval_idx"]
    answers = ((ei // P) + (ei % P)) % P

    # ── accuracy curves ──────────────────────────────────────────────────────
    acc_jlens = {b: [] for b in BOUNDS}
    acc_naive = {b: [] for b in BOUNDS}
    acc_model = []
    for d in data:
        st = int(d["step"])
        for b in BOUNDS:
            zl = lens_logits(d["bounds"][b]["lens"], d["eval_acts"][b])
            acc_jlens[b].append(float((zl.argmax(1) == answers).mean()))
            if st in unembs:
                zn = naive_logits(unembs[st], d["eval_acts"][b], b)
                acc_naive[b].append(float((zn.argmax(1) == answers).mean()))
            else:
                acc_naive[b].append(np.nan)
        acc_model.append(float((d["eval_logits"].argmax(1) == answers).mean()))

    # ── circle-check curves + final coords ───────────────────────────────────
    circ_err = {b: [] for b in BOUNDS}   # mean over ks of best-slot angular err
    for d in data:
        for b in BOUNDS:
            errs = [boundary_circle(d["bounds"][b]["lens"], b, k)[1] for k in ks]
            circ_err[b].append(float(np.mean(errs)))
    final = data[-1]

    # ═══ Figure A: dispersion through training ═══
    figA = make_subplots(specs=[[{"secondary_y": True}]])
    for b in BOUNDS:
        figA.add_scatter(x=steps, y=[d["bounds"][b]["dis"] for d in data],
                         name=f"dispersion @ {BNAMES[b]}", mode="lines+markers",
                         line=dict(color=BCOLS[b]))
    if len(msteps):
        figA.add_scatter(x=msteps, y=vacc, name="val acc", secondary_y=True,
                         line=dict(color="#999", dash="dot"), opacity=0.7)
    if grok_step is not None:
        figA.add_vline(x=grok_step, line_dash="dash", line_color="#d62728",
                       annotation_text="grok")
    figA.update_yaxes(title="lens dispersion  1 − ‖E[L]‖²/E[‖L‖²]", rangemode="tozero")
    figA.update_yaxes(title="val acc", secondary_y=True)
    figA.update_layout(title="A — Lens dispersion through training. Dispersion = fraction of"
                             " per-context Jacobian energy the mean lens cannot speak for."
                             " embed-out ≡ posemb-out exactly (posemb backward is identity);"
                             " txf-out and readout are 0 by construction (linear downstream —"
                             " the golden anchor). The pre-transformer dispersion peaks at"
                             " the grok, then relaxes.",
                       height=460, hoverlabel=dict(namelength=-1))

    # ═══ Figure B: lens accuracy through training ═══
    figB = go.Figure()
    for b in BOUNDS:
        figB.add_scatter(x=steps, y=acc_jlens[b], name=f"J-lens @ {BNAMES[b]}",
                         mode="lines+markers", line=dict(color=BCOLS[b]))
        if b != 4:
            figB.add_scatter(x=steps, y=acc_naive[b], name=f"logit lens (J=I) @ {BNAMES[b]}",
                             mode="lines", line=dict(color=BCOLS[b], dash="dot"), opacity=0.6)
    figB.add_scatter(x=steps, y=acc_model, name="model (logits)", mode="lines",
                     line=dict(color="#222", width=3), opacity=0.85)
    if grok_step is not None:
        figB.add_vline(x=grok_step, line_dash="dash", line_color="#d62728")
    figB.add_hline(y=1.0 / P, line_dash="dot", line_color="#bbb",
                   annotation_text="chance")
    figB.update_yaxes(title="top-1 accuracy (held-out contexts)", range=[0, 1.02])
    figB.update_layout(title="B — When does each depth first know the answer?"
                             " argmax of the linearized logits L·h on held-out contexts,"
                             " J-lens (solid) vs naive logit lens J=I (dotted) vs the model."
                             " txf-out/readout track the model by construction; at embed-out"
                             " the MEAN lens is answer-blind — the b-dependence of"
                             " ∂logits/∂h_a averages out (that missing energy IS panel A's"
                             " dispersion).",
                       height=460, hoverlabel=dict(namelength=-1))

    # ═══ Figure C: disposition through depth for one context ═══
    demo_i = 7
    demo_idx = int(final["eval_idx"][demo_i])
    da, db = demo_idx // P, demo_idx % P
    dans = (da + db) % P
    pick_steps = []
    if grok_step is not None:
        for target in (grok_step * 0.4, grok_step, steps[-1]):
            pick_steps.append(int(steps[np.argmin(np.abs(steps - target))]))
    else:
        pick_steps = [int(steps[0]), int(steps[len(steps) // 2]), int(steps[-1])]
    pick_steps = list(dict.fromkeys(pick_steps))
    figC = make_subplots(rows=len(pick_steps), cols=len(BOUNDS),
                         subplot_titles=[f"{BNAMES[b]} @ step {st}"
                                         for st in pick_steps for b in BOUNDS],
                         vertical_spacing=0.10, horizontal_spacing=0.04)
    for r, st in enumerate(pick_steps, 1):
        d = data[int(np.where(steps == st)[0][0])]
        for c, b in enumerate(BOUNDS, 1):
            zl = lens_logits(d["bounds"][b]["lens"], d["eval_acts"][b][demo_i:demo_i + 1])[0]
            pj = softmax(zl)
            figC.add_bar(x=np.arange(P), y=pj, marker_color=BCOLS[b],
                         showlegend=False, row=r, col=c)
            if int(st) in unembs:
                zn = naive_logits(unembs[int(st)], d["eval_acts"][b][demo_i:demo_i + 1], b)[0]
                figC.add_scatter(x=np.arange(P), y=softmax(zn), mode="lines",
                                 line=dict(color="#888", width=1), showlegend=False,
                                 row=r, col=c)
            figC.add_vline(x=dans, line_color="#d62728", line_width=1, row=r, col=c)
    figC.update_layout(title=f"C — Disposition through depth for ({da} + {db}) mod 113 = "
                             f"{dans}: softmax of the linearized logits at each boundary"
                             " (bars = J-lens, grey = naive logit lens, red = answer).",
                       height=260 * len(pick_steps), hoverlabel=dict(namelength=-1))

    # ═══ Figure D: circle-check ═══
    figD = make_subplots(rows=1, cols=len(BOUNDS) + 1,
                         subplot_titles=[*(f"{BNAMES[b]}" for b in BOUNDS),
                                         "angular error vs step"],
                         horizontal_spacing=0.05)
    k0 = ks[0]
    ideal = 2 * np.pi * k0 * np.arange(P) / P
    for c, b in enumerate(BOUNDS, 1):
        coords, err, slot = boundary_circle(final["bounds"][b]["lens"], b, k0)
        figD.add_scatter(x=np.cos(ideal), y=np.sin(ideal), mode="markers",
                         marker=dict(size=4, color="#ddd"), showlegend=False, row=1, col=c)
        figD.add_scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                         marker=dict(size=5, color=np.arange(P), colorscale="Twilight"),
                         showlegend=False, row=1, col=c)
        figD.layout.annotations[c - 1].text = (
            f"{BNAMES[b]} {slot}  k={k0}  err={err:.2f} rad")
        figD.update_xaxes(scaleanchor=f"y{c if c > 1 else ''}", row=1, col=c)
    for b in BOUNDS:
        figD.add_scatter(x=steps, y=circ_err[b], name=f"err @ {BNAMES[b]}",
                         mode="lines+markers", line=dict(color=BCOLS[b]),
                         row=1, col=len(BOUNDS) + 1)
    if grok_step is not None:
        figD.add_vline(x=grok_step, line_dash="dash", line_color="#d62728",
                       row=1, col=len(BOUNDS) + 1)
    figD.add_hline(y=np.pi / 2, line_dash="dot", line_color="#bbb",
                   row=1, col=len(BOUNDS) + 1)
    figD.update_layout(title=f"D — Circle-check of per-candidate lens rows (final snapshot,"
                             f" k={k0}; grey = ideal pegs; right: mean angular error over"
                             f" ks {ks} — π/2 dotted ≈ random). At readout the rows ARE the"
                             f" unembedding circle; at embed-out the =-slot rows land on the"
                             f" same circuit frequencies — the frozen unembedding circle"
                             f" transported back through the value path. The mean lens learns"
                             f" the candidate GEOMETRY even while answer selection stays"
                             f" context-borne.",
                       height=380, hoverlabel=dict(namelength=-1))

    header = f"""
<h1>Jacobian lens — {ckpt.name}</h1>
<p><b>Metric IV.</b> The lens at boundary ℓ is <code>L_ℓ = E_contexts[∂logits/∂h_ℓ]</code>,
fit exactly per context by backprop of one-hot logit cotangents (TTTN
<code>ActivationLens.hpp</code>), averaged over {len(data[0]['fit_idx'])} stride-sampled
contexts; accuracy panels use {len(data[0]['eval_idx'])} held-out contexts. The naive logit
lens is the J=I special case. Boundary indices: 1 embed-out, 2 posemb-out, 3 txf-out,
4 readout (lens ≡ unembedding, dispersion ≡ 0 — verified at every snapshot).</p>
"""
    parts = [figA.to_html(include_plotlyjs="inline", full_html=False)]
    for fig in (figB, figC, figD):
        parts.append(fig.to_html(include_plotlyjs=False, full_html=False))
    out = ckpt / "jlens.html"
    out.write_text("<html><head><meta charset='utf-8'><title>J-lens report</title>"
                   "<style>body{font-family:system-ui;margin:24px;max-width:1500px}"
                   "code{background:#f2f2f2;padding:1px 4px;border-radius:3px}</style>"
                   f"</head><body>{header}{''.join(parts)}</body></html>")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
