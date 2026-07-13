#!/usr/bin/env python3
"""Build the public GrokkingMetrics site → docs/index.html (GitHub Pages ready).

Sections:
  1. Overview          — what this is, headline results
  2. Ideal algorithm   — the interactive circle_algorithm.html panels, inlined
  3. The real model    — the trained v3 network running the algorithm live:
                         one-hot → embedding column → learned frequency planes
                         (real circle points vs ideal pegs) → readout sum point →
                         real logits vs ideal interference curve → argmax
  4. Ensemble results  — 10-seed mean ± σ bands (step- and grok-aligned)
  5. Instrumented run  — v3 deep dive (crystallization, emergence, spectra
                         handshake, leverage)
  6. Glossary

The real-model section is fully data-driven: the final v3 snapshot's embedding /
unembedding matrices, learned frequency-plane coordinates for every token, readout
circle coordinates for all 12,769 inputs, and quantized logits are baked into the
page as base64 — no server, no JS forward pass.

Usage: python3 tools/build_site.py   (run from repo root or tools/)
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from grokking_site import (  # noqa: E402
    CSS, GLOSSARY, build_ensemble, build_seed_state, build_seed_spectra,
    build_seed_leverage, run_curves,
)
from grokking_crystallization import (  # noqa: E402
    P, read_acts, read_snap_weights, embedding_freq_power,
)

ROOT = Path(__file__).parent.parent
V3 = ROOT / "checkpoints_nanda_grokking_v3"
ENS = ROOT / "ens_runs"
N_KEYS = 6


MODE_CSS = """
.modemenu{position:sticky;top:0;z-index:50;display:flex;gap:6px;flex-wrap:nowrap;
  overflow-x:auto;background:#141a26;padding:9px 12px}
.modebtn{font-size:13px;padding:7px 13px;border-radius:9px;border:1px solid #2c3547;
  background:#1c2536;color:#9fb0c9;cursor:pointer;font-weight:600;white-space:nowrap}
.modebtn:hover{background:#243049;color:#cdd8ea}
.modebtn.active{background:#3a5bd0;color:#fff;border-color:#3a5bd0}
.mode{display:none}
"""

ARITHMETIC_CARD = """
<div class="card"><h2>Modular addition, the way you'd do it by hand</h2>
<p class="sub">The whole task, before any network: (a + b) mod 113. It's clock arithmetic
on a 113-hour clock — add the two numbers, and if you pass 113, subtract one full turn.
That's it. Every later view (the circle algorithm, the toy transformer, the trained
network) is just a different machine computing this same answer. Pick a and b:</p>
<div class="ctl">a <input type="range" id="M0a" min="0" max="112" value="81"><span class="val" id="M0aV">81</span>
 &nbsp; b <input type="range" id="M0b" min="0" max="112" value="41"><span class="val" id="M0bV">41</span></div>
<div id="M0out" style="font-family:ui-monospace,Menlo,monospace;font-size:15px;
  line-height:2.1;margin-top:10px"></div>
</div>
<script>(function(){
  const $=id=>document.getElementById(id);
  function draw(){ const a=+$('M0a').value,b=+$('M0b').value,s=a+b,m=s%113;
    $('M0aV').textContent=a; $('M0bV').textContent=b;
    $('M0out').innerHTML = (s<113)
      ? `${a} + ${b} = <b>${s}</b> &nbsp; — under 113, so no wrap. &nbsp; (${a} + ${b}) mod 113 = <b style="color:#1b7f3b">${m}</b>.`
      : `${a} + ${b} = <b>${s}</b> &nbsp; — that's ≥ 113, so subtract one full turn:<br>` +
        `${s} − 113 = <b>${m}</b>. &nbsp; (${a} + ${b}) mod 113 = <b style="color:#1b7f3b">${m}</b>.`;
  }
  ['M0a','M0b'].forEach(id=>$(id).addEventListener('input',draw)); draw();
})();</script>
"""


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode()


def plane_coords(mat_td: np.ndarray, k: int, ideal_angles: np.ndarray):
    """Circle coordinates of each row of `mat_td` (rows = tokens/candidates/inputs,
    cols = dims) in the frequency-k plane found by DFT along the row axis of a
    generating matrix — here mat_td doubles as the generator (rows indexed by the
    cyclic variable). Returns (n, 2) coords aligned (rotation/reflection) to
    `ideal_angles`, normalized to mean radius 1."""
    X = mat_td - mat_td.mean(axis=0, keepdims=True)
    n = X.shape[0]
    ph = np.exp(-2j * np.pi * k * np.arange(n) / n)
    C = ph @ X                       # (dims,) complex DFT coefficient per dim
    u, v = np.real(C), -np.imag(C)
    x, y = X @ u, X @ v
    r = np.hypot(x, y).mean() + 1e-30
    x, y = x / r, y / r
    ang = np.arctan2(y, x)
    best = None
    for refl in (1, -1):
        a2 = np.arctan2(refl * y, x)
        rot = np.angle(np.exp(1j * (ideal_angles - a2)).mean())
        err = np.abs(np.angle(np.exp(1j * (a2 + rot - ideal_angles)))).mean()
        if best is None or err < best[0]:
            best = (err, refl, rot)
    _, refl, rot = best
    a2 = np.arctan2(refl * y, x) + rot
    rr = np.hypot(x, y)
    return np.stack([rr * np.cos(a2), rr * np.sin(a2)], axis=1).astype(np.float32)


def project_coords(mat_gen: np.ndarray, k: int, points: np.ndarray,
                   ideal_angles: np.ndarray):
    """Like plane_coords, but the plane comes from `mat_gen` (rows = cyclic var)
    while the projected `points` (n, dims) are arbitrary vectors (e.g. readouts)."""
    G = mat_gen - mat_gen.mean(axis=0, keepdims=True)
    ng = G.shape[0]
    ph = np.exp(-2j * np.pi * k * np.arange(ng) / ng)
    C = ph @ G
    u, v = np.real(C), -np.imag(C)
    Q = points - points.mean(axis=0, keepdims=True)
    x, y = Q @ u, Q @ v
    r = np.hypot(x, y).mean() + 1e-30
    x, y = x / r, y / r
    best = None
    for refl in (1, -1):
        a2 = np.arctan2(refl * y, x)
        rot = np.angle(np.exp(1j * (ideal_angles - a2)).mean())
        err = np.abs(np.angle(np.exp(1j * (a2 + rot - ideal_angles)))).mean()
        if best is None or err < best[0]:
            best = (err, refl, rot)
    _, refl, rot = best
    a2 = np.arctan2(refl * y, x) + rot
    rr = np.hypot(x, y)
    return np.stack([rr * np.cos(a2), rr * np.sin(a2)], axis=1).astype(np.float32), best[0]


def build_jlens(d: Path) -> str:
    """J-lens (Metric IV) card: dispersion through training, lens accuracy through
    depth, and the circle-check of per-candidate lens rows. Reads the dumps
    produced by ./nanda_jlens; returns "" gracefully if they are absent."""
    files = sorted(d.glob("jlens/jlens_[0-9]*.bin"),
                   key=lambda p: int(re.search(r"jlens_(\d+)", p.name).group(1)))
    if not files:
        return ""
    from jlens_analysis import read_jlens, boundary_circle, lens_logits, naive_logits
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = []
    for p in files:
        try:
            data.append(read_jlens(p))
        except Exception:
            pass
    steps = np.array([x["step"] for x in data])
    sizes = [int(x) for x in (d / "param_manifest.txt").read_text().split()]
    ks = [5, 8, 17, 49]

    ei = data[0]["eval_idx"]
    answers = ((ei // P) + (ei % P)) % P
    disp = [x["bounds"][1]["dis"] for x in data]
    accj, accn, errc = [], [], []
    for x in data:
        accj.append(float((lens_logits(x["bounds"][1]["lens"], x["eval_acts"][1])
                           .argmax(1) == answers).mean()))
        sp = d / "snaps" / f"snap_{int(x['step'])}.bin"
        if sp.exists():
            un = read_snap_weights(sp, sizes)[1]
            accn.append(float((naive_logits(un, x["eval_acts"][1], 1)
                               .argmax(1) == answers).mean()))
        else:
            accn.append(np.nan)
        errc.append(float(np.mean([boundary_circle(x["bounds"][1]["lens"], 1, k)[1]
                                   for k in ks])))
    accm = [float((x["eval_logits"].argmax(1) == answers).mean()) for x in data]
    _, _, grok, _ = run_curves(d)

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        specs=[[{"secondary_y": True}, {}, {}]],
                        subplot_titles=("lens dispersion (embed boundary)",
                                        "who knows the answer?",
                                        "circle error of lens rows"))
    fig.add_trace(go.Scatter(x=steps, y=disp, name="dispersion 1−‖E[L]‖²/E[‖L‖²]",
                             line=dict(color="#4c78a8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=accm, name="val acc (model, eval ctxs)",
                             line=dict(color="#bbb", dash="dot")),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=steps, y=accm, name="the model", showlegend=False,
                             line=dict(color="#222", width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=accj, name="mean J-lens @ embedding",
                             line=dict(color="#4c78a8")), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=accn, name="naive logit lens @ embedding",
                             line=dict(color="#f58518", dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=errc, name="mean angular err (rad, key ks)",
                             line=dict(color="#54a24b", width=2)), row=1, col=3)
    fig.add_hline(y=np.pi / 2, line_dash="dot", line_color="#bbb", row=1, col=3)
    for c in (1, 2, 3):
        fig.add_vline(x=grok, line_dash="dash", line_color="#d62728", row=1, col=c)
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[-0.02, 1.02], row=1, col=2)
    fig.update_yaxes(rangemode="tozero", row=1, col=3)
    fig.update_layout(height=380, width=1280, hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
                      margin=dict(t=40), plot_bgcolor="#f7f8fa")

    # final-snapshot circle of the =-slot lens rows, best key frequency
    final = data[-1]
    best_k = min(ks, key=lambda k: boundary_circle(final["bounds"][1]["lens"], 1, k)[1])
    coords, err, slot = boundary_circle(final["bounds"][1]["lens"], 1, best_k)
    ideal = 2 * np.pi * best_k * np.arange(P) / P
    figc = go.Figure()
    figc.add_scatter(x=np.cos(ideal), y=np.sin(ideal), mode="markers",
                     marker=dict(size=5, color="#ddd"), name="ideal pegs")
    figc.add_scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",
                     marker=dict(size=6, color=np.arange(P), colorscale="Twilight",
                                 showscale=False),
                     name=f"lens rows ({slot}, k={best_k}, err {err:.2f} rad)")
    figc.update_xaxes(scaleanchor="y", visible=False)
    figc.update_yaxes(visible=False)
    figc.update_layout(height=420, width=460, margin=dict(t=10, b=10),
                       legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
                       plot_bgcolor="#f7f8fa")

    return f"""
<div class='card' id='jlens'><h2>The Jacobian lens — asking an activation what it will become</h2>
<p class='sub'>Anthropic's J-lens (July 2026) asks, for an interior activation h at depth ℓ:
if everything downstream were replaced by its <i>average linear map</i>
L<sub>ℓ</sub> = E<sub>contexts</sub>[∂logits/∂h<sub>ℓ</sub>], what would the output be? We fit it
exactly here — one backward pass per logit per context, per training snapshot (TTTN's
<code>ActivationLens.hpp</code>; the familiar logit lens is the J = I special case). Because our
input space is enumerable, we can also measure what the big models can't: the
<b>dispersion</b> of per-context lenses around their mean — how much E[J] lies.
<b>Left:</b> at the embedding boundary the dispersion climbs and <i>peaks at the grok</i>, then
relaxes: per-context Jacobians are maximally diverse exactly at the transition.
<b>Middle:</b> the mean embedding-boundary lens never picks answers — the b-dependence of
∂logits/∂h<sub>a</sub> averages out (that missing energy <i>is</i> the dispersion) — while past
the transformer block the downstream map is linear and the lens equals the model by
construction. <b>Right:</b> yet the mean lens is not empty: its per-candidate rows land on the
circuit's key-frequency circles (angular error → ~0.1 rad through the grok) — the frozen
unembedding circle, transported back through the value path to the embedding boundary. The
lens learns the candidate <i>geometry</i> even while answer <i>selection</i> stays context-borne.</p>
{fig.to_html(include_plotlyjs=False, full_html=False)}
<p class='sub'>The =-slot rows of the final mean lens at the embedding boundary, projected
into their own frequency-{best_k} plane (grey: ideal pegs):</p>
{figc.to_html(include_plotlyjs=False, full_html=False)}
</div>"""


def build_jspace_lab(d: Path) -> str:
    """The intervention lab: the real trained block runs IN THE PAGE (JS forward
    from posemb-out, validated against the C++ dumps), so lens-row injection,
    embedding swaps, and J-space ablation happen live with sliders."""
    import struct

    snap = d / "snaps" / "snap_9999.bin"
    jl = d / "jlens" / "jlens_9999.bin"
    if not (snap.exists() and jl.exists()):
        return ""

    weights = np.fromfile(snap, dtype=np.float32)

    # boundary-2 mean lens + three golden (idx, logits) pairs for the in-page check
    buf = jl.read_bytes()
    off = 24
    lens2 = None
    for _ in range(4):
        bidx, tg, ac = struct.unpack_from("<QQQ", buf, off)
        off += 24
        block = np.frombuffer(buf, dtype=np.float32, count=tg * ac + tg + 2, offset=off)
        if bidx == 2:
            lens2 = block[:tg * ac].copy()
        off += 4 * (tg * ac + tg + 2)
    nctx = struct.unpack_from("<Q", buf, off)[0]
    off += 8
    fit_idx = np.frombuffer(buf, dtype=np.uint32, count=nctx, offset=off)
    off += 8 * nctx
    rec = 3 * 3 * 128 + 128 + P
    golds = []
    for i in (0, nctx // 2, nctx - 1):
        r = np.frombuffer(buf, dtype=np.float32, count=rec, offset=off + i * 4 * rec)
        golds.append((int(fit_idx[i]), r[3 * 3 * 128 + 128:]))
    gold_js = "[" + ",".join(
        f"{{idx:{gi},lg:dec32('{b64(gl.astype(np.float32))}')}}" for gi, gl in golds) + "]"

    return f"""
<div class='card' id='jspacelab'><h2>The intervention lab — steer the real model by hand</h2>
<p class='sub'>The trained network's final block runs <b>live in this page</b> (the forward
pass from the fully-embedded latent h₂ to the logits, real weights, verified against the
training dumps — see the check at the bottom). The lens L₂ = E<sub>contexts</sub>[∂logits/∂h₂]
was fit over 256 contexts; its 113 rows — one per candidate answer — are drawn below.
<b>Inject</b>: click a row c, slide α: h₂ ← h₂ + α·‖h₂‖·v̂<sub>c</sub>, and the model answers c
(~99% of contexts at α=1) even though a+b says otherwise. <b>Swap</b>: replace a's embedding
with a′ inside the latent — the answer moves to (a′+b) mod 113 every time: the circles are
causal. <b>Ablate</b>: greedily find the k lens atoms that best explain this h₂ (matching
pursuit) and delete its projection onto them. The 113 rows are massively redundant — they
live on circles inside the ~8-dimensional circuit subspace (the effective-rank result,
wearing a new hat) — so k≈16 removes the planes the winding lives on and the model is lost,
while removing 16 <i>random</i> directions (control button) does nothing.</p>
<div class='ctl'>a <input type='range' id='Ja' min='0' max='112' value='9'>
 <span class='val' id='JaV'>9</span>
 &nbsp; b <input type='range' id='Jb' min='0' max='112' value='38'>
 <span class='val' id='JbV'>38</span>
 &nbsp;&nbsp; <button class='jmode on' data-m='inject'>inject</button><button class='jmode'
 data-m='swap'>swap</button><button class='jmode' data-m='ablate'>ablate</button></div>
<div class='ctl' id='JctlInject'>target c: <b><span id='JcV'>87</span></b> (click a lens row)
 &nbsp; α <input type='range' id='Jalpha' min='0' max='200' value='100'>
 <span class='val' id='JalphaV'>1.00</span></div>
<div class='ctl' id='JctlSwap' style='display:none'>a′ <input type='range' id='Jap' min='0' max='112' value='61'>
 <span class='val' id='JapV'>61</span></div>
<div class='ctl' id='JctlAblate' style='display:none'>k <input type='range' id='Jk' min='0' max='16' value='8'>
 <span class='val' id='JkV'>8</span> atoms
 &nbsp; <label><input type='checkbox' id='Jrand'> remove k <i>random</i> directions instead (control)</label></div>
<div id='Jeq' style='font-family:ui-monospace,monospace;font-size:13px;margin:8px 0;padding:7px 10px;
 background:#f2f4f8;border-radius:6px'></div>
<h4>The lens matrix L₂ — 113 rows (candidate answers) × 384 dims (3 slots × 128)</h4>
<canvas id='Jlens' width='1140' height='226' style='cursor:pointer'></canvas>
<h4>Logits — grey: untouched model · colored: after your edit</h4>
<canvas id='Jlogits' width='1140' height='260'></canvas>
<div id='Jverdict' style='font-weight:640;margin:6px 0'></div>
<div class='sub' id='Jcheck'></div>
</div>
<style>.jmode{{border:1px solid #b9c3d4;background:#fff;padding:4px 14px;cursor:pointer;
font-size:13px}}.jmode.on{{background:#2b3f63;color:#fff;border-color:#2b3f63}}
.jmode:first-of-type{{border-radius:6px 0 0 6px}}.jmode:last-of-type{{border-radius:0 6px 6px 0}}</style>
<script>
(function(){{
function dec32(s){{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}}
const P=113,V=114,SEQ=3,E=128,H=4,HD=32,F=512,D2=SEQ*E;
const flat=dec32('{b64(weights)}');
const L2=dec32('{b64(lens2.astype(np.float32))}');
const GOLD={gold_js};
const sizes=[14592,128,384,128,128,16384,16384,16384,16384,128,128,65536,512,65536,128,14464,113];
const offs=[];{{let o=0;for(const s of sizes){{offs.push(o);o+=s;}}}}
const T=i=>flat.subarray(offs[i],offs[i]+sizes[i]);
const W={{embW:T(0),embB:T(1),pos:T(2),ln1g:T(3),ln1b:T(4),wq:T(5),wk:T(6),wv:T(7),wo:T(8),
ln2g:T(9),ln2b:T(10),w1:T(11),b1:T(12),w2:T(13),b2:T(14),wu:T(15),bu:T(16)}};

function ln(src,g,b){{const out=new Float64Array(D2);
  for(let s=0;s<SEQ;++s){{let m=0;for(let e=0;e<E;++e)m+=src[s*E+e];m/=E;
    let v=0;for(let e=0;e<E;++e){{const c=src[s*E+e]-m;v+=c*c;}}
    const is=1/Math.sqrt(v/E+1e-8);
    for(let e=0;e<E;++e)out[s*E+e]=(src[s*E+e]-m)*is*g[e]+b[e];}}
  return out;}}
function forwardFull(h2){{
  const h=Float64Array.from(h2);
  {{const u=ln(h,W.ln1g,W.ln1b);
    const q=new Float64Array(SEQ*H*HD),k=new Float64Array(SEQ*H*HD),v=new Float64Array(SEQ*H*HD);
    for(let s=0;s<SEQ;++s)for(let hh=0;hh<H;++hh)for(let d=0;d<HD;++d){{
      let sq=0,sk=0,sv=0;const wo=(hh*HD+d)*E;
      for(let e=0;e<E;++e){{const x=u[s*E+e];sq+=x*W.wq[wo+e];sk+=x*W.wk[wo+e];sv+=x*W.wv[wo+e];}}
      const o=(s*H+hh)*HD+d;q[o]=sq;k[o]=sk;v[o]=sv;}}
    const inv=1/Math.sqrt(HD),att=new Float64Array(H*SEQ*SEQ);
    for(let hh=0;hh<H;++hh)for(let qq=0;qq<SEQ;++qq){{
      const row=new Float64Array(SEQ);let mx=-1e30;
      for(let kk=0;kk<SEQ;++kk){{let sc=0;
        for(let d=0;d<HD;++d)sc+=q[(qq*H+hh)*HD+d]*k[(kk*H+hh)*HD+d];
        row[kk]=sc*inv;if(row[kk]>mx)mx=row[kk];}}
      let z=0;for(let kk=0;kk<SEQ;++kk){{row[kk]=Math.exp(row[kk]-mx);z+=row[kk];}}
      for(let kk=0;kk<SEQ;++kk)att[(hh*SEQ+qq)*SEQ+kk]=row[kk]/z;}}
    for(let s=0;s<SEQ;++s)for(let e=0;e<E;++e){{let sum=0;
      for(let hh=0;hh<H;++hh)for(let d=0;d<HD;++d){{let a=0;
        for(let kk=0;kk<SEQ;++kk)a+=att[(hh*SEQ+s)*SEQ+kk]*v[(kk*H+hh)*HD+d];
        sum+=a*W.wo[(e*H+hh)*HD+d];}}
      h[s*E+e]+=sum;}}}}
  const hA=Float64Array.from(h); // post-attention residual (attn-out boundary)
  {{const u=ln(h,W.ln2g,W.ln2b);
    for(let s=0;s<SEQ;++s){{const hid=new Float64Array(F);
      for(let f=0;f<F;++f){{let sum=W.b1[f];
        for(let e=0;e<E;++e)sum+=W.w1[f*E+e]*u[s*E+e];hid[f]=sum>0?sum:0;}}
      for(let e=0;e<E;++e){{let sum=W.b2[e];
        for(let f=0;f<F;++f)sum+=W.w2[e*F+f]*hid[f];h[s*E+e]+=sum;}}}}}}
  const lg=new Float64Array(P);
  for(let c=0;c<P;++c){{let sum=W.bu[c];
    for(let e=0;e<E;++e)sum+=W.wu[c*E+e]*h[2*E+e];lg[c]=sum;}}
  return {{hA,h3:h,lg}};}}
function forwardFromB2(h2){{return forwardFull(h2).lg;}}
function h2From(a,b){{const h=new Float64Array(D2);const tok=[a,b,P];
  for(let s=0;s<SEQ;++s)for(let e=0;e<E;++e)
    h[s*E+e]=W.embW[e*V+tok[s]]+W.embB[e]+W.pos[s*E+e];
  return h;}}
function h1From(a,b){{const h=new Float64Array(D2);const tok=[a,b,P];
  for(let s=0;s<SEQ;++s)for(let e=0;e<E;++e)
    h[s*E+e]=W.embW[e*V+tok[s]]+W.embB[e];
  return h;}}
// shared live model for downstream cards (Experiment 0 etc.)
window.GrokLive={{W,L2,forwardFull,forwardFromB2,h2From,h1From,P,V,SEQ,E,D2}};

// unit lens atoms
const atoms=new Float64Array(P*D2);
for(let c=0;c<P;++c){{let n=0;
  for(let i=0;i<D2;++i)n+=L2[c*D2+i]*L2[c*D2+i];n=Math.sqrt(n)+1e-30;
  for(let i=0;i<D2;++i)atoms[c*D2+i]=L2[c*D2+i]/n;}}

function mpPicks(h,kmax){{const r=Float64Array.from(h),picks=[];
  for(let k=0;k<kmax;++k){{let best=-1,bd=0;
    for(let c=0;c<P;++c){{let d=0;
      for(let i=0;i<D2;++i)d+=r[i]*atoms[c*D2+i];
      if(d>bd){{bd=d;best=c;}}}}
    if(best<0||bd<1e-6)break;
    picks.push(best);
    for(let i=0;i<D2;++i)r[i]-=bd*atoms[best*D2+i];}}
  return picks;}}
function removeSpan(h,dirs){{const basis=[];
  for(const dsrc of dirs){{const u=Float64Array.from(dsrc);
    for(const bv of basis){{let d=0;
      for(let i=0;i<D2;++i)d+=u[i]*bv[i];
      for(let i=0;i<D2;++i)u[i]-=d*bv[i];}}
    let n=0;for(let i=0;i<D2;++i)n+=u[i]*u[i];n=Math.sqrt(n);
    if(n<1e-4)continue;
    for(let i=0;i<D2;++i)u[i]/=n;basis.push(u);}}
  const out=Float64Array.from(h);
  for(const bv of basis){{let d=0;
    for(let i=0;i<D2;++i)d+=out[i]*bv[i];
    for(let i=0;i<D2;++i)out[i]-=d*bv[i];}}
  return out;}}

// deterministic random dirs for the ablate control
function randDirs(k){{let seed=1234;const rnd=()=>{{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff-0.5;}};
  const dirs=[];for(let j=0;j<k;++j){{const u=new Float64Array(D2);
    for(let i=0;i<D2;++i)u[i]=rnd();dirs.push(u);}}
  return dirs;}}

// ── state + UI ───────────────────────────────────────────────────────────────
const S={{a:9,b:38,mode:'inject',c:87,alpha:1.0,ap:61,k:8,rand:false}};
const $=id=>document.getElementById(id);

// lens heatmap (draw once to an offscreen image, redraw with row highlight)
const lc=$('Jlens'),lctx=lc.getContext('2d');
const cw=lc.width/D2, chh=lc.height/P;
let lensImg=null;
function drawLensBase(){{
  let mx=0;for(let i=0;i<P*D2;++i)mx=Math.max(mx,Math.abs(L2[i]));
  for(let c=0;c<P;++c)for(let i=0;i<D2;++i){{
    const v=L2[c*D2+i]/mx, t=Math.max(-1,Math.min(1,v*3));
    lctx.fillStyle=t>=0?`rgba(180,40,50,${{t}})`:`rgba(40,80,180,${{-t}})`;
    lctx.fillRect(i*cw,c*chh,Math.ceil(cw),Math.ceil(chh));}}
  lensImg=lctx.getImageData(0,0,lc.width,lc.height);}}
function drawLens(){{lctx.putImageData(lensImg,0,0);
  lctx.strokeStyle='#111';lctx.lineWidth=1.5;
  lctx.strokeRect(0,S.c*chh-1,lc.width,chh+2);
  lctx.fillStyle='#111';lctx.font='11px sans-serif';
  lctx.fillText('c='+S.c,4,Math.max(10,S.c*chh-3));
  for(let s=1;s<SEQ;++s){{lctx.strokeStyle='rgba(0,0,0,.25)';lctx.lineWidth=1;
    lctx.beginPath();lctx.moveTo(s*E*cw,0);lctx.lineTo(s*E*cw,lc.height);lctx.stroke();}}
  lctx.fillStyle='#555';lctx.font='10px sans-serif';
  lctx.fillText('a-slot',8,lc.height-4);lctx.fillText('b-slot',E*cw+8,lc.height-4);
  lctx.fillText('=-slot',2*E*cw+8,lc.height-4);}}
lc.addEventListener('click',ev=>{{const r=lc.getBoundingClientRect();
  S.c=Math.max(0,Math.min(P-1,Math.floor((ev.clientY-r.top)*(lc.height/r.height)/chh)));
  $('JcV').textContent=S.c;update();}});

function edited(){{
  const h=h2From(S.a,S.b);
  if(S.mode==='inject'){{let n=0;for(let i=0;i<D2;++i)n+=h[i]*h[i];n=Math.sqrt(n);
    const out=Float64Array.from(h);
    for(let i=0;i<D2;++i)out[i]+=S.alpha*n*atoms[S.c*D2+i];
    return {{h:out,eq:`h₂ ← h₂ + α·‖h₂‖·v̂_c    (α = ${{S.alpha.toFixed(2)}},  ‖h₂‖ = ${{n.toFixed(1)}},  c = ${{S.c}})`,
      target:S.c,tname:'your target c'}};}}
  if(S.mode==='swap'){{const out=Float64Array.from(h);
    for(let e=0;e<E;++e)out[e]+=W.embW[e*V+S.ap]-W.embW[e*V+S.a];
    return {{h:out,eq:`h₂[a-slot] ← h₂[a-slot] − E[${{S.a}}] + E[${{S.ap}}]    (predict: (${{S.ap}}+${{S.b}}) mod 113 = ${{(S.ap+S.b)%P}})`,
      target:(S.ap+S.b)%P,tname:'(a′+b) mod 113'}};}}
  // ablate
  const dirs=S.rand?randDirs(S.k)
    :mpPicks(h,S.k).map(c=>atoms.subarray(c*D2,(c+1)*D2));
  const out=removeSpan(h,dirs);
  return {{h:out,eq:S.rand
    ?`h₂ ← h₂ − proj onto ${{S.k}} RANDOM orthonormal directions (control)`
    :`h₂ ← h₂ − Σᵢ ⟨h₂,uᵢ⟩uᵢ    (uᵢ = Gram–Schmidt of h₂'s own top-${{S.k}} lens atoms)`,
    target:null,tname:''}};}}

const gc=$('Jlogits'),gctx=gc.getContext('2d');
function drawLogits(base,ed,target){{
  gctx.clearRect(0,0,gc.width,gc.height);
  const all=[...base,...ed];
  const mn=Math.min(...all),mx=Math.max(...all),pad=24;
  const y=v=>gc.height-14-(v-mn)/(mx-mn+1e-9)*(gc.height-30);
  const xw=(gc.width-pad)/P;
  const ans=(S.a+S.b)%P;
  let am=0,bm=0;
  for(let c=0;c<P;++c){{if(ed[c]>ed[am])am=c;if(base[c]>base[bm])bm=c;}}
  for(let c=0;c<P;++c){{const x=pad+c*xw;
    gctx.fillStyle='rgba(120,120,120,.35)';
    gctx.fillRect(x,y(base[c]),Math.max(1,xw-1.5),gc.height-14-y(base[c]));
    gctx.fillStyle=c===am?'#c33939':'#4c78a8';
    gctx.fillRect(x+xw*0.25,y(ed[c]),Math.max(1,xw*0.5),gc.height-14-y(ed[c]));}}
  const mark=(c,col,lab,dy)=>{{const x=pad+c*xw+xw/2;
    gctx.strokeStyle=col;gctx.lineWidth=1.4;gctx.setLineDash([4,3]);
    gctx.beginPath();gctx.moveTo(x,10);gctx.lineTo(x,gc.height-14);gctx.stroke();
    gctx.setLineDash([]);gctx.fillStyle=col;gctx.font='11px sans-serif';
    gctx.fillText(lab,Math.min(x+3,gc.width-90),dy);}};
  mark(ans,'#2c8a3d','(a+b) mod 113 = '+ans,20);
  if(target!==null&&target!==ans)mark(target,'#c33939','target = '+target,34);
  return {{am,bm}};}}

function update(){{
  ['Inject','Swap','Ablate'].forEach(m=>
    $('Jctl'+m).style.display=S.mode===m.toLowerCase()?'':'none');
  document.querySelectorAll('.jmode').forEach(bt=>
    bt.classList.toggle('on',bt.dataset.m===S.mode));
  const base=forwardFromB2(h2From(S.a,S.b));
  const e=edited();
  const lg=forwardFromB2(e.h);
  $('Jeq').textContent=e.eq;
  const {{am}}=drawLogits(base,lg,e.target);
  const ans=(S.a+S.b)%P;
  let msg=`model now says <b style='color:${{am===ans?'#2c8a3d':'#c33939'}}'>${{am}}</b>`;
  msg+=` &nbsp;(untouched answer: ${{ans}}`;
  if(e.target!==null)msg+=`, ${{e.tname}}: ${{e.target}} ${{am===e.target?'— <b>it moved</b> ✓':''}}`;
  msg+=')';
  $('Jverdict').innerHTML=msg;
  drawLens();}}

[['Ja','a',v=>{{S.a=v;$('JaV').textContent=v;}}],
 ['Jb','b',v=>{{S.b=v;$('JbV').textContent=v;}}],
 ['Jap','ap',v=>{{S.ap=v;$('JapV').textContent=v;}}],
 ['Jk','k',v=>{{S.k=v;$('JkV').textContent=v;}}]].forEach(([id,,fn])=>
  $(id).addEventListener('input',ev=>{{fn(+ev.target.value);update();}}));
$('Jalpha').addEventListener('input',ev=>{{S.alpha=+ev.target.value/100;
  $('JalphaV').textContent=S.alpha.toFixed(2);update();}});
$('Jrand').addEventListener('change',ev=>{{S.rand=ev.target.checked;update();}});
document.querySelectorAll('.jmode').forEach(bt=>
  bt.addEventListener('click',()=>{{S.mode=bt.dataset.m;update();}}));

// self-check: in-page forward vs C++ training dumps
{{let worst=0,ok=0;
  for(const g of GOLD){{const a=Math.floor(g.idx/P),b=g.idx%P;
    const lg=forwardFromB2(h2From(a,b));
    let am=0,gm=0;
    for(let c=0;c<P;++c){{worst=Math.max(worst,Math.abs(lg[c]-g.lg[c]));
      if(lg[c]>lg[am])am=c;if(g.lg[c]>g.lg[gm])gm=c;}}
    ok+=(am===gm)?1:0;}}
  $('Jcheck').textContent=`live-model check: page forward vs training dump on ${{GOLD.length}} contexts — max |Δlogit| = ${{worst.toExponential(1)}}, argmax ${{ok}}/${{GOLD.length}} ${{ok===GOLD.length?'✓':'✗ MISMATCH'}}`;}}

drawLensBase();update();
}})();
</script>"""


def build_exp0(d: Path) -> str:
    """Experiment 0 of the dispositional-abstraction program: same answer from
    different inputs — when do the DISPOSITIONS (L_l h) align while the STATES (h)
    stay distinct? Interactive pair picker over the live in-page model (shares
    window.GrokLive with the intervention lab) + population bands (same-sum vs
    different-sum pairs) + the training-collapse panel computed from every
    retained snapshot's lens dump."""
    import struct
    files = sorted(d.glob("jlens/jlens_[0-9]*.bin"),
                   key=lambda p: int(re.search(r"jlens_(\d+)", p.name).group(1)))
    if not files:
        return ""
    sys.path.insert(0, str(Path(__file__).parent))
    from jlens_analysis import read_jlens

    def read_split(p: Path):
        """jlens_split_<step>.bin from nanda_jlens_split: attn-out lens + acts."""
        buf = p.read_bytes()
        off = 24
        _, tg, ac = struct.unpack_from("<QQQ", buf, off)
        off += 24
        lens = np.frombuffer(buf, np.float32, tg * ac, off).reshape(tg, ac).astype(np.float64)
        off += 4 * (tg * ac + tg + 2)
        n = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        fit = np.frombuffer(buf, np.uint32, n, off)
        off += 8 * n  # fit + eval indices
        acts = np.frombuffer(buf, np.float32, 2 * n * tg * 0 + 2 * n * ac, off
                             ).reshape(2 * n, ac).astype(np.float64)
        return dict(lens=lens, fit=fit, acts=acts)

    rng = np.random.default_rng(7)

    def pair_sets(idx):
        sums = ((idx // P) + (idx % P)) % P
        by_sum = {}
        for i, s in enumerate(sums):
            by_sum.setdefault(int(s), []).append(i)
        same = []
        for g in by_sum.values():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    same.append((g[i], g[j]))
        same = [same[i] for i in rng.permutation(len(same))[:500]]
        diff = []
        n = len(idx)
        while len(diff) < 500:
            i, j = rng.integers(0, n, 2)
            if i != j and sums[i] != sums[j]:
                diff.append((int(i), int(j)))
        return same, diff

    def mean_cos(M, pairs):
        Mc = M - M.mean(axis=0, keepdims=True)
        n = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-30)
        return float(np.mean([float(n[i] @ n[j]) for i, j in pairs]))

    # ── training-collapse curves over all snapshots ──────────────────────────
    steps, curves = [], {k: [] for k in
                         ("D2s", "D2d", "D3s", "D3d", "S2s", "S2d", "S3s", "S4s",
                          "DAs", "DAd", "SAs")}
    final, final_split = None, None
    for p in files:
        try:
            dd = read_jlens(p)
        except Exception:
            continue
        idx = np.concatenate([dd["fit_idx"], dd["eval_idx"]])
        h2 = np.concatenate([dd["fit_acts"][2], dd["eval_acts"][2]])
        h3 = np.concatenate([dd["fit_acts"][3], dd["eval_acts"][3]])
        h4 = np.concatenate([dd["fit_acts"][4], dd["eval_acts"][4]])
        d2 = h2 @ dd["bounds"][2]["lens"].T
        d3 = h3 @ dd["bounds"][3]["lens"].T
        same, diff = pair_sets(idx)
        steps.append(int(dd["step"]))
        curves["D2s"].append(mean_cos(d2, same)); curves["D2d"].append(mean_cos(d2, diff))
        curves["D3s"].append(mean_cos(d3, same)); curves["D3d"].append(mean_cos(d3, diff))
        curves["S2s"].append(mean_cos(h2, same)); curves["S2d"].append(mean_cos(h2, diff))
        curves["S3s"].append(mean_cos(h3, same)); curves["S4s"].append(mean_cos(h4, same))
        sp = p.parent / f"jlens_split_{dd['step']}.bin"
        if sp.exists():
            ss = read_split(sp)
            assert np.array_equal(ss["fit"], dd["fit_idx"]), "split context sets diverge"
            da = ss["acts"] @ ss["lens"].T
            curves["DAs"].append(mean_cos(da, same)); curves["DAd"].append(mean_cos(da, diff))
            curves["SAs"].append(mean_cos(ss["acts"], same))
            if final_split is None or dd["step"] >= max(steps):
                final_split = ss
        else:
            for k in ("DAs", "DAd", "SAs"):
                curves[k].append(None)
        final = dd
    if final_split is None:
        print("experiment-0 SKIPPED: no jlens_split dumps — run ./nanda_jlens_split first")
        return ""
    _, _, grok, _ = run_curves(d)
    import json
    curves_js = json.dumps({"steps": steps,
                            **{k: [None if v is None else round(v, 4) for v in vv]
                               for k, vv in curves.items()}})

    # ── bake final-snapshot lenses + boundary means (for live centering) ─────
    L3 = final["bounds"][3]["lens"].astype(np.float32)
    L4 = final["bounds"][4]["lens"].astype(np.float32)
    LA = final_split["lens"].astype(np.float32)
    means = {}
    for b in (1, 2, 3):
        means[f"h{b}"] = np.concatenate([final["fit_acts"][b], final["eval_acts"][b]]
                                        ).mean(axis=0).astype(np.float32)
    means["hA"] = final_split["acts"].mean(axis=0).astype(np.float32)
    means["h4"] = np.concatenate([final["fit_acts"][4], final["eval_acts"][4]]
                                 ).mean(axis=0).astype(np.float32)
    means["lg"] = np.concatenate([final["fit_logits"], final["eval_logits"]]
                                 ).mean(axis=0).astype(np.float32)
    mean_js = ",".join(f"{k}:dec32('{b64(v)}')" for k, v in means.items())

    return f"""
<div class='card' id='exp0'><h2>Experiment 0 — dispositional abstraction: same answer, different inputs</h2>
<p class='sub'>From the <b>dispositional-abstraction program</b>: state-sameness is
cos(h, h′); disposition-sameness is cos(L·h, L·h′) — sameness <i>as the output sees it</i>,
state-sameness quotiented by ker L. Pick two inputs with the same sum (12+73 and 80+5 both
mean "85"): they are literally different prompts, so their latents differ — but somewhere in
depth their <b>dispositions</b> must snap together. Everything below runs live on the trained
model, with the transformer block split at its seam so the lens sees <b>attn-out</b>
(post-attention, pre-MLP) as its own boundary. <b>Predictions, registered before looking:</b>
(1) the snap lands at mlp-out — after attention the "="-slot finally <i>holds</i> both
operand angles (transport complete), but any linear read of it is still additive in
θ_a, θ_b; the sum-feature cos(θ_a+θ_b) is born in the MLP's products, so D at attn-out
should improve only modestly and snap only at mlp-out. If D partially snaps at attn-out
instead, that's evidence some multiplication already happens inside attention.
(Outcome: partial snap — attention deposits an arrow at the <i>half-angle</i> of the
answer whose signed gain 2cosΔ breathes and flips with |a−b|; see "the half-angle arrow"
panel in mode ② for the trig behind the wide attn-out band.)
(2) S(mlp-out) stays visibly below 1 while D(mlp-out) ≈ 1: <i>different latents, identical
disposition</i>. (3) At the readout the state itself merges (the readout ≈ the circle-point
for a+b) — the network has <i>forgotten the operands</i>, not just routed past them. (4) At
the embedding boundaries the mean lens is answer-blind (the spin-and-cancel result), so D
there is high for same-sum and different-sum pairs alike — which is why the <b>bands are
the measurement</b>: same-sum (blue) vs different-sum (grey) populations of 150 random
pairs each. Where the bands
separate is where disposition discriminates <i>meaning</i>. Cosines are centered against the
dataset mean (the "="-slot and positional embedding are shared constants that would inflate
everything); flip to raw to see the uncentered version. Centered curves at embed-out and
posemb-out are identical by construction — the positional embedding adds the same constant
to every prompt.</p>
<div class='ctl'>pair 1: a <input type='range' id='Xa' min='0' max='112' value='12'>
 <span class='val' id='XaV'>12</span>
 b <input type='range' id='Xb' min='0' max='112' value='73'>
 <span class='val' id='XbV'>73</span>
 &nbsp;&nbsp; pair 2: a′ <input type='range' id='Xa2' min='0' max='112' value='80'>
 <span class='val' id='Xa2V'>80</span>
 b′ <input type='range' id='Xb2' min='0' max='112' value='5'>
 <span class='val' id='Xb2V'>5</span>
 &nbsp; <button id='Xsame' class='jmode'>make same-sum</button>
 &nbsp; <label><input type='checkbox' id='Xraw'> raw (uncentered)</label></div>
<div id='Xverdict' style='font-weight:640;margin:6px 0'></div>
<canvas id='Xchart' width='1140' height='330'></canvas>
</div>
<div class='card' id='exp0train'><h2>Abstraction crystallizing — the dispositional collapse through training</h2>
<p class='sub'>The same measurement swept across every retained snapshot: for 512 contexts
per snapshot, all same-sum pairs (same meaning, different inputs) vs random different-sum
pairs, 500 pairs per class, centered cosines against that snapshot's own dataset mean —
each snapshot read through <b>its own</b> lens, so this is the instrument watching the
abstraction being built. Two things to watch: the blue same-sum disposition curve pulling
away from its grey control (disposition beginning to encode <i>meaning</i>), and where that
separation sits relative to the grok line — does the dispositional collapse lead, track,
or lag the behavioral jump? The purple curve is the readout's <i>state</i> within class:
when it rises, the network isn't just routing same-sum inputs to the same answer, it is
physically merging them — forgetting the operands.</p>
<canvas id='Xtrain' width='1140' height='300'></canvas>
</div>
<script>
(function(){{
function dec32(s){{const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}}
const G=window.GrokLive;
if(!G){{document.getElementById('Xverdict').textContent='live model unavailable';return;}}
const P=G.P,E=G.E,D2=G.D2,NB=6;
const L3=dec32('{b64(L3)}'),L4=dec32('{b64(L4)}'),LA=dec32('{b64(LA)}');
const MEAN={{{mean_js}}};
const TR={curves_js};
const GROK={int(grok)};

function applyLens(L,h,rows,dim){{const out=new Float64Array(rows);
  for(let c=0;c<rows;++c){{let s=0;
    for(let i=0;i<dim;++i)s+=L[c*dim+i]*h[i];out[c]=s;}}
  return out;}}
function cosv(x,y,mean,raw){{let dx,dy,dd=0,nx=0,ny=0;
  for(let i=0;i<x.length;++i){{dx=x[i]-(raw?0:mean[i]);dy=y[i]-(raw?0:mean[i]);
    dd+=dx*dy;nx+=dx*dx;ny+=dy*dy;}}
  return dd/(Math.sqrt(nx*ny)+1e-30);}}

const cache=new Map();
function feats(a,b){{const key=a*113+b;
  if(cache.has(key))return cache.get(key);
  const h1=G.h1From(a,b),h2=G.h2From(a,b);
  const r=G.forwardFull(h2);
  const h4=Float64Array.from(r.h3.subarray(2*E,3*E));
  const f={{h:[h1,h2,r.hA,r.h3,h4,r.lg],
    d:[applyLens(G.L2,h1,P,D2),applyLens(G.L2,h2,P,D2),applyLens(LA,r.hA,P,D2),
       applyLens(L3,r.h3,P,D2),applyLens(L4,h4,P,E),r.lg]}};
  cache.set(key,f);return f;}}
const HM=[MEAN.h1,MEAN.h2,MEAN.hA,MEAN.h3,MEAN.h4,MEAN.lg];
const DM=[applyLens(G.L2,MEAN.h1,P,D2),applyLens(G.L2,MEAN.h2,P,D2),
          applyLens(LA,MEAN.hA,P,D2),applyLens(L3,MEAN.h3,P,D2),
          applyLens(L4,MEAN.h4,P,E),MEAN.lg];
function curvesFor(a,b,a2,b2,raw){{
  const f1=feats(a,b),f2=feats(a2,b2);
  const S=[],D=[];
  for(let l=0;l<NB;++l){{S.push(cosv(f1.h[l],f2.h[l],HM[l],raw));
    D.push(cosv(f1.d[l],f2.d[l],DM[l],raw));}}
  return {{S,D}};}}

// population bands (computed lazily after first paint)
let bands=null;
function computeBands(){{const rnd=(()=>{{let s=99;
    return ()=>{{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;}}}})();
  const mk=(same)=>{{const Ds=[],Ss=[];
    for(let l=0;l<NB;++l){{Ds.push([]);Ss.push([]);}}
    for(let t=0;t<150;++t){{
      const a=Math.floor(rnd()*P),b=Math.floor(rnd()*P);
      let a2=Math.floor(rnd()*P),b2;
      if(same){{if(a2===a)a2=(a2+1)%P;b2=((a+b-a2)%P+P)%P;}}
      else{{b2=Math.floor(rnd()*P);
        if((a2+b2)%P===(a+b)%P)b2=(b2+1)%P;}}
      const c=curvesFor(a,b,a2,b2,S0.raw);
      for(let l=0;l<NB;++l){{Ds[l].push(c.D[l]);Ss[l].push(c.S[l]);}}}}
    const pct=(arr,q)=>{{const s=[...arr].sort((x,y)=>x-y);
      return s[Math.floor(q*(s.length-1))];}};
    return {{Dlo:Ds.map(a=>pct(a,.1)),Dhi:Ds.map(a=>pct(a,.9)),
             Smed:Ss.map(a=>pct(a,.5))}};}};
  bands={{same:mk(true),diff:mk(false)}};}}

const S0={{a:12,b:73,a2:80,b2:5,raw:false}};
const cv=document.getElementById('Xchart'),cx=cv.getContext('2d');
const LBL=['embed-out','posemb-out','attn-out','mlp-out','readout','logits'];
function draw(){{
  const c=curvesFor(S0.a,S0.b,S0.a2,S0.b2,S0.raw);
  cx.clearRect(0,0,cv.width,cv.height);
  const x0=60,x1=cv.width-20,y0=18,y1=cv.height-40;
  const ymin=-0.25,ymax=1.05;
  const X=l=>x0+(x1-x0)*l/(NB-1), Y=v=>y1-(Math.max(ymin,Math.min(ymax,v))-ymin)/(ymax-ymin)*(y1-y0);
  cx.strokeStyle='#ddd';cx.lineWidth=1;
  for(const g of [0,0.5,1]){{cx.beginPath();cx.moveTo(x0,Y(g));cx.lineTo(x1,Y(g));cx.stroke();
    cx.fillStyle='#999';cx.font='11px sans-serif';cx.fillText(g.toFixed(1),x0-28,Y(g)+4);}}
  cx.fillStyle='#555';
  for(let l=0;l<NB;++l)cx.fillText(LBL[l],X(l)-24,y1+16);
  const band=(lo,hi,col)=>{{cx.fillStyle=col;cx.beginPath();
    cx.moveTo(X(0),Y(lo[0]));
    for(let l=1;l<NB;++l)cx.lineTo(X(l),Y(lo[l]));
    for(let l=NB-1;l>=0;--l)cx.lineTo(X(l),Y(hi[l]));
    cx.closePath();cx.fill();}};
  if(bands){{band(bands.same.Dlo,bands.same.Dhi,'rgba(76,120,168,.18)');
    band(bands.diff.Dlo,bands.diff.Dhi,'rgba(110,110,110,.18)');
    const med=(m,col)=>{{cx.strokeStyle=col;cx.setLineDash([5,4]);cx.lineWidth=1.2;
      cx.beginPath();cx.moveTo(X(0),Y(m[0]));
      for(let l=1;l<NB;++l)cx.lineTo(X(l),Y(m[l]));cx.stroke();cx.setLineDash([]);}};
    med(bands.same.Smed,'rgba(220,130,50,.5)');med(bands.diff.Smed,'rgba(110,110,110,.5)');}}
  const line=(v,col,w)=>{{cx.strokeStyle=col;cx.lineWidth=w;cx.beginPath();
    cx.moveTo(X(0),Y(v[0]));
    for(let l=1;l<NB;++l)cx.lineTo(X(l),Y(v[l]));cx.stroke();
    for(let l=0;l<NB;++l){{cx.fillStyle=col;cx.beginPath();
      cx.arc(X(l),Y(v[l]),3.2,0,7);cx.fill();}}}};
  line(c.S,'#dc8232',2.4);line(c.D,'#2b5fa8',2.8);
  cx.font='12px sans-serif';
  cx.fillStyle='#2b5fa8';cx.fillText('D(ℓ) = cos(L·h, L·h′) — disposition (your pair)',x0+8,y0+12);
  cx.fillStyle='#dc8232';cx.fillText('S(ℓ) = cos(h, h′) — state (your pair)',x0+8,y0+28);
  if(bands){{cx.fillStyle='#777';
    cx.fillText('bands: D over 150 same-sum (blue) / 150 different-sum (grey) pairs · dashed: S medians',x0+8,y0+44);}}
  const s1=(S0.a+S0.b)%P,s2=(S0.a2+S0.b2)%P;
  const gap3=c.D[3]-c.S[3];
  document.getElementById('Xverdict').innerHTML=
    `${{S0.a}}+${{S0.b}} ≡ <b>${{s1}}</b> &nbsp;vs&nbsp; ${{S0.a2}}+${{S0.b2}} ≡ <b>${{s2}}</b>`+
    ` &nbsp;(${{s1===s2?'<span style="color:#2c8a3d">same meaning</span>':'<span style="color:#c33939">different meaning</span>'}})`+
    ` &nbsp;·&nbsp; attn-out: D = ${{c.D[2].toFixed(3)}}, S = ${{c.S[2].toFixed(3)}}`+
    ` &nbsp;·&nbsp; mlp-out: D = ${{c.D[3].toFixed(3)}}, S = ${{c.S[3].toFixed(3)}},`+
    ` <b>gap = ${{gap3.toFixed(3)}}</b>${{gap3>0.15?' — a distinction carried in the state that behavior has discarded':''}}`;
}}

// training panel
function drawTrain(){{const tv=document.getElementById('Xtrain'),tc=tv.getContext('2d');
  const x0=60,x1=tv.width-20,y0=16,y1=tv.height-36;
  const smin=TR.steps[0],smax=TR.steps[TR.steps.length-1];
  const ymin=-0.2,ymax=1.05;
  const X=s=>x0+(x1-x0)*(s-smin)/(smax-smin),
        Y=v=>y1-(Math.max(ymin,Math.min(ymax,v))-ymin)/(ymax-ymin)*(y1-y0);
  tc.strokeStyle='#ddd';
  for(const g of [0,0.5,1]){{tc.beginPath();tc.moveTo(x0,Y(g));tc.lineTo(x1,Y(g));tc.stroke();
    tc.fillStyle='#999';tc.font='11px sans-serif';tc.fillText(g.toFixed(1),x0-28,Y(g)+4);}}
  tc.strokeStyle='#d62728';tc.setLineDash([5,4]);
  tc.beginPath();tc.moveTo(X(GROK),y0);tc.lineTo(X(GROK),y1);tc.stroke();tc.setLineDash([]);
  tc.fillStyle='#d62728';tc.fillText('grok',X(GROK)+4,y0+10);
  const line=(key,col,w,dash)=>{{tc.strokeStyle=col;tc.lineWidth=w;
    if(dash)tc.setLineDash(dash);tc.beginPath();let pen=false;
    TR.steps.forEach((s,i)=>{{const v=TR[key][i];
      if(v===null){{pen=false;return;}}
      pen?tc.lineTo(X(s),Y(v)):tc.moveTo(X(s),Y(v));pen=true;}});
    tc.stroke();tc.setLineDash([]);}};
  line('D3s','#2b5fa8',2.8);line('D3d','#888',1.6);
  line('DAs','#1f9e89',2.4);line('DAd','#9ecac1',1.2,[3,3]);
  line('S3s','#dc8232',2.2);line('S4s','#8a5fbf',1.8,[6,4]);
  line('D2s','#2b5fa8',1.4,[3,3]);
  tc.font='12px sans-serif';let ly=y0+12;
  const leg=(t,c)=>{{tc.fillStyle=c;tc.fillText(t,x0+8,ly);ly+=15;}};
  leg('D(mlp-out) same-sum — disposition within meaning-class','#2b5fa8');
  leg('D(attn-out) same-sum — transport done, multiplication not yet','#1f9e89');
  leg('D(mlp-out) different-sum — control (attn-out control dotted teal)','#888');
  leg('S(mlp-out) same-sum — the state never merges here','#dc8232');
  leg('S(readout) same-sum — but the readout does (forgetting)','#8a5fbf');
  leg('dashed blue: D(posemb-out) same-sum — the blind boundary','#77a');
  tc.fillStyle='#555';
  for(const s of [0,2500,5000,7500,10000])if(s>=smin&&s<=smax)tc.fillText(s,X(s)-10,y1+16);
}}

const $=id=>document.getElementById(id);
[['Xa','a'],['Xb','b'],['Xa2','a2'],['Xb2','b2']].forEach(([id,k])=>
  $(id).addEventListener('input',ev=>{{S0[k]=+ev.target.value;
    $(id+'V').textContent=ev.target.value;draw();}}));
$('Xsame').addEventListener('click',()=>{{
  S0.b2=(((S0.a+S0.b-S0.a2)%P)+P)%P;
  $('Xb2').value=S0.b2;$('Xb2V').textContent=S0.b2;draw();}});
$('Xraw').addEventListener('change',ev=>{{S0.raw=ev.target.checked;
  bands=null;draw();
  setTimeout(()=>{{computeBands();draw();}},30);}});
draw();
setTimeout(()=>{{computeBands();draw();drawTrain();}},60);
}})();
</script>"""


def build_toylens() -> str:
    """The J-lens on the IDEAL machine (toy mode): attention = pure transport
    (chord midpoint), MLP = pure squaring, unembed = frozen rows. Everything is
    closed form, so the lens phenomena become theorems: the mean lens at
    attn-out is identically zero (E[J] of a squarer cancels), the per-context
    lens disposition is exactly 2x the model output at every boundary (Euler's
    homogeneous-function theorem), and the antipodal flips are uncensored.
    The card ends with the ideal-vs-real table: the distance is the finding."""
    return """
<div class='card' id='toylens'><h2>8 · The lens on the ideal machine — the theorem the real model deviates from</h2>
<p class='sub'>Same instrument as the real-model experiment, applied to the <b>ideal</b>
algorithm: embed on the circuit's four circles → attention = <b>pure transport</b> (the
chord midpoint of Panel 7) → MLP = <b>pure squaring</b> (angle doubling) → frozen rows →
interference. Because every stage is closed-form, the lens results here are not data —
they are <b>derivations</b>: <b>(i)</b> the <b>mean lens at attn-out is exactly zero</b>.
The sensitivity through a squarer is d(x²)/dx = 2x — proportional to the deposit itself —
and deposits point everywhere on the circle, so E[J] ≡ 0. A pure-transport machine gives
the mean lens <i>nothing</i> to read before the MLP: dispersion ≡ 1. <b>(ii)</b> the
<b>per-context lens disposition is exactly 2× the model's own logits at every
boundary</b> (the network is homogeneous of degree 2 in its state, so J·h = 2·F(h) —
Euler's theorem): the teal line below is perfectly flat. <b>(iii)</b> antipodal same-class
deposits are <b>uncensored</b>: the per-frequency plane cosines for (1,112) vs (56,57)
are exactly (−1)<sup>k</sup> — no product channel exists to drown them. Your clean mental
model is the theorem; the trained network is the experiment; <b>the lens measures the
distance between them</b>, and that distance has a name: value-composition — the real
attention pattern swings with content and leaks products, which is the only thing a mean
lens can see at attn-out.</p>
<div class='ctl'>pair 1: a <input type='range' id='Ta' min='0' max='112' value='12'>
 <span class='val' id='TaV'>12</span>
 b <input type='range' id='Tb' min='0' max='112' value='73'>
 <span class='val' id='TbV'>73</span>
 &nbsp;&nbsp; pair 2: a′ <input type='range' id='Ta2' min='0' max='112' value='80'>
 <span class='val' id='Ta2V'>80</span>
 b′ <input type='range' id='Tb2' min='0' max='112' value='5'>
 <span class='val' id='Tb2V'>5</span>
 &nbsp; <button id='Tsame' class='jmode'>make same-sum</button>
 &nbsp; <button id='Tanti' class='jmode'>antipodal exhibit (1,112)v(56,57)</button></div>
<div id='Tverdict' style='font-weight:640;margin:6px 0'></div>
<canvas id='Tchart' width='1140' height='330'></canvas>
<div id='Ttable' style='margin-top:10px;overflow-x:auto'></div>
</div>
<script>
(function(){
const P=113,TAU=Math.PI*2,KS=[5,8,17,49],NK=4,DIM=8;
const $=id=>document.getElementById(id);
const th=(k,x)=>TAU*k*x/P;

// ideal states per boundary: [embed(24), attn(24), mlp(24), readout(8), logits(113)]
function states(a,b){
  const aslot=new Float64Array(DIM),bslot=new Float64Array(DIM),
        m=new Float64Array(DIM),z=new Float64Array(DIM);
  for(let i=0;i<NK;++i){
    const k=KS[i];
    aslot[2*i]=Math.cos(th(k,a)); aslot[2*i+1]=Math.sin(th(k,a));
    bslot[2*i]=Math.cos(th(k,b)); bslot[2*i+1]=Math.sin(th(k,b));
    const mx=(aslot[2*i]+bslot[2*i])/2, my=(aslot[2*i+1]+bslot[2*i+1])/2;
    m[2*i]=mx; m[2*i+1]=my;
    z[2*i]=mx*mx-my*my; z[2*i+1]=2*mx*my;      // squaring: cos²Δ · e(2σ)
  }
  const cat=(x,y,w)=>{const o=new Float64Array(24);o.set(x,0);o.set(y,8);o.set(w,16);return o;};
  const zero=new Float64Array(DIM);
  const lg=new Float64Array(P);
  for(let c=0;c<P;++c){let s=0;
    for(let i=0;i<NK;++i){const k=KS[i];
      s+=z[2*i]*Math.cos(th(k,c))+z[2*i+1]*Math.sin(th(k,c));}
    lg[c]=s;}
  return {h:[cat(aslot,bslot,zero),cat(aslot,bslot,m),cat(aslot,bslot,z),z,lg],
          m,z,lg};
}
// per-context lens disposition at any boundary = 2·logits exactly (Euler, degree 2).
function cosv(x,y){let d=0,nx=0,ny=0;
  for(let i=0;i<x.length;++i){d+=x[i]*y[i];nx+=x[i]*x[i];ny+=y[i]*y[i];}
  return d/(Math.sqrt(nx*ny)+1e-30);}

// mean lens, computed honestly by averaging per-context Jacobians at attn-out
// over a large sample (the derivation says exactly 0; the number verifies it).
function meanLensAttnNorm(){
  // E[m] over the FULL grid: for each i, mean of the midpoint components — the
  // Jacobian of the squarer is linear in m, so E[J] reduces to J at E[m].
  const mx=new Float64Array(NK),my=new Float64Array(NK); let cnt=0;
  for(let a=0;a<P;++a)for(let b=0;b<P;++b){
    for(let i=0;i<NK;++i){const k=KS[i];
      mx[i]+=(Math.cos(th(k,a))+Math.cos(th(k,b)))/2;
      my[i]+=(Math.sin(th(k,a))+Math.sin(th(k,b)))/2;}
    ++cnt;}
  let s=0;
  for(let c=0;c<P;++c)for(let i=0;i<NK;++i){
    const k=KS[i],rc=Math.cos(th(k,c)),rs=Math.sin(th(k,c));
    const x=mx[i]/cnt,y=my[i]/cnt;
    const u=2*(rc*x+rs*y),v=2*(-rc*y+rs*x);
    s+=u*u+v*v;}
  return Math.sqrt(s);
}

const S0={a:12,b:73,a2:80,b2:5};
const LBL=['embed','attn-out (transport)','mlp-out (squared)','readout','logits'];
const cv=$('Tchart'),cx=cv.getContext('2d');
function draw(){
  const s1=states(S0.a,S0.b),s2=states(S0.a2,S0.b2);
  const S=[],Dm=[],Dp=[];
  for(let l=0;l<5;++l)S.push(cosv(s1.h[l],s2.h[l]));
  const dl=cosv(s1.lg,s2.lg);
  for(let l=0;l<5;++l){Dp.push(dl);Dm.push(l>=2?dl:null);} // mean lens ≡0 before mlp-out
  cx.clearRect(0,0,cv.width,cv.height);
  const x0=60,x1=cv.width-20,y0=18,y1=cv.height-40,ymin=-1.05,ymax=1.05;
  const X=l=>x0+(x1-x0)*l/4,Y=v=>y1-(Math.max(ymin,Math.min(ymax,v))-ymin)/(ymax-ymin)*(y1-y0);
  cx.strokeStyle='#ddd';
  for(const g of [-1,-0.5,0,0.5,1]){cx.beginPath();cx.moveTo(x0,Y(g));cx.lineTo(x1,Y(g));cx.stroke();
    cx.fillStyle='#999';cx.font='11px sans-serif';cx.fillText(g.toFixed(1),x0-32,Y(g)+4);}
  cx.fillStyle='#555';
  for(let l=0;l<5;++l)cx.fillText(LBL[l],X(l)-30,y1+16);
  // hatched "mean lens ≡ 0" zone over embed..attn
  cx.fillStyle='rgba(43,95,168,.06)';cx.fillRect(X(0),y0,X(1.5)-X(0),y1-y0);
  cx.fillStyle='#2b5fa8';cx.font='11px sans-serif';
  cx.fillText('mean lens ≡ 0 here (E[J] of a squarer cancels) — nothing to read',X(0)+6,y0+12);
  const line=(v,col,w,dash)=>{cx.strokeStyle=col;cx.lineWidth=w;
    if(dash)cx.setLineDash(dash);cx.beginPath();let pen=false;
    for(let l=0;l<5;++l){if(v[l]===null){pen=false;continue;}
      pen?cx.lineTo(X(l),Y(v[l])):cx.moveTo(X(l),Y(v[l]));pen=true;}
    cx.stroke();cx.setLineDash([]);
    for(let l=0;l<5;++l)if(v[l]!==null){cx.fillStyle=col;cx.beginPath();cx.arc(X(l),Y(v[l]),3.2,0,7);cx.fill();}};
  line(S,'#dc8232',2.4);
  line(Dp,'#1f9e89',2.8);
  line(Dm,'#2b5fa8',2.8,[6,4]);
  cx.font='12px sans-serif';
  cx.fillStyle='#dc8232';cx.fillText('S(ℓ) — state cos (exact)',x0+8,y1-46);
  cx.fillStyle='#1f9e89';cx.fillText('D per-context lens — flat ≡ cos(logits): J·h = 2F(h), Euler',x0+8,y1-30);
  cx.fillStyle='#2b5fa8';cx.fillText('D mean lens — defined only once downstream is linear',x0+8,y1-14);
  const sum1=(S0.a+S0.b)%P,sum2=(S0.a2+S0.b2)%P;
  $('Tverdict').innerHTML=
    `${S0.a}+${S0.b} ≡ <b>${sum1}</b> vs ${S0.a2}+${S0.b2} ≡ <b>${sum2}</b>`+
    ` (${sum1===sum2?'<span style="color:#2c8a3d">same meaning</span>':'<span style="color:#c33939">different meaning</span>'})`+
    ` · S(attn-out) = ${S[1].toFixed(3)} · cos(logits) = ${dl.toFixed(3)}`;
  // per-frequency table for the current pair at attn-out
  let rows='';
  for(let i=0;i<NK;++i){
    const k=KS[i];
    const p1=[s1.m[2*i],s1.m[2*i+1]],p2=[s2.m[2*i],s2.m[2*i+1]];
    const n1=Math.hypot(p1[0],p1[1]),n2=Math.hypot(p2[0],p2[1]);
    const pc=(p1[0]*p2[0]+p1[1]*p2[1])/(n1*n2+1e-30);
    rows+=`<tr><td>k=${k}</td><td>${n1.toFixed(3)}</td><td>${n2.toFixed(3)}</td>`+
      `<td style="color:${pc<0?'#c33939':'#2c8a3d'};font-weight:640">${pc.toFixed(3)}</td></tr>`;
  }
  $('Ttable').innerHTML=
    `<table style='border-collapse:collapse;font-size:13px'>`+
    `<tr style='background:#f2f4f8'><th style='padding:4px 14px'>plane</th>`+
    `<th style='padding:4px 14px'>|cosΔ| pair 1</th><th style='padding:4px 14px'>|cosΔ| pair 2</th>`+
    `<th style='padding:4px 14px'>midpoint plane-cos</th></tr>${rows}</table>`+
    `<p class='sub' style='margin-top:10px'><b>Ideal vs trained, at attn-out:</b> mean-lens
     dispersion <b>1.000 (theorem)</b> vs <b>0.482 (measured)</b> — the trained net's mean
     lens keeps ~52% of its Jacobian energy there, all of it attention's leaked products
     (<code>tools/dissect_deposit.py</code>: gate-free read of the antipodal pair −0.51,
     lens FFN-path +0.74). Same-class state cos here can reach −1 (uncensored flips);
     the trained net's =-slot states bottom out near −0.56 for the same reason, but its
     <i>disposition</i> floor stays at +0.71 — the product channel doesn't flip.</p>`;
}
const bind=(id,k)=>$(id).addEventListener('input',ev=>{S0[k]=+ev.target.value;
  $(id+'V').textContent=ev.target.value;draw();});
bind('Ta','a');bind('Tb','b');bind('Ta2','a2');bind('Tb2','b2');
$('Tsame').addEventListener('click',()=>{S0.b2=(((S0.a+S0.b-S0.a2)%P)+P)%P;
  $('Tb2').value=S0.b2;$('Tb2V').textContent=S0.b2;draw();});
$('Tanti').addEventListener('click',()=>{S0.a=1;S0.b=112;S0.a2=56;S0.b2=57;
  for(const [id,v] of [['Ta',1],['Tb',112],['Ta2',56],['Tb2',57]]){$(id).value=v;$(id+'V').textContent=v;}
  draw();});
// numeric verification of the theorem, shown once
setTimeout(()=>{
  const nrm=meanLensAttnNorm();
  $('Tverdict').innerHTML+=` · ‖mean lens @ attn-out‖ over all 12,769 contexts = ${nrm.toExponential(1)} ✓ (theorem: 0)`;
},80);
draw();
})();
</script>"""


def build_real_model() -> str:
    sizes = [int(x) for x in (V3 / "param_manifest.txt").read_text().split()]
    snaps = sorted(V3.glob("snaps/snap_*.bin"),
                   key=lambda p: int(re.search(r"snap_(\d+)", p.name).group(1)))
    emb, unemb = read_snap_weights(snaps[-1], sizes)      # (128,114), (113,128)
    astep, _, acts, logits = read_acts(sorted(
        V3.glob("snaps/acts_*.bin"),
        key=lambda p: int(re.search(r"acts_(\d+)", p.name).group(1)))[-1])

    fp = embedding_freq_power(emb)
    candidates = np.argsort(fp)[::-1][:2 * N_KEYS] + 1
    print(f"real-model bake: snapshot step {astep}, spectral candidates = {sorted(candidates.tolist())}")

    a_idx = np.arange(P * P) // P
    b_idx = np.arange(P * P) % P
    tok = np.arange(P)

    # Functional filter: keep only frequencies whose readout sum-point actually
    # lands on the ideal peg (mean angular error < 0.5 rad); spectral power alone
    # admits impostors the unembedding never uses.
    emb_tok = emb[:, :P].T                                 # (113, 128) rows=tokens
    coords_tok, coords_sum, keys = {}, {}, []
    for k in sorted(int(x) for x in candidates):
        cs, err = project_coords(unemb, k, acts, 2 * np.pi * k * (a_idx + b_idx) / P)
        status = "circuit" if err < 0.5 else "impostor (skipped)"
        print(f"  k={k}: readout angular error {err:.3f} rad — {status}")
        if err >= 0.5 or len(keys) >= N_KEYS:
            continue
        keys.append(k)
        coords_sum[k] = cs
        coords_tok[k] = plane_coords(emb_tok, k, 2 * np.pi * k * tok / P)
    keys = np.array(keys)

    lscale = float(np.abs(logits).max() / 32767.0)
    logits_i16 = np.round(logits / lscale).astype(np.int16)

    js_data = f"""
const P={P}, KEYS={list(int(k) for k in keys)}, LSCALE={lscale};
const EMB=dec32("{b64(emb.astype(np.float32))}");            // 128x114
const CT={{{",".join(f"{k}:dec32('{b64(coords_tok[int(k)])}')" for k in keys)}}}; // per k: 113x2
const CS={{{",".join(f"{k}:dec32('{b64(coords_sum[int(k)])}')" for k in keys)}}}; // per k: 12769x2
const LOG=dec16("{b64(logits_i16)}");                        // 12769x113 int16
"""

    html = """
<div class='card' id='real'><h2>The real model, running the algorithm</h2>
<p class='sub'>Everything below is the <b>trained network from the instrumented run</b>
(final snapshot) — no idealization. Pick a and b: the one-hot tokens light up, their
embedding columns appear, their positions on each <i>learned</i> frequency plane are
plotted against the ideal pegs (hollow = where the perfect algorithm would put them,
filled = where the trained embedding actually puts them), the readout's sum-point is
shown on the unembedding-side planes, and the real logits are drawn against the ideal
interference curve. The red argmax is the model's answer.</p>
<div class='ctl'>a <input type='range' id='Ra' min='0' max='112' value='9'>
 <span class='val' id='RaV'>9</span>
 &nbsp; b <input type='range' id='Rb' min='0' max='112' value='38'>
 <span class='val' id='RbV'>38</span>
 &nbsp; <span id='Rverdict' style='font-weight:640'></span></div>

<h4>1 · Tokens (one-hot) and embedding columns</h4>
<canvas id='Ronehot' width='1140' height='120'></canvas>

<h4>2 · The learned frequency planes — ideal pegs (hollow) vs trained points (filled)</h4>
<p class='sub'>Blue = a, green = b, red = the <b>readout's</b> position on the
unembedding-side plane — the model's computed a+b. Watch red land on the peg the ideal
algorithm predicts.</p>
<div id='Rcircles' style='display:flex;flex-wrap:wrap;gap:10px'></div>

<h4>3 · Scoring — real logits vs ideal interference</h4>
<canvas id='Rlogits' width='1140' height='300'></canvas>
<div class='legend' id='Rtop3'></div>
</div>
<script>
(function(){ // scoped: the inlined ideal-algorithm script owns the top-level names
function dec32(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function dec16(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Int16Array(u.buffer);}
""" + js_data + """
const TAU=Math.PI*2;
const circDivs={};
(function(){const host=document.getElementById('Rcircles');
 for(const k of KEYS){const w=document.createElement('div');
   w.innerHTML=`<div style='text-align:center;font-size:12px;color:#555'>k = ${k}</div>
   <canvas id='Rc${k}' width='178' height='178'></canvas>`;
   host.appendChild(w); circDivs[k]=document.getElementById('Rc'+k);}})();

function drawReal(){
  const a=+document.getElementById('Ra').value, b=+document.getElementById('Rb').value;
  document.getElementById('RaV').textContent=a;
  document.getElementById('RbV').textContent=b;
  const i=a*P+b, cstar=(a+b)%P;

  // one-hot + embedding columns
  const c1=document.getElementById('Ronehot').getContext('2d');
  c1.clearRect(0,0,1140,120); c1.font='11px sans-serif'; c1.fillStyle='#333';
  c1.fillText('one-hot a',2,12); c1.fillText('one-hot b',2,40);
  c1.fillText('embed col a',2,72); c1.fillText('embed col b',2,102);
  for(let t=0;t<114;t++){
    const x=80+t*9;
    c1.fillStyle=(t===a)?'#1f77b4':'#e3e7f0'; c1.fillRect(x,4,8,12);
    c1.fillStyle=(t===b)?'#2ca02c':'#e3e7f0'; c1.fillRect(x,32,8,12);
  }
  let mx=0; for(let d=0;d<128;d++){mx=Math.max(mx,Math.abs(EMB[d*114+a]),Math.abs(EMB[d*114+b]));}
  for(let d=0;d<128;d++){
    const xa=80+d*8;
    const va=EMB[d*114+a]/mx, vb=EMB[d*114+b]/mx;
    c1.fillStyle=va>0?`rgba(31,119,180,${Math.abs(va)})`:`rgba(214,39,40,${Math.abs(va)})`;
    c1.fillRect(xa,62,7,14);
    c1.fillStyle=vb>0?`rgba(44,160,44,${Math.abs(vb)})`:`rgba(214,39,40,${Math.abs(vb)})`;
    c1.fillRect(xa,92,7,14);
  }

  // circles
  for(const k of KEYS){
    const cv=circDivs[k].getContext('2d'), cx=89, cy=89, R=70;
    cv.clearRect(0,0,178,178);
    cv.strokeStyle='#dde3ee'; cv.beginPath(); cv.arc(cx,cy,R,0,TAU); cv.stroke();
    cv.fillStyle='#dbe2ef';
    for(let t=0;t<P;t++){const th=TAU*k*t/P;
      cv.beginPath(); cv.arc(cx+R*Math.cos(th),cy-R*Math.sin(th),1.3,0,TAU); cv.fill();}
    const ideal=(n,col)=>{const th=TAU*k*n/P;
      cv.strokeStyle=col; cv.lineWidth=1.6; cv.beginPath();
      cv.arc(cx+R*Math.cos(th),cy-R*Math.sin(th),6,0,TAU); cv.stroke();};
    ideal(a,'#1f77b4'); ideal(b,'#2ca02c'); ideal(a+b,'#d62728');
    const pt=(x,y,col)=>{cv.fillStyle=col; cv.beginPath();
      cv.arc(cx+R*x,cy-R*y,4,0,TAU); cv.fill();};
    pt(CT[k][a*2],CT[k][a*2+1],'#1f77b4');
    pt(CT[k][b*2],CT[k][b*2+1],'#2ca02c');
    pt(CS[k][i*2],CS[k][i*2+1],'#d62728');
  }

  // logits
  const c3=document.getElementById('Rlogits').getContext('2d');
  c3.clearRect(0,0,1140,300);
  const real=new Array(P), ideal=new Array(P);
  let rmn=1e9,rmx=-1e9;
  for(let c=0;c<P;c++){ real[c]=LOG[i*P+c]*LSCALE; rmn=Math.min(rmn,real[c]); rmx=Math.max(rmx,real[c]);
    let s=0; for(const k of KEYS) s+=Math.cos(TAU*k*(a+b-c)/P); ideal[c]=s; }
  const X=c=>30+1080*c/(P-1);
  const Yr=v=>270-235*(v-rmn)/(rmx-rmn||1);
  const Yi=v=>270-235*(v-(-KEYS.length))/(2*KEYS.length);
  c3.strokeStyle='#c9cfdd'; c3.setLineDash([4,4]); c3.lineWidth=1.3; c3.beginPath();
  ideal.forEach((v,c)=>{c?c3.lineTo(X(c),Yi(v)):c3.moveTo(X(c),Yi(v));}); c3.stroke();
  c3.setLineDash([]);
  c3.strokeStyle='#1f77b4'; c3.lineWidth=1.8; c3.beginPath();
  real.forEach((v,c)=>{c?c3.lineTo(X(c),Yr(v)):c3.moveTo(X(c),Yr(v));}); c3.stroke();
  let am=0; for(let c=1;c<P;c++) if(real[c]>real[am]) am=c;
  c3.fillStyle='#d62728'; c3.beginPath(); c3.arc(X(am),Yr(real[am]),6,0,TAU); c3.fill();
  c3.fillStyle='#111'; c3.font='12px sans-serif';
  c3.fillText(`model argmax = ${am}`, X(am)+9, Yr(real[am])-8);
  c3.fillStyle='#555';
  c3.fillText(`solid = real logits · dashed = ideal Σ cos over ks {${KEYS.join(', ')}}`, 30, 16);
  const order=[...real.keys()].sort((x,y)=>real[y]-real[x]).slice(0,3);
  document.getElementById('Rtop3').innerHTML =
    `top-3 logits: ${order.map(c=>`c=${c} (${real[c].toFixed(2)})`).join(' · ')}`;
  const ok = am===cstar;
  const V=document.getElementById('Rverdict');
  V.textContent = ok ? `✓ correct: (${a}+${b}) mod 113 = ${cstar}`
                     : `✗ model says ${am}, truth ${cstar}`;
  V.style.color = ok ? '#1b7f3b' : '#c22';
}
document.getElementById('Ra').addEventListener('input',drawReal);
document.getElementById('Rb').addEventListener('input',drawReal);
drawReal();
})();
</script>"""
    return html


_SONIFY_HTML = """
<div class='card' id='sonify'>
<h2>Listen to the model train — the embedding spectrum as sound</h2>
<p class='sub'>The embedding Fourier spectrum, per snapshot, played out loud. Every frequency
<b>k</b> becomes a partial at pitch <b>k × fundamental</b>, its loudness √(spectral share).
At init the power is smeared across all 56 frequencies — a dense, buzzing noise; as training
runs it collapses onto a handful of circuit frequencies and the noise resolves into a steady
harmony. Nothing is added by hand — the sound <i>is</i> the DFT of the embedding rows, all at
once, swept over training. Pick any of the 10 ensemble seeds (or the canonical run): each
finds its own circuit, so each groks to its own chord.</p>
<div class='ctl'>
  <button id='sonPlay'>▶ play</button>
  <button id='sonStop' disabled>⏹ stop</button>
  &nbsp; run <select id='sonSeed'>__OPTIONS__</select>
  &nbsp; length <select id='sonDur'><option value='12'>12 s</option><option value='20' selected>20 s</option><option value='32'>32 s</option></select>
  &nbsp; fundamental <select id='sonF0'><option value='55'>55 Hz · A1</option><option value='65.41' selected>65 Hz · C2</option><option value='43.65'>44 Hz · F1</option></select>
  &nbsp; <span id='sonCirc' style='color:#d62728;font-weight:600'></span>
  &nbsp; <span id='sonStep' style='color:#555;font-variant-numeric:tabular-nums'></span>
</div>
<canvas id='sonHeat' width='1180' height='300' style='width:100%;max-width:1180px;height:auto;margin-top:8px;border-radius:6px'></canvas>
<div class='legend'>heatmap = the selected run's embedding spectrum, k = 1…56 (rows, k=1 bottom)
× training step (columns); brighter = more power · red ticks = that run's circuit
frequencies · the white line is the playhead.</div>
<canvas id='sonCv' width='1180' height='150' style='width:100%;max-width:1180px;height:auto;margin-top:6px'></canvas>
<div class='legend'>bars = the 56 partials at the current instant · height = √share (what you
hear) · red = circuit frequencies. Watch the bands take root as you hear them lock in.</div>
<script>
(function(){
  const SEEDS=__SEEDS__, ORDER=__ORDER__, N=56;
  const heat=document.getElementById('sonHeat'), hcx=heat.getContext('2d');
  const cv=document.getElementById('sonCv'), cx=cv.getContext('2d');
  const stepEl=document.getElementById('sonStep'), circEl=document.getElementById('sonCirc');
  const bPlay=document.getElementById('sonPlay'), bStop=document.getElementById('sonStop');
  const selSeed=document.getElementById('sonSeed');
  const cache=document.createElement('canvas'); cache.width=heat.width; cache.height=heat.height;
  let ctx=null, oscs=[], gains=[], master=null, comp=null, raf=0, playing=false, t0=0;
  let key=ORDER[0], D=SEEDS[key], smax=1;
  const lerp=(a,b,t)=>a+(b-a)*t;
  function magma(t){ t=Math.max(0,Math.min(1,t));
    const cs=[[0,0,4],[81,18,124],[183,55,121],[252,137,97],[252,253,191]];
    const s=t*4, i=Math.min(3,Math.floor(s)), f=s-i, a=cs[i], b=cs[i+1];
    return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')'; }
  function frameAt(f){
    const n=D.spec.length, i=Math.max(0,Math.min(Math.floor(f), n-2)), t=f-i;
    const A=D.spec[i], B=D.spec[i+1], amp=new Array(N);
    for(let k=0;k<N;k++) amp[k]=Math.max(0, lerp(A[k],B[k],t));
    return {amp, step:Math.round(lerp(D.steps[i],D.steps[i+1],t))};
  }
  function buildHeat(){
    const W=heat.width, H=heat.height, n=D.spec.length, cw=W/n, rh=H/N, c=cache.getContext('2d');
    c.fillStyle='#000'; c.fillRect(0,0,W,H);
    for(let j=0;j<n;j++) for(let k=0;k<N;k++){
      c.fillStyle=magma(Math.sqrt(D.spec[j][k])/smax);
      c.fillRect(j*cw, H-(k+1)*rh, Math.ceil(cw)+0.5, Math.ceil(rh)+0.5); }
    c.fillStyle='#d62728'; D.circuit.forEach(k=>c.fillRect(W-5, H-k*rh, 5, Math.max(2,rh)));
    c.fillStyle='#cfd6e4'; c.font='11px sans-serif';
    c.fillText('k=56',4,13); c.fillText('k=1',4,H-4);
  }
  function paintHeat(frac){
    const W=heat.width, H=heat.height;
    hcx.clearRect(0,0,W,H); hcx.drawImage(cache,0,0);
    if(frac!=null){ const x=frac*W; hcx.strokeStyle='#fff'; hcx.lineWidth=1.6;
      hcx.beginPath(); hcx.moveTo(x,0); hcx.lineTo(x,H); hcx.stroke(); }
  }
  function drawBars(fr){
    const W=cv.width, H=cv.height, bw=W/N;
    cx.clearRect(0,0,W,H);
    for(let k=0;k<N;k++){
      const h=(Math.sqrt(fr.amp[k])/smax)*(H-22), on=D.circuit.indexOf(k+1)>=0;
      cx.globalAlpha= on?1:0.5; cx.fillStyle= on?'#d62728':'#c8a94a';
      cx.fillRect(k*bw+1, H-16-h, bw-1.6, h); }
    cx.globalAlpha=1; cx.fillStyle='#8a93a6'; cx.font='11px sans-serif';
    cx.fillText('k=1',2,H-3); cx.fillText('k=56',W-36,H-3);
  }
  function setSeed(k){
    key=k; D=SEEDS[k]; let m=0;
    for(const r of D.spec) for(const v of r) if(v>m) m=v;
    smax=Math.sqrt(m)||1;
    circEl.textContent='circuit: '+D.circuit.map(x=>'k='+x).join(', ');
    stepEl.textContent=''; buildHeat(); paintHeat(null); drawBars(frameAt(D.spec.length-1));
  }
  function stop(){
    if(!playing && !ctx) return; playing=false; cancelAnimationFrame(raf);
    if(ctx){ const now=ctx.currentTime; master.gain.cancelScheduledValues(now);
      master.gain.setTargetAtTime(0.0001, now, 0.05);
      setTimeout(()=>{ oscs.forEach(o=>{try{o.stop()}catch(e){}}); oscs=[];
        if(ctx){ctx.close(); ctx=null;} }, 220); }
    bPlay.disabled=false; bStop.disabled=true; paintHeat(null);
  }
  function play(){
    if(playing) return; playing=true; bPlay.disabled=true; bStop.disabled=false;
    ctx=new (window.AudioContext||window.webkitAudioContext)();
    const F0=parseFloat(document.getElementById('sonF0').value);
    const DUR=parseFloat(document.getElementById('sonDur').value);
    comp=ctx.createDynamicsCompressor(); comp.threshold.value=-14; comp.ratio.value=12;
    comp.connect(ctx.destination);
    master=ctx.createGain(); master.gain.value=0.0001; master.connect(comp);
    master.gain.setTargetAtTime(0.16, ctx.currentTime, 0.1);
    oscs=[]; gains=[];
    for(let k=0;k<N;k++){ const o=ctx.createOscillator(), g=ctx.createGain();
      o.type='sine'; o.frequency.value=F0*(k+1); g.gain.value=0.0001;
      o.connect(g); g.connect(master); o.start(); oscs.push(o); gains.push(g); }
    t0=ctx.currentTime; const n=D.spec.length;
    function apply(fr){ const now=ctx.currentTime;
      for(let k=0;k<N;k++) gains[k].gain.setTargetAtTime(Math.max(0.0001,Math.sqrt(fr.amp[k])), now, 0.03); }
    function loop(){
      if(!playing) return;
      const frac=(ctx.currentTime-t0)/DUR;
      if(frac>=1){ const fr=frameAt(n-1); apply(fr); paintHeat(1); drawBars(fr);
        stepEl.textContent='step '+fr.step+' — grokked'; setTimeout(stop, 700); return; }
      const fr=frameAt(frac*(n-1)); apply(fr); paintHeat(frac); drawBars(fr);
      stepEl.textContent='step '+fr.step;
      raf=requestAnimationFrame(loop);
    }
    loop();
  }
  bPlay.onclick=play; bStop.onclick=stop;
  selSeed.onchange=()=>{ if(playing) stop(); setTimeout(()=>setSeed(selSeed.value), playing?240:0); };
  setSeed(ORDER[0]);
})();
</script>
</div>"""


def build_spectrum_sonifier(runs) -> str:
    """Per-run embedding Fourier spectra → a seed-selectable heatmap + additive-synthesis
    sound. runs: list of (label, dir). Each frequency k is a partial (pitch = k x
    fundamental, amplitude sqrt(spectral share)); flat/buzzy at init, collapsing onto that
    run's circuit frequencies as it groks. Same data as the embedding-spectrum heatmap."""
    import json
    seeds, order = {}, []
    for label, d in runs:
        mpath = d / "param_manifest.txt"
        snaps = sorted(d.glob("snaps/snap_*.bin"),
                       key=lambda p: int(re.search(r"snap_(\d+)", p.name).group(1)))
        if not mpath.exists() or len(snaps) < 2:
            continue
        sizes = [int(x) for x in mpath.read_text().split()]
        steps, spec = [], []
        for sp in snaps:
            step = int(re.search(r"snap_(\d+)", sp.name).group(1))
            emb, _ = read_snap_weights(sp, sizes)
            pe = embedding_freq_power(emb); pe = pe / pe.sum()
            steps.append(step)
            spec.append([round(float(x), 4) for x in pe])
        circuit = sorted(int(i + 1) for i in np.argsort(np.array(spec[-1]))[-4:])
        seeds[label] = {"steps": steps, "spec": spec, "circuit": circuit}
        order.append(label)
    if not seeds:
        return ""
    options = "".join(
        "<option value='%s'%s>%s</option>" % (lbl, " selected" if i == 0 else "", lbl)
        for i, lbl in enumerate(order))
    html = (_SONIFY_HTML
            .replace("__SEEDS__", json.dumps(seeds))
            .replace("__ORDER__", json.dumps(order))
            .replace("__OPTIONS__", options))
    print("sonifier baked: %d runs — %s" % (
        len(order), "; ".join("%s→%s" % (l, seeds[l]["circuit"]) for l in order)))
    return html


def main() -> int:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # ── ideal-algorithm explainer, inlined and split into its six panels ──────
    exp = (ROOT / "circle_algorithm.html").read_text()
    exp_style = exp.split("<style>")[1].split("</style>")[0]
    exp_body = exp.split("</header>")[1].split("</body>")[0]
    # separate the shared panel script from the panel markup
    exp_markup, panel_script = exp_body.split("<script>", 1)
    panel_script = panel_script.rsplit("</script>", 1)[0]
    exp_inner = exp_markup.split('<div class="wrap">', 1)[1].rsplit("</div>", 1)[0]
    _panels = [c.strip() for c in re.split(r"<!-- ═+ PANEL [A-G] ═+ -->", exp_inner) if c.strip()]
    assert len(_panels) == 7, f"expected 7 circle panels, found {len(_panels)}"
    PAN1, PAN2, PAN3, PAN4, PAN5, PAN6, PAN7 = _panels

    seed_dirs = sorted((p for p in ENS.glob("seed_*") if (p / "metrics.csv").exists()),
                       key=lambda p: int(p.name.split("_")[1]))
    ens_html, groks = build_ensemble(seed_dirs)
    _, _, v3_grok, _ = run_curves(V3)

    sonify_runs = [("canonical (v3)", V3)] + [
        ("seed " + p.name.split("_")[1], p) for p in seed_dirs]
    sonify_html = build_spectrum_sonifier(sonify_runs)
    v3_html = (build_seed_state(V3) + build_seed_spectra(V3)
               + (build_seed_leverage(V3) if (V3 / "leverage_realized.bin").exists() else ""))

    real_html = build_real_model()

    jlens_html = build_jlens(V3)
    if jlens_html:
        print("j-lens section baked")

    jspace_html = build_jspace_lab(V3)
    if jspace_html:
        print("intervention lab baked")

    exp0_html = build_exp0(V3)
    if exp0_html:
        print("experiment-0 abstraction cards baked")

    # Rent-knob section — needs the wd_runs clean grid; skip gracefully without it.
    knob_html = ""
    if (ROOT / "wd_runs" / "clean_grid_summary.json").exists():
        from wd_knob import build_section as build_knob
        knob_html = build_knob()
        print("rent-knob section baked")

    gloss = "".join(f"<dt>{t}</dt><dd>{b}</dd>" for t, b in GLOSSARY)
    import plotly.offline as po

    # ── the three modes ──────────────────────────────────────────────────────
    mode_paper = ARITHMETIC_CARD + PAN1 + PAN2 + PAN5
    mode_toy = PAN6 + PAN7 + build_toylens()
    mode_real = f"""
<div class='card'><h2>What happened here</h2>
<p class='sub'>Modular addition, (a + b) mod 113, trained on 30% of all pairs with heavy
weight decay. The network first memorizes (train accuracy → 100%, validation near
chance), then — hundreds of steps later — abruptly generalizes: the grok. Underneath the
abrupt jump, every structural instrument shows smooth, early movement: the readout
representation collapses from ~45 effective dimensions to ~8, the embedding concentrates
onto a handful of Fourier frequencies, the embedding and unembedding matrices agree on
those frequencies <i>before</i> the accuracy moves, and functional leverage concentrates
onto a small circuit of parameters. The algorithm being formed is geometric: numbers
become points on circles, addition becomes rotation, and answers are read off by
interference.</p></div>
{real_html}
{jspace_html}
{exp0_html}
<div class='card' id='ens'><h2>Ensemble — 10 seeds, mean ± 1σ</h2>
<p class='sub'>Left: raw training step. Right: grok-aligned — each run shifted so its own
grok moment (peak validation-accuracy slope) sits at τ = 0. Alignment is per-run, which
is why the transition stays sharp despite grok steps spanning {min(groks)}–{max(groks)}.</p>
{ens_html}</div>
<div class='card' id='v3'><h2>The instrumented run</h2>
<p class='sub'>The canonical seed with the full instrument suite: crystallization and
emergence, the embedding/unembedding spectral handshake, and leverage — realized
per-parameter influence against the architecture's structural prior.</p>
{v3_html}</div>
{knob_html}
{sonify_html}
{jlens_html}
<p class='sub' style='margin-top:6px'>The two Fourier tools below are how the frequencies
were <i>found</i> in these trained matrices, and why several of them make the logits sharp
— the analysis lens for everything above.</p>
{PAN3}
{PAN4}
<div class='card' id='gloss'><h2>Glossary</h2><dl class='gloss'>{gloss}</dl></div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>GrokkingMetrics — a grokked transformer, instrumented and explained</title>
<style>{CSS}{exp_style}{MODE_CSS}</style>
<script>{po.get_plotlyjs()}</script></head><body>
<header><h1>GrokkingMetrics</h1>
<p>A one-layer transformer learns modular addition, grokks, and gets caught in the act.
Trained and instrumented from scratch in <a href='https://github.com/benmeyersUSC/TTNN'
style='color:#9fc0ff'>TTTN</a> (a header-only C++20 ML library). Three views, same task:
the algorithm on paper, an idealized toy transformer you tune by hand, and the real
trained network with its full instrument suite. Grok steps across 10 seeds:
{min(groks)}–{max(groks)} (canonical run: {int(v3_grok)}).</p>
</header>
<div class='modemenu'>
  <button class='modebtn' data-mode='paper'>① Modular Addition — on paper &amp; on a circle</button>
  <button class='modebtn' data-mode='toy'>② Neural Modular Addition — the idealized toy model</button>
  <button class='modebtn' data-mode='real'>③ TTTN-Grokked Modular Transformer — the trained torus</button>
</div>
<div class='wrap'>
<div class='mode' id='mode-paper'>{mode_paper}</div>
<div class='mode' id='mode-toy'>{mode_toy}</div>
<div class='mode' id='mode-real'>{mode_real}</div>

<p style='color:#888;font-size:12.5px;text-align:center'>Built from scratch:
<a href='https://github.com/benmeyersUSC/TTNN'>TTTN</a> ·
<a href='https://github.com/benmeyersUSC/GrokkingMetrics'>GrokkingMetrics</a></p>
</div>
<script>{panel_script}</script>
<script>
(function(){{
  const modes={{paper:document.getElementById('mode-paper'),
               toy:document.getElementById('mode-toy'),
               real:document.getElementById('mode-real')}};
  const btns=[...document.querySelectorAll('.modebtn')];
  function show(m){{ if(!modes[m]) m='paper';
    for(const k in modes) modes[k].style.display=(k===m)?'block':'none';
    btns.forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
    // hash prefixed so it can't collide with any in-page element id (e.g. #real)
    history.replaceState(null,'','#view-'+m);
    window.scrollTo(0,0);
    requestAnimationFrame(()=>{{
      modes[m].querySelectorAll('.plotly-graph-div').forEach(d=>{{
        try{{ if(window.Plotly) Plotly.Plots.resize(d); }}catch(e){{}} }});
      window.dispatchEvent(new Event('resize'));
      window.scrollTo(0,0);
    }});
  }}
  btns.forEach(b=>b.addEventListener('click',()=>show(b.dataset.mode)));
  const h=(location.hash||'').replace('#view-','');
  show(modes[h]?h:'paper');
}})();
</script>
</body></html>"""

    out = docs / "index.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
