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
    files = sorted(d.glob("jlens/jlens_*.bin"),
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
    _panels = [c.strip() for c in re.split(r"<!-- ═+ PANEL [A-F] ═+ -->", exp_inner) if c.strip()]
    assert len(_panels) == 6, f"expected 6 circle panels, found {len(_panels)}"
    PAN1, PAN2, PAN3, PAN4, PAN5, PAN6 = _panels

    seed_dirs = sorted((p for p in ENS.glob("seed_*") if (p / "metrics.csv").exists()),
                       key=lambda p: int(p.name.split("_")[1]))
    ens_html, groks = build_ensemble(seed_dirs)
    _, _, v3_grok, _ = run_curves(V3)

    sonify_runs = [("canonical (v3)", V3)] + [
        ("seed " + p.name.split("_")[1], p) for p in seed_dirs]
    v3_html = (build_seed_state(V3) + build_seed_spectra(V3) + build_spectrum_sonifier(sonify_runs)
               + (build_seed_leverage(V3) if (V3 / "leverage_realized.bin").exists() else ""))

    real_html = build_real_model()

    jlens_html = build_jlens(V3)
    if jlens_html:
        print("j-lens section baked")

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
    mode_toy = PAN6
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
<div class='card' id='ens'><h2>Ensemble — 10 seeds, mean ± 1σ</h2>
<p class='sub'>Left: raw training step. Right: grok-aligned — each run shifted so its own
grok moment (peak validation-accuracy slope) sits at τ = 0. Alignment is per-run, which
is why the transition stays sharp despite grok steps spanning {min(groks)}–{max(groks)}.</p>
{ens_html}</div>
<div class='card' id='gloss'><h2>Glossary</h2><dl class='gloss'>{gloss}</dl></div>
{knob_html}
<div class='card' id='v3'><h2>The instrumented run</h2>
<p class='sub'>The canonical seed with the full instrument suite: crystallization and
emergence, the embedding/unembedding spectral handshake, and leverage — realized
per-parameter influence against the architecture's structural prior.</p>
{v3_html}</div>
{jlens_html}
<p class='sub' style='margin-top:6px'>The two Fourier tools below are how the frequencies
were <i>found</i> in these trained matrices, and why several of them make the logits sharp
— the analysis lens for everything above.</p>
{PAN3}
{PAN4}"""

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
