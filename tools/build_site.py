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


def main() -> int:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)

    # ── ideal-algorithm explainer, inlined ───────────────────────────────────
    exp = (ROOT / "circle_algorithm.html").read_text()
    exp_style = exp.split("<style>")[1].split("</style>")[0]
    exp_body = exp.split("</header>")[1].split("</body>")[0]

    seed_dirs = sorted((p for p in ENS.glob("seed_*") if (p / "metrics.csv").exists()),
                       key=lambda p: int(p.name.split("_")[1]))
    ens_html, groks = build_ensemble(seed_dirs)
    _, _, v3_grok, _ = run_curves(V3)

    v3_html = (build_seed_state(V3) + build_seed_spectra(V3)
               + (build_seed_leverage(V3) if (V3 / "leverage_realized.bin").exists() else ""))

    real_html = build_real_model()

    # Rent-knob section — needs the wd_runs clean grid; skip gracefully without it.
    knob_html, knob_nav = "", ""
    if (ROOT / "wd_runs" / "clean_grid_summary.json").exists():
        from wd_knob import build_section as build_knob
        knob_html = build_knob()
        knob_nav = "<a href='#knob'>Rent knob</a>"
        print("rent-knob section baked")

    gloss = "".join(f"<dt>{t}</dt><dd>{b}</dd>" for t, b in GLOSSARY)
    import plotly.offline as po

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>GrokkingMetrics — a grokked transformer, instrumented and explained</title>
<style>{CSS}{exp_style}</style>
<script>{po.get_plotlyjs()}</script></head><body>
<header><h1>GrokkingMetrics</h1>
<p>A one-layer transformer learns modular addition, grokks, and gets caught in the act.
Trained and instrumented from scratch in <a href='https://github.com/benmeyersUSC/TTNN'
style='color:#9fc0ff'>TTTN</a> (a header-only C++20 ML library): movement metrics,
crystallization spectra, leverage geometry — and the circle algorithm the network
actually learns, taught interactively and then demonstrated live by the trained model.
Grok steps across 10 seeds: {min(groks)}–{max(groks)} (canonical run: {int(v3_grok)}).</p>
</header>
<nav><a href='#ideal'>The algorithm</a><a href='#real'>The real model</a>
<a href='#ens'>Ensemble</a>{knob_nav}<a href='#v3'>Instrumented run</a><a href='#gloss'>Glossary</a></nav>
<div class='wrap'>

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
interference. Sections below: learn the algorithm, watch the trained model run it,
then see the training dynamics that produced it.</p></div>

<div id='ideal'>{exp_body}</div>

{real_html}

<div class='card' id='ens'><h2>Ensemble — 10 seeds, mean ± 1σ</h2>
<p class='sub'>Left: raw training step. Right: grok-aligned — each run shifted so its own
grok moment (peak validation-accuracy slope) sits at τ = 0. Alignment is per-run, which
is why the transition stays sharp despite grok steps spanning {min(groks)}–{max(groks)}.</p>
{ens_html}</div>

{knob_html}

<div class='card' id='v3'><h2>The instrumented run</h2>
<p class='sub'>The canonical seed with the full instrument suite: crystallization and
emergence, the embedding/unembedding spectral handshake, and leverage — realized
per-parameter influence against the architecture's structural prior.</p>
{v3_html}</div>

<div class='card' id='gloss'><h2>Glossary</h2><dl class='gloss'>{gloss}</dl></div>

<p style='color:#888;font-size:12.5px;text-align:center'>Built from scratch:
<a href='https://github.com/benmeyersUSC/TTNN'>TTTN</a> ·
<a href='https://github.com/benmeyersUSC/GrokkingMetrics'>GrokkingMetrics</a></p>
</div></body></html>"""

    out = docs / "index.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
