#!/usr/bin/env python3
"""Weight-decay knob — rent λ vs the frequency circuit, over the clean grid.

Bakes the 3-init × 5-λ grid (wd_runs/clean_s{init}_wd{λ}) into a self-contained
HTML section: per init column, the embedding spectrum (56 bins, share-normalized)
at a chosen (λ, training step), the eviction ladder (final circuit membership per
λ), and the val-accuracy curve with step cursor + grok marker.

`build_section()` returns the fragment build_site.py inlines into docs/index.html;
running this file standalone writes wd_runs/knob.html for local poking.

Usage: python3 tools/wd_knob.py   (after the grid has been run)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from grokking_crystallization import read_snap_weights, embedding_freq_power  # noqa: E402

ROOT = Path(__file__).parent.parent
WD = ROOT / "wd_runs"
INITS = [101, 202, 303]
LAMS = ["0.25", "0.5", "1.0", "2.0", "4.0"]


def snap_steps(d: Path) -> list[int]:
    return sorted(int(re.search(r"snap_(\d+)", p.name).group(1)) for p in d.glob("snaps/snap_*.bin"))


def spectra(d: Path, steps: list[int], sizes: list[int]) -> list[list[float]]:
    out = []
    for st in steps:
        emb, _ = read_snap_weights(d / f"snaps/snap_{st}.bin", sizes)
        p = embedding_freq_power(emb)
        out.append([round(float(x), 5) for x in (p / p.sum())])
    return out


def val_curve(d: Path) -> list[list[float]]:
    rows = list(csv.DictReader(open(d / "metrics.csv")))
    return [[int(r["step"]), round(float(r["val_acc"]), 2)] for r in rows[::5]]


def bake() -> dict:
    sizes = [int(x) for x in (ROOT / "ens_runs/seed_1/param_manifest.txt").read_text().split()]
    summary = json.load(open(WD / "clean_grid_summary.json"))
    steps = snap_steps(WD / f"clean_s{INITS[0]}_wd{LAMS[0]}")
    data = {"inits": INITS, "lams": LAMS, "steps": steps,
            "spec": {}, "val": {}, "circ": {}, "grok": {}, "fval": {}}
    for init in INITS:
        data["spec"][init], data["val"][init] = {}, {}
        data["circ"][init], data["grok"][init], data["fval"][init] = {}, {}, {}
        for lam in LAMS:
            d = WD / f"clean_s{init}_wd{lam}"
            assert snap_steps(d) == steps, f"snapshot grid mismatch in {d}"
            data["spec"][init][lam] = spectra(d, steps, sizes)
            data["val"][init][lam] = val_curve(d)
            s = summary[f"{init}_{lam}"]
            data["circ"][init][lam] = s["ks"]
            data["grok"][init][lam] = s["grok"]
            data["fval"][init][lam] = s["val"]
    return data


def build_section() -> str:
    """The knob card + script, ready to inline in docs/index.html."""
    return SECTION.replace("__DATA__", json.dumps(bake(), separators=(",", ":")))


SECTION = r"""
<style>
.ktiles{display:flex;gap:10px;margin:8px 0 10px}
.ktile{background:#f4f6fb;border-radius:9px;padding:8px 14px;text-align:center;min-width:76px}
.ktile .n{font-size:22px;font-weight:650;color:#1c1e21}
.ktile .l{font-size:11px;color:#5a6472;margin-top:1px}
.kcols{display:flex;gap:16px;flex-wrap:wrap}
.kcol{flex:1;min-width:480px}
#ktip{position:fixed;pointer-events:none;background:#141a26;color:#fff;font-size:12px;
     padding:5px 9px;border-radius:7px;display:none;z-index:9;max-width:260px}
table.ktbl{border-collapse:collapse;font-size:12px}
table.ktbl td,table.ktbl th{border:1px solid #e3e7f0;padding:3px 8px;text-align:right;
     font-family:ui-monospace,Menlo,monospace}
table.ktbl th{background:#f4f6fb}
</style>
<div class="card" id="knob"><h2>The rent knob — weight decay chooses S</h2>
<p class="sub">A clean grid: 3 fixed inits × 5 weight decays (λ), identical data split,
everything else equal. Weight decay is rent — every parameter pays λ·‖θ‖² per step, so
each frequency circuit must earn its keep: more circles buy logit margin
(cross-entropy reward), but each costs weight norm (decay tax). Turn the rent knob and
watch each init's frequency <b>family</b> get evicted member by member; scrub training
steps to watch the spectrum crystallize out of noise. The <b>law</b> — S falls as λ
rises — holds for every init; <b>which</b> frequencies form the family is the init's own
lottery, and evictions run down a mostly-nested hierarchy (see init 303's ladder).
Heavier rent groks earlier, until it can't hold at all: at λ=4 the circuit forms and is
torn back down. Circuit membership is the functional test (readout sum-point lands on
the ideal peg) at the final snapshot.</p>
<div class="ctl">
 <b>rent λ</b> <input type="range" id="Kl" min="0" max="4" value="2" step="1">
 <span class="val" id="KlV">1.0</span>
 &nbsp;&nbsp; <b>training step</b> <input type="range" id="Ks" min="0" max="40" value="40" step="1">
 <span class="val" id="KsV">—</span>
 &nbsp;&nbsp;<span class="legend"><span class="dot" style="background:#1f77b4"></span>circuit frequency
 &nbsp;<span class="dot" style="background:#c1c9d8"></span>non-circuit
 &nbsp;<span class="dot" style="background:#d62728"></span>grok (50% val)</span></div>
<div class="kcols" id="Kcols"></div>
<details style="margin-top:8px"><summary style="font-size:12.5px;color:#5a6472;cursor:pointer">
table view (final circuits per init × λ)</summary><div id="Ktbl" style="overflow-x:auto"></div></details>
</div>
<div id="ktip"></div>
<script>
(function(){
"use strict";
const D=__DATA__;
const $k=id=>document.getElementById(id);
const tip=$k("ktip");
const COLS={spec:{}, lad:{}, curve:{}};

(function build(){
  const host=$k("Kcols");
  for(const init of D.inits){
    const c=document.createElement("div"); c.className="kcol";
    c.innerHTML=`<h4 style='margin:6px 0'>init ${init}</h4>
    <div class="ktiles">
      <div class="ktile"><div class="n" id="S_${init}">–</div><div class="l">circles S</div></div>
      <div class="ktile"><div class="n" id="V_${init}">–</div><div class="l">final val</div></div>
      <div class="ktile"><div class="n" id="G_${init}">–</div><div class="l">grok step</div></div>
    </div>
    <canvas id="sp_${init}" width="470" height="170"></canvas>
    <div class="legend">embedding spectral share by frequency k (1–56) at the chosen step</div>
    <canvas id="la_${init}" width="470" height="102"></canvas>
    <div class="legend">the eviction ladder — final circuit members at each rent</div>
    <canvas id="cv_${init}" width="470" height="80"></canvas>
    <div class="legend">validation accuracy over training · cursor = chosen step</div>`;
    host.appendChild(c);
    COLS.spec[init]=$k("sp_"+init); COLS.lad[init]=$k("la_"+init); COLS.curve[init]=$k("cv_"+init);
  }
  let t=`<table class="ktbl"><tr><th>init</th>${D.lams.map(l=>`<th>λ=${l}</th>`).join("")}</tr>`;
  for(const init of D.inits)
    t+=`<tr><td>${init}</td>${D.lams.map(l=>{const s=D.circ[init][l];
        return `<td>S=${s.length}: {${s.join(", ")}}</td>`;}).join("")}</tr>`;
  $k("Ktbl").innerHTML=t+"</table>";
})();

function drawAll(){
  const lam=D.lams[+$k("Kl").value], si=+$k("Ks").value, step=D.steps[si];
  $k("KlV").textContent=lam; $k("KsV").textContent="step "+step;
  for(const init of D.inits){
    const circ=new Set(D.circ[init][lam]);
    $k("S_"+init).textContent=circ.size;
    $k("V_"+init).textContent=D.fval[init][lam].toFixed(1)+"%";
    $k("G_"+init).textContent=D.grok[init][lam]??"—";
    const cv=COLS.spec[init], g=cv.getContext("2d");
    g.clearRect(0,0,470,170);
    const spec=D.spec[init][lam][si], mx=Math.max(...spec);
    g.strokeStyle="#e3e7f0"; g.beginPath(); g.moveTo(12,146); g.lineTo(458,146); g.stroke();
    for(let k=1;k<=56;k++){
      const x=12+(k-1)*7.96, h=Math.max(1.5,120*spec[k-1]/(mx||1));
      g.fillStyle=circ.has(k)?"#1f77b4":"#c1c9d8";
      g.beginPath(); g.roundRect(x,146-h,6,h,[2,2,0,0]); g.fill();
    }
    g.fillStyle="#333"; g.font="10.5px sans-serif"; g.textAlign="center";
    for(const k of circ){ const x=12+(k-1)*7.96+3;
      const h=Math.max(1.5,120*spec[k-1]/(mx||1));
      g.fillText(k, x, Math.min(140,166-h-26)); }
    g.textAlign="start"; g.fillStyle="#8a93a6";
    g.fillText("k=1",12,160); g.fillText("k=56",436,160);
    g.fillText("max share "+(100*mx).toFixed(1)+"%", 12, 14);
    const lg=COLS.lad[init].getContext("2d");
    lg.clearRect(0,0,470,102);
    D.lams.forEach((l,ri)=>{
      const y=14+ri*18;
      if(l===lam){ lg.fillStyle="#eef2fa"; lg.fillRect(0,y-8,470,17); }
      lg.fillStyle=l===lam?"#2b4a9b":"#8a93a6"; lg.font=(l===lam?"bold ":"")+"11px sans-serif";
      lg.fillText("λ="+l, 6, y+4);
      for(const k of D.circ[init][l]){
        lg.fillStyle=l===lam?"#1f77b4":"#aab8d6";
        lg.beginPath(); lg.arc(52+(k-1)*7.3, y, l===lam?4:3, 0, 6.284); lg.fill();
      }
    });
    const vg=COLS.curve[init].getContext("2d");
    vg.clearRect(0,0,470,80);
    const vc=D.val[init][lam];
    const X=s=>12+446*s/10000, Y=v=>70-58*v/100;
    vg.strokeStyle="#e3e7f0"; vg.beginPath(); vg.moveTo(12,Y(0)); vg.lineTo(458,Y(0)); vg.stroke();
    const grok=D.grok[init][lam];
    if(grok!=null){ vg.strokeStyle="#d62728"; vg.setLineDash([3,3]);
      vg.beginPath(); vg.moveTo(X(grok),6); vg.lineTo(X(grok),72); vg.stroke(); vg.setLineDash([]); }
    vg.strokeStyle="#8a93a6"; vg.setLineDash([2,3]);
    vg.beginPath(); vg.moveTo(X(step),6); vg.lineTo(X(step),72); vg.stroke(); vg.setLineDash([]);
    vg.strokeStyle="#1f77b4"; vg.lineWidth=1.8; vg.beginPath();
    vc.forEach((p,i)=>{ i?vg.lineTo(X(p[0]),Y(p[1])):vg.moveTo(X(p[0]),Y(p[1])); }); vg.stroke();
    vg.lineWidth=1;
  }
}
["Kl","Ks"].forEach(id=>$k(id).addEventListener("input",drawAll));
$k("Ks").max=D.steps.length-1; $k("Ks").value=D.steps.length-1;

for(const init of D.inits){
  COLS.spec[init].addEventListener("mousemove",e=>{
    const k=Math.round((e.offsetX-12)/7.96)+1;
    if(k<1||k>56){tip.style.display="none";return;}
    const lam=D.lams[+$k("Kl").value], si=+$k("Ks").value;
    const s=D.spec[init][lam][si][k-1], c=D.circ[init][lam].includes(k);
    tip.innerHTML=`init ${init} · k=${k} · share ${(100*s).toFixed(2)}%`+
      (c?" · <b>circuit member</b>":"");
    tip.style.display="block"; tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+10)+"px";
  });
  COLS.lad[init].addEventListener("mousemove",e=>{
    const ri=Math.round((e.offsetY-14)/18), k=Math.round((e.offsetX-52)/7.3)+1;
    if(ri<0||ri>4||k<1||k>56){tip.style.display="none";return;}
    const l=D.lams[ri], mem=D.circ[init][l].includes(k);
    tip.innerHTML=`init ${init} · λ=${l} · k=${k}: ${mem?"<b>in circuit</b>":"not in circuit"}`;
    tip.style.display="block"; tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+10)+"px";
  });
  for(const cv of [COLS.spec[init],COLS.lad[init]])
    cv.addEventListener("mouseleave",()=>tip.style.display="none");
}
drawAll();
})();
</script>
"""

STANDALONE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weight-decay knob — rent vs circles</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;background:#eef0f4;color:#1c1e21}
.wrap{max-width:1560px;margin:0 auto;padding:20px 16px 60px}
.card{background:#fff;border-radius:14px;box-shadow:0 1px 5px rgba(20,26,38,.09);
      padding:18px 20px;margin:18px 0}
h2{font-size:19px;margin:0 0 4px}
.sub{color:#5a6472;font-size:13px;margin:0 0 12px;line-height:1.5;max-width:1000px}
.ctl{font-size:13px;color:#333;display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0}
.ctl input[type=range]{width:260px}
.val{font-family:ui-monospace,Menlo,monospace;background:#eef2fa;border-radius:5px;
     padding:1px 8px;font-size:12.5px;color:#2b4a9b}
canvas{background:#f7f8fa;border-radius:10px;display:block;margin:6px 0}
.legend{font-size:12px;color:#555}
.dot{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:4px;vertical-align:middle}
</style></head><body><div class="wrap">
__SECTION__
</div></body></html>
"""


def main() -> int:
    section = build_section()
    out = WD / "knob.html"
    out.write_text(STANDALONE.replace("__SECTION__", section))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
