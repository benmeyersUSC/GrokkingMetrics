# GrokkingMetrics — project brief for Claude

Nanda-style grokking (mod-113 addition, 1-layer transformer) replicated and instrumented
in TTTN (Ben's header-only C++20 ML library, pulled as a dep into `deps/TTTN` via
`./fetch_deps.sh`; remote is `benmeyersUSC/TTNN.git`). Instrumentation lives in TTTN;
the experiment, analysis, and visualization live here.

## End goal

A public write-up + one **interactive dashboard** (static, GitHub Pages from `docs/`):
1. Training results (ensemble bands, per-seed deep dives, leverage, spectra handshake)
2. The idealized circle algorithm, taught interactively (circle_algorithm.html panels)
3. **The real model running the algorithm**: pick (a, b), watch the trained network do it —
   one-hot → embedding column → projection onto learned frequency planes (real circle
   points vs ideal pegs) → the sum point in the readout → dot products with unembedding
   rows → real logits vs ideal interference curve → argmax. Then link each algorithm
   feature to training snapshots showing its formation.
Ben iterates by using it; first passes don't need to be perfect.

## Key results already established (v3 run + 10-seed ensemble)

- Grok (peak val-slope) ~step 1800 (ensemble 1360–2490). All state metrics LEAD:
  effective rank collapse (45→~7-10 ≈ 2×#key-freqs) leads by ~500 steps; R²(Fourier
  basis) 0.13→0.91; discovered 8-group-means R² 0.22→0.81 (emergence without priors).
- v3 key frequencies: k = {5, 8, 17, 49}. Embedding/unembedding spectral handshake
  (⟨E,U⟩×56, 1 = chance) goes 1.0 → ~10×; top-4 ks identical in both matrices from
  step 2500, 7/8 overlap by 1250 — handshake precedes the grok.
- Leverage: Gini(realized |J_i|) 0.51→0.87; attn.Q suppressed vs prior, attn.V/O ~12×
  above (pattern frozen, value path carries the function). Structural potential is
  architecture-level → seed-independent (`structural_potential.bin` copyable).

## The four acts (Ben's math deep-dive — status)

1. **Embedding** — OWNED. Pegs/winding/primality, k↔p−k mirrors, dials (α = out-of-plane
   tilt, φ = in-plane orientation), rows-are-coordinates, u/v planes from FFT.
   **DFT** — OWNED. Freeze-and-stack vs spin-and-cancel; matched magnitude p·A/2;
   worked p=7 example incl. alias; peg-sum-zero by rotation invariance.
2. **Attention** — TO LEARN. How heads at "=" attend to the a/b slots; the role of
   positional embeddings in fixing the pattern; what the W_V/W_O images carry; why the
   pattern saturates (our leverage data shows Q suppressed); whether any multiplication
   already happens in attention (Nanda's value-composition notes).
3. **MLP** — TO LEARN (the big one). How ReLU synthesizes the products for
   cos(w_a+w_b) = cos·cos − sin·sin; neuron clusters per frequency; promised: a tiny
   hand-computed network (p=5, one frequency, every matrix explicit).
4. **Unembedding/scoring** — mostly owned (frozen candidate rows, dot = cos of angle
   difference, interference margin: one k correct, five confident). Remaining: phase
   alignment conditions across layers; why readout planes may differ from embedding
   planes; logit scale/temperature.

## J-lens workstream (2026-07-09/10 — BUILT, all four phases)

Anthropic's Jacobian lens (transformer-circuits.pub/2026/workspace) implemented in TTTN
and applied here + NeuralCompiler. `L_ℓ = E[∂logits/∂h_ℓ]` fit by backprop of one-hot
logit cotangents (`BackwardRange` returns the interior grad); logit lens = J=I special
case. Core: TTTN `src/ActivationLens.hpp` (FitActivationLens, ApplyLens, LensVector,
ActivationLensAccumulator with accord-style dispersion 1−‖E[L]‖²/E[‖L‖²]) +
`ForwardFrom<Batch,I>` on BlockSequence/TTN + `ForwardInterior`/`BackwardInterior` on
EncoderDecoderBlock; golden tests in TTTN `tests/activation_lens_test.cpp` (8/8).

**Results (v3, this repo — `nanda_jlens`, `tools/jlens_analysis.py` → jlens.html,
site card in build_site.py, `nanda_jspace`):**
- Structural identities: lens(embed-out) ≡ lens(posemb-out) (posemb backward = identity);
  boundaries 3/4 have LINEAR downstream → dispersion ≡ 0, lens ≡ model (golden anchor,
  checked every snapshot; readout lens ≡ W_U to 1.5e-8).
- **Dispersion at the embedding boundary PEAKS at the grok** (0.52→0.82 @ ~1760, relaxes
  to ~0.55): per-context Jacobians maximally diverse at the transition (not a collapse —
  a peak; refine hypothesis).
- Mean embedding-boundary lens is answer-blind (b-dependence of ∂logits/∂h_a averages
  out — that energy IS the dispersion), yet its per-candidate =-slot rows crystallize
  onto circuit-frequency circles (angular err 1.01→0.10 rad by step 2500): the frozen
  unembedding circle transported back through the value path.
- **J-space interventions @ posemb-out (nonlinear downstream)**: inject α·‖h‖·v̂_c moves
  the answer to arbitrary c 82%@α=.5, 98.8%@α=1 (random dir ≤0.6%); ablating h's own
  top-16 J-space atoms → 8.6% acc (16 random dirs: 100%); a-slot embedding swap → answer
  moves to (a′+b) mod p **100%** — circles are causal. MP@readout: top-1 atom = answer
  100%, varexp 0.879 @ k=10.
- **Intervention lab on the public site** (`build_jspace_lab` in tools/build_site.py):
  the trained block runs LIVE in the page — JS ForwardFrom(posemb-out) with real weights
  (snap_9999 + b2 mean lens baked as base64, ~1.5MB). Modes: inject (click lens row,
  α slider), swap (a′ slider), ablate (top-k MP atoms + Gram-Schmidt removal, random
  control checkbox). In-page self-check vs training dumps (6e-6). Validated headlessly
  via `osascript -l JavaScript` + DOM shim (no node on this machine) — extract script
  from built HTML, shim atob/document, drive S/update().
- NeuralCompiler (`jlens_compiler.cpp`, report jlens_report.md): dispersion falls
  monotonically with depth 0.20→0.08; decoder depth table shows the answer forming
  (dec_in_0 structural guesses → dec_in_3 ≈ model); golden anchor 7e-7. NOTE: NC's
  deps/TTTN is pinned to an OLD TTTN generation (BatchedForward-era); EncoderDecoder.hpp
  is identical in both trees — keep edits mirrored.

## Commands

```
./build.sh [--local]                       # fetch deps + compile runner & leverage tool
./nanda_grokking <dir> <seed> <jacEvery> <perParam01>
./nanda_leverage <dir> [KInits]
python3 tools/grokking_site.py ens_runs checkpoints_nanda_grokking_v3   # dashboard.html
python3 tools/build_site.py                # docs/index.html (the public site)
```

Data dirs (gitignored): `checkpoints_nanda_grokking_{run1,run2,v3}/`, `ens_runs/seed_1..10/`.
v3 is the fully-instrumented canonical run (snapshots every 250 steps: weights + full
dataset readout acts + logits + embedding). Snap weight files are raw float32 in
all_params order — offsets from `param_manifest.txt` (embed.W first, unembed.W second-to-last).

## Style

- Ben drives design; keep responses short and pointed in math discussions; he learns by
  building the picture himself — give precise corrections, not lectures.
- Dashboards: self-contained HTML, plotly inline or vanilla JS canvas, light card theme.
- TTTN rule (applies there, not here): no doc comments in .hpp; README-only docs.
