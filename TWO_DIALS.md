# The Two-Dial Parable

*A chronicle of the smallest possible circle-algorithm transformer, the random phases
that broke it, and what orthogonality is actually worth. (2026-07-05, from a working
session; all numbers are real.)*

## The setup

Ben asked for the minimal machine: mod-113 addition, one frequency, k = 1, exactly two
embedding dimensions, both gains α = 1 — and, offhandedly, "take these two random
phases: φ₁ = 3.4, φ₂ = 4.5." Assume the combine stage (attention + MLP) is perfect, so
the readout is exactly the column the correct answer would have. Embed → add angles →
dot against 113 frozen candidate rows → argmax.

Every piece was specified correctly. The machine is five lines of arithmetic.

## The walk

Input a = 81, b = 41; truth: (81+41) mod 113 = 9.

- Dials: each dimension d stores `cos(θ_x − φ_d)`, the point's shadow on an axis at
  orientation φ_d. Stacked per-dial: u = (cos φ₁, cos φ₂) = (−0.9668, −0.2108),
  v = (sin φ₁, sin φ₂) = (−0.2555, −0.9775).
- Embed: col_81 = (+0.4501, +1.0000), col_41 = (+0.4354, −0.6048). (Column identity:
  col_x = cos θ_x·u + sin θ_x·v — the column is the point's coordinates re-expressed
  through the dials. Verified numerically.)
- Combine: θ_81 + θ_41 = 6.7836 → wraps → 0.5004 = θ_9 exactly. Readout =
  (−0.9709, −0.6540), the shadow pair token 9 would have. No angle, no arctan, no "red
  dot" is ever materialized — two numbers are the entire state.
- Decode: logit(c) = readout · row_c over all 113 candidate rows.

**Result: argmax = 12. logit(12) = 1.3947 > logit(9) = 1.3702.** The machine answers
81 + 41 ≡ 12. Top five: 12, 13, 11, 14, 10 — the truth doesn't even win locally.

## The diagnosis

Expand one logit with the product-to-sum identity and sum the two dimensions:

```
logit(c) = cos(θ_s − θ_c)  +  cos(φ₁ − φ₂) · cos(θ_s + θ_c − φ₁ − φ₂)
           └── the score ──┘   └──────────── the junk ────────────────┘
```

The junk's amplitude is cos(3.4 − 4.5) = cos(−1.1) = **0.4536 — which is exactly u·v**,
the shared direction of the two basis axes. We wanted cos(π/2) = 0 there; we paid for
what we specified instead. A second cosine wave, nearly half the height of the signal,
rides over the interference curve and shoves the peak from 9 to 12.

Geometrically: col_x = cos θ·u + sin θ·v traces a **circle only if u ⊥ v with equal
norms; otherwise it traces an ellipse** — and dot products against an ellipse leak.

The subtle inversion that made the trap invisible: **two non-parallel dials always
suffice to *encode* the point** (a nonlinear reader can drop perpendiculars and
intersect them — it can undo any oblique basis). But the model's *decoder* is a single
linear map: the only thing it can do is dot products, and dot-product decoding
additionally demands an orthonormal basis. Encoding tolerates obliqueness; linear
decoding taxes it, at a rate of exactly u·v.

## The fix (and the rerun)

Set φ = (0, π/2). Then u = (1, 0), v = (0, 1): the dossiers *are* the ideal circle
points, col_81 = (−0.2070, −0.9783) = the point itself. Rerun the identical walk:

**argmax = 9, logit(9) = 1.000000, and logits(c) = cos(θ_9 − θ_c) exactly.**
Runner-up: 0.9985 = cos(2π/113) — the agonizing one-frequency margin; correctness with
no confidence. (That's Act 4's reason for multiple frequencies; a second circle is the
cure for the margin, not for the junk.)

## The lessons

1. **Minimality removes slack, not difficulty.** In a 128-dial model, "phase" feels
   like a free parameter — pick anything. At exactly 2 dials, every remaining degree of
   freedom is load-bearing. The spec was one random choice away from correct, and the
   error was not approximate — it was the precise, computable price of that choice.
2. **Orthogonality is information density.** An orthonormal pair is the unique basis in
   which "similarity" (dot product) and "geometry" (angle) agree with no correction
   term. Non-orthogonality doesn't lose information — the encoding is still perfect —
   it moves the cost downstream to every future dot product, forever, at rate u·v.
3. **Redundancy is the statistical version of orthogonality.** With D dials the junk is
   Σ_d α_d² cos(θ_s + θ_c − 2φ_d): a sum of arrows at the *doubled* phases. If the
   phases are diverse, the junk arrows spin and cancel **among themselves** — the same
   freeze-and-stack / spin-and-cancel mechanism as the DFT, reappearing as the reason
   over-parameterization is safe. Two dials must be perpendicular exactly; 128 dials
   only need their doubled phases to cancel collectively (u ⊥ v and ‖u‖ = ‖v‖ is
   precisely Σ α² e^{2iφ_d} = 0). Note it is cancellation, not dilution: the good terms
   stack coherently AND the junk terms destroy each other. Diversity of phases, not
   count of dials, is what's purchased.
4. **What the trained model must therefore be doing:** among everything gradient descent
   is choosing when it grokkifies the embedding, one implicit choice is driving the
   doubled-phase sum toward zero per key frequency — rounding the ellipse into a circle
   — because CE punishes the junk term directly. This is measurable in v3.

## Experiment 1 — measured in v3 (2026-07-05): confirmed, with a twist

Metric: E(k) = |Σ_d C_d²| / Σ_d |C_d|², where C_d is dim d's complex DFT coefficient at
frequency k (E = the junk-to-signal amplitude ratio; for the broken two-dial machine it
is exactly 0.4536). Normalized as Z = E·√D_eff against the random-phase null (Z ≈ 1 =
chance-level ellipticity, Z < 1 = rounder than chance), per snapshot, embedding and
unembedding sides.

Findings (grok ≈ step 1800):
1. **Circles are born elliptical.** During circuit formation (steps ~500–1500) several
   key frequencies sit *above* chance — k=49 in the embedding peaks at Z ≈ 1.7 at step
   1000 before collapsing to ≈ 0.2 by 2000. The forming plane really is an ellipse that
   gets rounded.
2. **Rounding happens at the grok, hardest on the readout side.** Embedding mean-Z over
   the circuit: ~0.8 pre-grok → ~0.3–0.45 after. Unembedding mean-Z: ~0.9–1.16 before →
   0.18 at step 2000 → **0.08 at 2250**. The decode side rounds most decisively — as the
   parable predicts, since the junk term taxes the logits directly and CE pushes there.
3. **It is a with-the-grok metric, not a leading one** (unlike rank collapse, which
   leads by ~500): the collapse is centered on 1750–2250.
4. **Fix B confirmed over Fix A**: D_eff ≈ 70 dims participate at k=5 throughout — the
   trained model rounds the circle collectively across ~70 redundant dials (doubled
   phases closing the loop), not by electing two clean perpendicular ones.
5. Residual ellipticity never reaches 0 (raw E ≈ 0.02–0.06 post-grok, noisy per
   snapshot) — the trained circle is round to a few percent, not exactly.

## Open experiment (for later)
- **Train the two-dial model** (the romantic one): can a network with a 2-dim embedding
  grok mod-113, and does it *find* φ₂ − φ₁ → π/2 (u·v → 0) on its own? Caveat: in the
  real architecture d_model is shared by attention/MLP/residual, so a literal d=2
  transformer may not train; the honest minimal version is learning E and W_U with the
  combine stage hard-wired. If it does grok: watch u·v(t) — the parable's junk
  amplitude — as a training curve.
