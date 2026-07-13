# Abstraction and leverage in J-space

*Experiment program. Two questions, one instrument.*

> **Q1 — Where does abstraction take root?** Give the model N prompts that *mean the same thing* and watch, layer by layer, for the moment their **dispositions** snap together. Not their states — their dispositions.
>
> **Q2 — Where does the model steer hardest?** Push a fixed-size nudge into J-space at depth ℓ and measure how much behavior moves. Sweep ℓ. That profile is a **leverage curve**, and it's a causal-emergence measurement in disguise.

The J-lens is what makes both askable: it turns "what is this state disposed toward" into a computable, token-named vector.

---

## 0. The instrument, and the one move the whole program rests on

For residual `h` at source depth `ℓ`, position `t`:

```
L_ℓ h  ∈ ℝ^vocab     entry_token = ∂logit_token/∂h,
                     averaged over contexts and over targets t′ ≥ t, applied to this h
```

Two sameness metrics:

- **State:** `cos(h_i, h_j)` — same internal representation?
- **Disposition:** `cos(L_ℓ h_i, L_ℓ h_j)` — same *implications for behavior*?

Always **full vocab-length cosine** over the whole disposition vector. Never argmax, never top-k. We compare the *shape of the implied continuation field*, not the winning token. Argmax agreement is confluence; vector agreement is shared frame.

**Why the pair, not either alone.** `L` is many-to-one, so identical `h` ⟹ identical `Lh`, never the reverse: **dispositional-sameness is a strictly coarser equivalence than state-sameness** — state-sameness quotiented by what the output cannot see (`ker L`). Two states differing *inside* `ker L` are different latents with identical disposition: the network hasn't merged them, but has made their difference causally inert. That is abstraction visible only from the output's vantage.

```
gap(i,j; ℓ) = cos(L_ℓ h_i, L_ℓ h_j) − cos(h_i, h_j)
```

A large positive gap = the model carries a distinction internally that it has **decided doesn't matter for behavior**. Representational detail preserved, dispositional detail discarded. That's abstraction being *performed*.

**Axis hygiene (settle this once).**
- **Source depth `ℓ`** — which `h` we differentiate *from*. **Vary freely.**
- **Target — always logits.** The unembedding^T composition is what makes disposition **legible** (rows are token-named). Retargeting to an interior activation gives a residual→residual Jacobian with un-named rows — illegible, off-program. **Never retarget.**

---

# PART I — Abstraction

## 1. Experiment 0 — modular net: perfect, and therefore degenerate

The `(a+b) mod p` transformer (p=113) is the right place to **prove the machinery** and the wrong place to **test the real question**.

**Why perfect.** Its entire downstream behavior is one readout, argmaxed — the disposition is **terminal** (the answer is a fixed point; nothing downstream re-routes on it). At p=113 the lens can be fit **per-context, exactly, full-batch** — zero sampling error. Clean bench for the pipeline and for the equivalence-class prediction:

> All pairs `(a,b)` with fixed `a+b mod p` should become **dispositionally identical** (same answer) while remaining **state-distinct** (different operands).
> **Prediction:** `cos(Lh) → 1` within each sum-class while `cos(h)` stays well below 1.

Cross with grokking snapshots: watch within-class dispositional cosine climb across training; locate the climb against the grok step and the effective-rank curve. Abstraction crystallizing = dispositional collapse within class.

**Why degenerate.** Terminal disposition ⟹ the depth-profile collapses to a point. There's no rich downstream *bundle* to share, only the answer logit — so "same disposition" can't be distinguished from "same next token," because there *is* only the next token. It validates the metric, then hands off. Non-terminal abstraction is the LLM story.

## 2. **Experiment 1 (headline) — the convergence plot: when disposition snaps together**

**The question.** Ask the same thing ten different ways. The model *should* end up disposed to do the same thing. In early layers it can't be — the surface idiosyncrasies (word choice, syntax, framing) are still being chewed. Somewhere in depth, the network throws the surface away and realizes it should be doing the same thing. **Where?**

**The plot.** For a set of meaning-equivalent prompts `{p_1 … p_N}`, sweep source depth ℓ and plot both curves on the same axes:

```
D(ℓ) = mean over pairs of  cos(L_ℓ h_i, L_ℓ h_j)      ← disposition
S(ℓ) = mean over pairs of  cos(h_i, h_j)               ← state
```

Read three things off it:

1. **The snap.** The layer where `D(ℓ)` turns up sharply. That is *abstraction taking root* — the depth at which "how it was asked" stops mattering and "what was asked" is all that's left.
2. **The gap `D(ℓ) − S(ℓ)`.** How much coarser the disposition is than the state, at every depth. The area between the curves is the total abstraction performed.
3. **The onset asymmetry — the signature to hunt.** If `D` rises *before* `S`, the network has **decided two prompts mean the same thing before it has finished making their representations look alike.** Same-disposition-before-same-state. Strictly stronger and more surprising than "they end up similar."

> Report the **full curves and the area between them**, not a threshold-crossing layer. A fixed threshold is arbitrary and a reviewer will say so.

**Why disposition and not state is the right y-axis** (the point that makes the plot mean something): the states may *never* fully merge — the model can carry provenance ("this was phrased as a riddle") to the very end. But dispositions can converge completely. And disposition is what the model *acts on*. Plotting `S` alone would understate abstraction; plotting `D` alone would hide the residue. Plot both.

## 3. Experiment 2 — the control that makes Experiment 1 mean anything

A skeptic's reply to Experiment 1: *"Of course `Lh` converges — `L` is low-rank, it merges everything."* Kill this with a **double dissociation**:

| | **same surface** | **different surface** |
|---|---|---|
| **same meaning** | — | **paraphrases** → want `cos(Lh) ≫ cos(h)` |
| **different meaning** | **minimal pairs** → want `cos(Lh) ≪ cos(h)` | — |

- **Paraphrases:** *"What's the capital of France?"* / *"France's capital city is?"* — surface differs, meaning identical.
- **Minimal pairs:** *"capital of France"* / *"capital of Finland"*; or the same tokens reordered to flip the ask — surface nearly identical, meaning differs.

**The headline number is the interaction:**

```
abstraction_signal = gap(paraphrases) − gap(minimal pairs)
```

A merely-lossy low-rank map would collapse the minimal pairs *too*. A map that **separates minimal pairs while merging paraphrases** is coarsening along **meaning**, controlled for rank. Without this cell, Experiment 1 is unfalsifiable.

## 4. Experiment 3 — two read positions = two verbs of abstraction

Same two vectors, read at two source positions; they answer different questions.

**(a) The entity/content token** (e.g. the position of *"France"*, shared across prompts). `h` still holds the subject broadly; `Lh` fans across the **reader-bundle** (capital-ness, language-ness, continent-ness at once). Tests abstraction as **convergence** — disposition converging while the subject is still live. *Caveat:* the token sits at different absolute positions across paraphrases; alignment is messy. Accept it — this cell is about richness.

**(b) Final position (−1)** — the last word of the prompt. `h_{−1}` is a **more resolved** state; attention has gathered everything, the mind is near-decided. For terminal rephrasings this *forces* `cos(Lh_{−1}) ≈ 1`. Looks boring; is secretly the **sharpest control**, because pinning one term near-constant turns the metric into a pure readout of the other:

> With disposition saturated, the **state** gap carries all the signal.
> `cos(h_{−1}) < 1` under `cos(Lh_{−1}) ≈ 1` = **residue abstraction failed to erase** — the state still remembers it was a riddle vs a plain ask; provenance the disposition already discarded.
> This tests abstraction as **forgetting**.

**Convergence and forgetting are separable abilities.** A model can route everything to the right answer yet keep a full transcript of how it was asked, distinguishable in `h` forever. Position −1 is the only place forgetting is visible — *because* disposition is saturated and stops competing for the signal.

| position | disposition | the gap measures | verb |
|---|---|---|---|
| entity token | rich, multi-target | `D` rising above `S` | **convergence** |
| −1 | saturated | residual `S < 1` under `D ≈ 1` | **forgetting** |

## 4b. **Experiment 3b — the robustness transfer function: how much surface can the abstraction absorb?**

*Prerequisite: Experiment 1. "Robustness" is meaningless without a depth to measure it at — trivially low at layer 2 (everything is still surface), trivially high at the last layer (everything has funneled). The quantity of interest is **post-abstraction robustness**: how much surface noise the abstraction can launder once it exists. Find the snap first, then measure here.*

**The upgrade over Experiment 1.** Paraphrases hold meaning fixed and vary surface *categorically*. Here we vary surface along **graded, orthogonal axes with a dose we control** — so the output is not a cosine between two points but a **response curve**: dispositional divergence as a function of perturbation magnitude.

```
R_axis(δ, ℓ) = 1 − cos( L_ℓ h(x),  L_ℓ h(perturb_axis(x, δ)) )
```

Sweep the dose δ, sweep depth ℓ, one curve per axis. This turns *"does it abstract?"* into **"what is the abstraction's transfer function?"** — you get a slope, a threshold, a saturation point, and a *shape*.

**The axes (each a different kind of insult, and the comparison across them is the finding):**

| axis | dose | what it actually tests |
|---|---|---|
| **typos** | % chars corrupted | pure **noise rejection** — the perturbation carries no information |
| **disfluency** | "um"/"uh"/filler density | noise rejection, but with *token-level* plausibility (real words, wrong content) |
| **irrelevant detail** | # of appended true-but-unrelated clauses | **distractor suppression** — information that is real but off-task |
| **synonym swap** | # substitutions | lexical invariance at fixed syntax |
| **terseness ↔ fluff** | tokens per unit meaning | **compression invariance** — see below; the deep one |

**Prediction to write down first:** the profiles will be *sharply different in shape*, not merely in magnitude. Plausible: near-immunity to typos, mild sensitivity to disfluency, and steep sensitivity to irrelevant detail (because a distractor is *real information* the network is obliged to represent, whereas a typo is noise it can denoise). If that ordering holds, it says the model's abstraction is a **content filter**, not a noise filter — it laundders garbage easily and struggles with relevance.

### The terseness axis is the special one

Typos and disfluencies are **noise** — zero information, so discarding them is pure robustness. Fluff-vs-terseness is different: it varies **information density per token while holding total information fixed.** A verbose phrasing spreads the same meaning over more tokens; a terse one concentrates it. So this doesn't test noise-rejection at all — it tests whether disposition is invariant to *how the same information is distributed across the sequence*. That is a **compression-invariant representation of meaning**, and it's a much stronger property.

Sharp, answerable predictions:
- **Convergence:** `L h` at the decision position should converge for terse and fluffy phrasings of identical content — even though the terse version reached that state in far fewer tokens, doing far more work per token.
- **Delay:** does fluff **push the snap deeper**? Plot the depth-of-abstraction (from Experiment 1) as a function of verbosity. If verbose phrasings abstract *later in depth*, the model is spending layers on compression before it can spend them on meaning. That's a real, novel, and beautiful result.
- **Cost:** terse and verbose may reach the same disposition but with different *dispersion* — does the model's confidence in the shared frame differ by how it was told?

### The target signature (Ben's phrasing, kept)

> **Early latent states wildly different; later disposition states extremely similar.**

That's not a single number, it's a **divergence-collapse profile**: how far apart the perturbed inputs start in `h`, and how completely the network launders that difference away in `Lh` by depth. Plot `1 − cos(h)` and `1 − cos(Lh)` vs depth on the same axes, one panel per perturbation axis, one line per dose.

### Why this is newly possible (state it precisely — don't overclaim)

You still run a **forward pass** to get `h`. What you no longer need is to **generate**, and to then do post-hoc semantic analysis of the generated text. Historically, robustness has been measured *behaviorally*: generate outputs under perturbation, judge them (expensive, subjective, LLM-as-judge, contaminated by sampling noise). With the lens it is a **cosine between two vectors from two forward passes**.

> **Robustness becomes a geometric measurement instead of a behavioral one.**

That is the claim, and it is strong enough without overreaching. The lens is what makes latent space *legible in a principled way* — not the logit lens, which pretends a mapping exists; here we **compute** the mapping. That legibility is what turns a linguistics question ("is the model robust to disfluency?") into a geometry question ("what is `1 − cos(L h, L h′)`?").

### Why linguistics will want this

Every one of these axes is a live question in psycholinguistics and NLP — graceful degradation under noise, the processing cost of disfluency, the semantics of redundancy, distractor interference. The lens makes each of them a **measurement on an internal state** rather than an inference from behavior. Expect this to be the most-borrowed piece of the program.

---

# PART II — Leverage

## 5. Experiment 4 — the leverage curve: where does a push move the most?

**The question.** Fix a J-space direction `v̂_t` and a step size α. Inject `h_ℓ ← h_ℓ + α·‖h_ℓ‖·v̂_t` at depth ℓ. Measure how much behavior moves (top-1 movement rate; KL of the output distribution; success rate of the intended semantic redirect). **Sweep ℓ.** Plot leverage vs depth.

**Two opposing mechanisms — which is why the curve is interesting rather than obvious:**

- **Amplification (favors early).** An early perturbation is fed to *all* the downstream machinery — the entire coherence-making apparatus re-cores its computation around the changed content. You don't just move a logit; you move the *reasoning*, and everything re-coheres around the new subject. That's the France→China property: whole belief-sets re-subject with no seam.
- **Washout (favors late).** An early perturbation must survive many layers of attention mixing and normalization; much of it gets absorbed, re-normalized, or overwritten by context before it reaches the output.

**Prediction:** the curve is **non-monotonic**, peaking in a middle band — deep enough to survive, shallow enough to recruit the downstream machine. If that peak coincides with the paper's **workspace layers**, that is independent confirmation of the workspace regime *from leverage alone*, without any of their probing machinery. Write this prediction down before running it.

**The coherence asymmetry — measure it alongside magnitude.** Leverage isn't just *how much* moves; it's *how cleanly*. Score two things at every depth:
- **Magnitude:** did the output change?
- **Coherence:** did it change into something *whole* — the entire belief-set re-subjected — or into a **seam** (right token, broken reasoning)?

**Prediction:** late-layer pushes buy **magnitude without coherence** (you get the token, you don't get the re-reasoning — you've reached past the synthesizer to its output). Early/mid pushes buy **both**. That is the "upstream of the coherence-maker" thesis, made a number.

> **Steering at the logits is censorship. Steering at the latent is conviction.**

## 6. Experiment 5 — causal emergence, with the lens as the coarse-graining

**The frame.** Hoel-style emergence: a **coarse-grained** variable carries *more* effective information over the outcome than the fine-grained variables composing it. The J-lens hands us the coarse-graining for free:

- **Micro variables:** the `d_model` raw coordinates of `h`.
- **Macro variable:** a **J-space coordinate** — the projection of `h` onto a lens direction `v_t`. A bundle of micro-dimensions bound into one semantically-named handle.

**The measurement.** Compare interventions at *matched cost* (matched perturbation norm, or matched bits injected):
- push along **random directions** in `h` (micro),
- push along **J-space directions** `v_t` (macro),
- push at the **logits** directly (maximally-downstream fine intervention).

Score effective information / behavioral movement / coherence for each. **The emergence claim:** the macro handle beats matched-cost micro handles — and, sharpened, its advantage **grows with depth-of-recruitment**, tracking the leverage curve of §5.

> The paper's own control is already the seed of this: **k≈16 J-space directions kill the behavior; k≈16 random directions do nothing.** That *is* an emergence result — it simply hasn't been named as one. Run it as a **dose–response across depth × direction-type** and it becomes one.

**Why language is the perfect substrate** (thesis-level): human language already exhibits absurd **leverage-per-bit** — we steer enormous behavioral change through tiny lexical differences, filtered through a very complicated system. That thinness *is* what makes language language. An LLM is the first system where we can put a **number** on it: how much behavioral redirection does one macro-handle buy, at what depth, at what cost, with what coherence. Causal emergence stops being a philosophical claim and becomes a dose–response curve.

## 7. Experiment 6 — the playground (build it)

An interactive latent-space explorer: fly `h` around, watch the disposition vector update live, see which tokens brighten. Not a demo *of* the math — **the math made walkable.**

- Player position = `h`.
- Lens rows = **fixed beacons at infinity** (directions, not places).
- Looking / moving = the dot product `v_t · h`; the HUD ranks tokens by it.
- **Swap** = a rail you snap along: translate by `(v_China − v_France)` scaled by your current France-coordinate. Other beacons don't move — the whole belief re-subjects around you.
- **Inject** = a thruster along `v̂_t` with a throttle (α).
- **Ablate** = a wall that flattens motion in the top-k beacon directions.

**Design law:** the beacons **never** move with the camera. The disposition field is the model's geometry, not the viewer's. Player moves; the loom stays strung.

**Honest-dimensionality fork:** (a) a genuinely tiny `d_model = 2–3` net where the player is *literally* in residual space and every dot product shown is real — start here; the modular net's frequency plane is the natural home, with 113 answer-beacons on the true circle. Or (b) a real net with a PCA shadow, clearly labeled a shadow. Earn (b) by doing (a).

---

## 8. Protocol notes & known subtleties

- **Vocab-length cosine, always.** Argmax is confluence, not frame.
- **Position handling.** Compare at decision-carrying positions. Don't let prompt-length differences leak into `cos(h)` and inflate the gap.
- **Why `L h_{−1}` is *not* muzzled to the next token.** The averaged operator was shaped by all (source, future-target) pairs across the corpus; applying it at −1 yields the full averaged disposition of a residual like this one. The *concentration* at −1 lives in the **activation** (`h_{−1}` is already near-decided), not in the operator. That's why −1 reads "resolved verdict" and the entity token reads "live frame" — same operator, differently-resolved inputs.
- **Teacher-forcing bias (known; out of scope).** `L` integrates `t′ ≥ t` through the **written prompt's** attention routing, not through futures the model would *sample*. So `Lh` is not a predictor of free generation. For this program that's exactly right: we feed fixed prompts and ask what dispositional field *those prompts* induce. The bias would only bite if we read `Lh_{−1}` as "what it's about to say if we let it run." We don't.
- **Why there is no exact composed matrix, ever.** If the net were linear, every layer's "lens" would be an exact composed matrix `W_final ⋯ W_ℓ` — no averaging, no lens needed. Nonlinearity forbids it: the effective linear map from `h_ℓ` to logits **is a function of `h`**. Each pointwise Jacobian is *exactly right locally*; the lens stitches exact local truths into a global object by averaging — and **the dispersion of per-context Jacobians around `E[J]` is the precise bill for that stitching.** The lens is a linearization; the dispersion is the price.

## 9. Predictions, in falsifiable form

1. **Modular net:** within-class `cos(Lh) → 1`, `cos(h)` bounded away from 1; the collapse tracks the grok step.
2. **Convergence plot:** `D(ℓ)` shows a clear snap; `D` rises *earlier in depth* than `S` on paraphrases (**decision before merger**).
3. **Control:** minimal-pair gap is *negative*. The interaction (paraphrase gap − minimal-pair gap) is the headline number. If **both** cells come back positive → low-rank blur, not abstraction, and the control did its job.
4. **Forgetting at −1:** disposition saturates while state stays distinguishable; the residual state-gap scales with surface distance (riddle-vs-plain > synonym swap).
4b. **Robustness profiles differ in *shape*, not just magnitude:** near-immunity to typos, mild sensitivity to disfluency, steep sensitivity to irrelevant detail. If so, the abstraction is a **content filter, not a noise filter**.
4c. **Compression invariance:** terse and verbose phrasings of identical content converge in disposition — but verbose phrasings **abstract later in depth** (fluff delays the snap).
5. **Leverage curve:** non-monotonic in depth, peaking in a middle band; the peak coincides with the workspace layers.
6. **Coherence asymmetry:** late pushes → magnitude without coherence (seams); early/mid → both.
7. **Emergence:** J-space macro-handles beat matched-cost micro (random-direction) interventions, and the advantage tracks the leverage curve.

---

## Appendix — one-line reductions

- Disposition = `Lh` (a vocab-length arrow); state = `h`. Abstraction = agreement arrives **coarser and sooner** in the arrow than in the state.
- Dispositional-sameness = state-sameness quotiented by `ker L`.
- **Headline plot:** `D(ℓ)` and `S(ℓ)` on one axis, across paraphrases. Find the snap. Find whether `D` leads `S`.
- **Headline number:** paraphrase gap − minimal-pair gap.
- **Headline curve:** leverage vs depth — non-monotonic, peaking at the workspace.
- Steering at the logits is censorship. Steering at the latent is conviction.
- The modular net proves the machinery, then must be set aside: its disposition is terminal.
