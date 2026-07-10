// nanda_jspace.cpp
// Phase 4 — J-space operations on the grokked mod-113 net.
//
// The per-token lens rows define J-space (Anthropic, July 2026): sparse
// nonnegative combinations of lens vectors that carry the model's "held
// answers". Here we (1) decompose readouts onto the unembedding rows by
// nonneg matching pursuit, and (2) intervene at posemb-out (boundary 2 — the
// last boundary whose downstream is NONLINEAR, so nothing is trivial):
//     inject  h += alpha*||h||*v_c_hat      -> does the answer move to c?
//     ablate  h -= proj onto its own top-k J-space atoms -> does accuracy die?
//     swap    a-slot embedding a -> a'      -> does the answer move to (a'+b)%p?
// Controls: random unit directions for inject/ablate.
//
// Usage: ./nanda_jspace <ckpt_dir> [step=9999] [n_fit=256] [n_eval=339] [n_trials=339]

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "nanda_net.hpp"

using Lens2T = LensTensor<NandaNet, 2>;              // Tensor<113, 3, 128>
using Act2T  = BoundaryActivation<NandaNet, 2>;      // Tensor<3, 128>
using Lens4T = LensTensor<NandaNet, 4>;              // Tensor<113, 128>

static constexpr size_t D2 = SeqLen * EmbDim;        // 384
static constexpr size_t D4 = EmbDim;                 // 128

// ── nonneg matching pursuit over unit atoms ──────────────────────────────────
struct MPResult {
    std::vector<std::pair<size_t, float>> picks; // (atom, coefficient)
    float var_explained = 0.f;
};

// atoms: n_atoms x dim (unit rows); h: dim
static MPResult MatchingPursuit(const std::vector<float> &atoms, size_t n_atoms,
                                const float *h, size_t dim, size_t kmax) {
    MPResult res;
    std::vector<float> r(h, h + dim);
    float h_sq = 0.f;
    for (size_t i = 0; i < dim; ++i) h_sq += h[i] * h[i];
    for (size_t k = 0; k < kmax; ++k) {
        size_t best = 0;
        float best_dot = 0.f;
        for (size_t c = 0; c < n_atoms; ++c) {
            float d = 0.f;
            for (size_t i = 0; i < dim; ++i) d += r[i] * atoms[c * dim + i];
            if (d > best_dot) { best_dot = d; best = c; }
        }
        if (best_dot <= 1e-4f * std::sqrt(h_sq)) break;
        res.picks.emplace_back(best, best_dot);
        for (size_t i = 0; i < dim; ++i) r[i] -= best_dot * atoms[best * dim + i];
    }
    float r_sq = 0.f;
    for (size_t i = 0; i < dim; ++i) r_sq += r[i] * r[i];
    res.var_explained = h_sq > 0.f ? 1.f - r_sq / h_sq : 0.f;
    return res;
}

static void normalize_rows(std::vector<float> &m, size_t rows, size_t dim) {
    for (size_t c = 0; c < rows; ++c) {
        float n = 0.f;
        for (size_t i = 0; i < dim; ++i) n += m[c * dim + i] * m[c * dim + i];
        n = std::sqrt(n) + 1e-30f;
        for (size_t i = 0; i < dim; ++i) m[c * dim + i] /= n;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: nanda_jspace <ckpt_dir> [step] [n_fit] [n_eval] [n_trials]\n"); return 1; }
    const std::string dir = argv[1];
    const size_t step     = argc > 2 ? std::stoul(argv[2]) : 9999;
    const size_t n_fit    = argc > 3 ? std::stoul(argv[3]) : 256;
    const size_t n_eval   = argc > 4 ? std::stoul(argv[4]) : 339;
    const size_t n_trials = argc > 5 ? std::stoul(argv[5]) : 339;

    auto net = std::make_unique<NandaNet>();
    net->Load(dir + "/snaps/snap_" + std::to_string(step) + ".bin");
    std::printf("loaded snapshot @ step %zu\n", step);

    std::mt19937 rng(11);

    auto encode1 = [](uint32_t idx, Tensor<1, SeqLen, Vocab> &x) {
        InputT Xs; OutputT Ys;
        Dataset::EncodeSample(idx, Xs, Ys);
        for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
    };

    // ── fit the boundary-2 mean lens (E over contexts) ───────────────────────
    std::printf("fitting boundary-2 lens over %zu contexts...\n", n_fit);
    ActivationLensAccumulator<NandaNet, 2> acc2;
    {
        Tensor<1, SeqLen, Vocab> x;
        for (size_t i = 0; i < n_fit; ++i) {
            encode1(static_cast<uint32_t>((i * Total) / n_fit), x);
            acc2.Add(*net, x);
        }
    }
    const Lens2T lens2 = acc2.Mean();
    std::printf("boundary-2 dispersion: %.3f\n", acc2.Dispersion());

    // boundary-4 lens (== W_U exactly; fit once, one context)
    Lens4T lens4;
    {
        Tensor<1, SeqLen, Vocab> x;
        encode1(0, x);
        lens4 = FitActivationLens<4, 1>(*net, x);
    }

    // unit atoms
    std::vector<float> atoms4(P * D4), atoms2(P * D2);
    for (size_t i = 0; i < atoms4.size(); ++i) atoms4[i] = lens4.flat(i);
    for (size_t i = 0; i < atoms2.size(); ++i) atoms2[i] = lens2.flat(i);
    normalize_rows(atoms4, P, D4);
    normalize_rows(atoms2, P, D2);

    // eval contexts: half-stride offset from the fit set
    std::vector<uint32_t> eval_idx(n_eval);
    for (size_t i = 0; i < n_eval; ++i)
        eval_idx[i] = static_cast<uint32_t>(((i * Total) / n_eval + Total / (2 * n_eval)) % Total);

    // ── A. matching pursuit at the readout ───────────────────────────────────
    size_t mp_top1 = 0, base_ok = 0;
    double mp_ve = 0.0, mp_k = 0.0;
    for (const uint32_t idx : eval_idx) {
        Tensor<1, SeqLen, Vocab> x;
        encode1(idx, x);
        const auto acts = net->ForwardAll<1>(x);
        const auto &h4b = acts.get<4>();
        float h4[D4];
        for (size_t k = 0; k < D4; ++k) h4[k] = h4b.flat(k);
        const auto mp = MatchingPursuit(atoms4, P, h4, D4, 10);
        const uint32_t ans = ((idx / P) + (idx % P)) % P;
        if (!mp.picks.empty() && mp.picks[0].first == ans) ++mp_top1;
        mp_ve += mp.var_explained;
        mp_k += static_cast<double>(mp.picks.size());
        size_t am = 0;
        const auto &lg = acts.get<5>();
        for (size_t v = 1; v < P; ++v) if (lg.flat(v) > lg.flat(am)) am = v;
        base_ok += (am == ans);
    }
    std::printf("\nA. matching pursuit @ readout (k<=10, %zu contexts):\n", n_eval);
    std::printf("   model acc %.3f | MP top-1 atom == answer: %.3f | mean varexp %.3f | mean k %.1f\n",
                double(base_ok) / n_eval, double(mp_top1) / n_eval, mp_ve / n_eval, mp_k / n_eval);

    // ── B. inject at boundary 2 ──────────────────────────────────────────────
    std::uniform_int_distribution<uint32_t> uctx(0, Total - 1), utok(0, P - 1);
    std::normal_distribution<float> gauss(0.f, 1.f);
    const float alphas[4] = {0.25f, 0.5f, 1.f, 2.f};
    std::printf("\nB. inject h2 += a*||h2||*v_c (lens row vs random control), %zu trials:\n", n_trials);
    std::printf("   %-8s %-12s %-12s\n", "alpha", "moved->c", "control");
    for (const float a : alphas) {
        size_t moved = 0, moved_ctl = 0;
        std::mt19937 trng(23); // same trials per alpha
        for (size_t t = 0; t < n_trials; ++t) {
            const uint32_t idx = uctx(trng);
            uint32_t c = utok(trng);
            const uint32_t ans = ((idx / P) + (idx % P)) % P;
            if (c == ans) c = (c + 1) % P;
            Tensor<1, SeqLen, Vocab> x;
            encode1(idx, x);
            const auto acts = net->ForwardAll<1>(x);
            const auto &h2 = acts.get<2>(); // Tensor<1,3,128>
            float hn = 0.f;
            for (size_t k = 0; k < D2; ++k) hn += h2.flat(k) * h2.flat(k);
            hn = std::sqrt(hn);

            Tensor<1, SeqLen, EmbDim> hin = h2, hctl = h2;
            // control direction: random unit vector (same per trial across alphas)
            float ctl[D2];
            float cn = 0.f;
            for (size_t k = 0; k < D2; ++k) { ctl[k] = gauss(trng); cn += ctl[k] * ctl[k]; }
            cn = std::sqrt(cn) + 1e-30f;
            for (size_t k = 0; k < D2; ++k) {
                hin.flat(k)  += a * hn * atoms2[c * D2 + k];
                hctl.flat(k) += a * hn * ctl[k] / cn;
            }
            const auto y  = net->ForwardFrom<1, 2>(hin);
            const auto yc = net->ForwardFrom<1, 2>(hctl);
            size_t am = 0, amc = 0;
            for (size_t v = 1; v < P; ++v) {
                if (y.flat(v)  > y.flat(am))   am = v;
                if (yc.flat(v) > yc.flat(amc)) amc = v;
            }
            moved     += (am == c);
            moved_ctl += (amc == c);
        }
        std::printf("   %-8.2f %-12.3f %-12.3f\n", a,
                    double(moved) / n_trials, double(moved_ctl) / n_trials);
    }

    // ── C. ablate top-k J-space projections at boundary 2 ───────────────────
    std::printf("\nC. ablate h2's own top-k J-space atoms (Gram-Schmidt projection removal):\n");
    std::printf("   %-6s %-12s %-12s\n", "k", "acc", "acc-random");
    for (const size_t kk : {4ul, 8ul, 16ul}) {
        size_t ok = 0, ok_ctl = 0;
        std::mt19937 trng(31);
        for (const uint32_t idx : eval_idx) {
            const uint32_t ans = ((idx / P) + (idx % P)) % P;
            Tensor<1, SeqLen, Vocab> x;
            encode1(idx, x);
            const auto acts = net->ForwardAll<1>(x);
            const auto &h2 = acts.get<2>();
            float h[D2];
            for (size_t k = 0; k < D2; ++k) h[k] = h2.flat(k);

            // pick atoms by MP against h, then remove the projection onto their span
            const auto mp = MatchingPursuit(atoms2, P, h, D2, kk);
            std::vector<float> basis; // Gram-Schmidt orthonormal
            basis.reserve(kk * D2);
            for (const auto &[c, coef] : mp.picks) {
                (void) coef;
                std::vector<float> u(atoms2.begin() + c * D2, atoms2.begin() + (c + 1) * D2);
                for (size_t bda = 0; bda * D2 < basis.size(); ++bda) {
                    float d = 0.f;
                    for (size_t k = 0; k < D2; ++k) d += u[k] * basis[bda * D2 + k];
                    for (size_t k = 0; k < D2; ++k) u[k] -= d * basis[bda * D2 + k];
                }
                float n = 0.f;
                for (size_t k = 0; k < D2; ++k) n += u[k] * u[k];
                n = std::sqrt(n);
                if (n < 1e-4f) continue;
                for (size_t k = 0; k < D2; ++k) u[k] /= n;
                basis.insert(basis.end(), u.begin(), u.end());
            }
            Tensor<1, SeqLen, EmbDim> habl = h2, hctl = h2;
            for (size_t bda = 0; bda * D2 < basis.size(); ++bda) {
                float d = 0.f;
                for (size_t k = 0; k < D2; ++k) d += habl.flat(k) * basis[bda * D2 + k];
                for (size_t k = 0; k < D2; ++k) habl.flat(k) -= d * basis[bda * D2 + k];
            }
            // control: remove projections onto the same NUMBER of random orthonormal dirs
            {
                std::vector<float> rb;
                for (size_t j = 0; j * D2 < basis.size(); ++j) {
                    std::vector<float> u(D2);
                    for (auto &v : u) v = gauss(trng);
                    for (size_t bda = 0; bda * D2 < rb.size(); ++bda) {
                        float d = 0.f;
                        for (size_t k = 0; k < D2; ++k) d += u[k] * rb[bda * D2 + k];
                        for (size_t k = 0; k < D2; ++k) u[k] -= d * rb[bda * D2 + k];
                    }
                    float n = 0.f;
                    for (size_t k = 0; k < D2; ++k) n += u[k] * u[k];
                    n = std::sqrt(n) + 1e-30f;
                    for (size_t k = 0; k < D2; ++k) u[k] /= n;
                    rb.insert(rb.end(), u.begin(), u.end());
                }
                for (size_t bda = 0; bda * D2 < rb.size(); ++bda) {
                    float d = 0.f;
                    for (size_t k = 0; k < D2; ++k) d += hctl.flat(k) * rb[bda * D2 + k];
                    for (size_t k = 0; k < D2; ++k) hctl.flat(k) -= d * rb[bda * D2 + k];
                }
            }
            const auto y  = net->ForwardFrom<1, 2>(habl);
            const auto yc = net->ForwardFrom<1, 2>(hctl);
            size_t am = 0, amc = 0;
            for (size_t v = 1; v < P; ++v) {
                if (y.flat(v)  > y.flat(am))   am = v;
                if (yc.flat(v) > yc.flat(amc)) amc = v;
            }
            ok     += (am == ans);
            ok_ctl += (amc == ans);
        }
        std::printf("   %-6zu %-12.3f %-12.3f\n", kk,
                    double(ok) / n_eval, double(ok_ctl) / n_eval);
    }

    // ── D. circle swap: replace the a-token's embedding at boundary 2 ───────
    // h2[0,:] -= E[:,a]; h2[0,:] += E[:,a'] — biases/pos-emb cancel. Predicted
    // answer moves to (a'+b) mod p if the circles are causal.
    const auto &EW = std::get<0>(net->all_params()).value; // embed W: Tensor<128,114>
    size_t swap_hit = 0, swap_kept = 0;
    {
        std::mt19937 trng(47);
        for (size_t t = 0; t < n_trials; ++t) {
            const uint32_t idx = uctx(trng);
            const uint32_t aa = idx / P, bb = idx % P;
            uint32_t a2 = utok(trng);
            if (a2 == aa) a2 = (a2 + 1) % P;
            Tensor<1, SeqLen, Vocab> x;
            encode1(idx, x);
            const auto acts = net->ForwardAll<1>(x);
            Tensor<1, SeqLen, EmbDim> h = acts.get<2>();
            for (size_t e = 0; e < EmbDim; ++e)
                h.flat(e) += EW.flat(e * Vocab + a2) - EW.flat(e * Vocab + aa);
            const auto y = net->ForwardFrom<1, 2>(h);
            size_t am = 0;
            for (size_t v = 1; v < P; ++v) if (y.flat(v) > y.flat(am)) am = v;
            swap_hit  += (am == (a2 + bb) % P);
            swap_kept += (am == (aa + bb) % P);
        }
    }
    std::printf("\nD. a-slot embedding swap a->a' at boundary 2 (%zu trials):\n", n_trials);
    std::printf("   answer moves to (a'+b) mod p: %.3f | stays at (a+b) mod p: %.3f\n",
                double(swap_hit) / n_trials, double(swap_kept) / n_trials);

    return 0;
}
