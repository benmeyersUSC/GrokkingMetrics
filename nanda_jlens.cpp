// nanda_jlens.cpp
// Metric IV — the Jacobian lens — over training, for the Nanda mod-113 net.
//
// For each retained weight snapshot (snaps/snap_<step>.bin), fit the activation
// lens L_l = E_contexts[d logits / d h_l] at the four interior boundaries:
//   1 = embed-out   Tensor<3,128>
//   2 = posemb-out  Tensor<3,128>
//   3 = txf-out     Tensor<3,128>
//   4 = readout     Tensor<128>
// The fit is exact per context (Batch=1 backward per one-hot logit cotangent);
// the accumulator streams the mean lens AND the cross-context dispersion
// (1 - ||E_c[L]||^2 / E_c[||L||^2]) — the accord-ratio idea across contexts.
// Boundary 4's lens must equal unembed.W with dispersion == 0 (golden anchor,
// checked here at every snapshot).
//
// Contexts: a deterministic stride sample of the canonical (a,b) grid for the
// fit set, and the half-stride-offset sample as a held-out eval set. For both
// sets we also dump the boundary activations and final logits so the Python
// side can apply mean lenses, compute the naive logit-lens (J=I) baseline from
// unembed.W, and plot disposition-through-depth without re-running C++.
//
// Emits into <ckpt_dir>/jlens/jlens_<step>.bin (format below).
//
// Usage: ./nanda_jlens <ckpt_dir> [n_ctx=256] [every=1]

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "nanda_net.hpp"

static constexpr uint64_t kMagic   = 0x4A4C454E53303031ull; // "JLENS001"
static constexpr size_t   kBounds  = 4;                     // boundaries 1..4

template<typename T>
static void put(std::ofstream &f, const T &v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
static void put_flats(std::ofstream &f, const auto &t, const size_t n, const size_t off = 0) {
    for (size_t i = 0; i < n; ++i) {
        const float v = t.flat(off + i);
        put(f, v);
    }
}

// Fit one boundary over the fit set; write mean lens + row coherence + scalars.
template<size_t B>
static void fit_boundary(NandaNet &net, const std::vector<uint32_t> &fit_idx, std::ofstream &f) {
    ActivationLensAccumulator<NandaNet, B> acc;
    Tensor<1, InputT::Shape[0], InputT::Shape[1]> x;
    for (const uint32_t idx : fit_idx) {
        InputT Xs; OutputT Ys;
        Dataset::EncodeSample(idx, Xs, Ys);
        for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
        acc.Add(net, x);
    }
    using Acc = ActivationLensAccumulator<NandaNet, B>;
    const auto mean = acc.Mean();
    const auto rows = acc.RowCoherence();
    const float coh = acc.Coherence();
    const float dis = acc.Dispersion();

    put<uint64_t>(f, B);
    put<uint64_t>(f, Acc::TgtSize);
    put<uint64_t>(f, Acc::ActSize);
    put_flats(f, mean, Acc::TgtSize * Acc::ActSize);
    put_flats(f, rows, Acc::TgtSize);
    put(f, coh);
    put(f, dis);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nanda_jlens <ckpt_dir> [n_ctx=256] [every=1]\n";
        return 1;
    }
    const std::string ckpt_dir = argv[1];
    const size_t n_ctx = argc > 2 ? std::stoul(argv[2]) : 256;
    const size_t every = argc > 3 ? std::stoul(argv[3]) : 1;

    // Deterministic context sets: fit = stride sample of the canonical grid,
    // eval = the same sample shifted by half a stride (held out from the fit).
    std::vector<uint32_t> fit_idx(n_ctx), eval_idx(n_ctx);
    for (size_t i = 0; i < n_ctx; ++i) {
        fit_idx[i]  = static_cast<uint32_t>((i * Total) / n_ctx);
        eval_idx[i] = static_cast<uint32_t>(((i * Total) / n_ctx + Total / (2 * n_ctx)) % Total);
    }

    std::vector<std::pair<uint64_t, std::filesystem::path>> snaps;
    const std::regex pat(R"(snap_(\d+)\.bin)");
    for (const auto &e : std::filesystem::directory_iterator(ckpt_dir + "/snaps")) {
        std::smatch m;
        const std::string name = e.path().filename().string();
        if (std::regex_match(name, m, pat))
            snaps.emplace_back(std::stoull(m[1]), e.path());
    }
    std::sort(snaps.begin(), snaps.end());
    std::cout << snaps.size() << " weight snapshots found; fitting every " << every
              << " with " << n_ctx << " fit + " << n_ctx << " eval contexts\n";

    std::filesystem::create_directories(ckpt_dir + "/jlens");

    auto net = std::make_unique<NandaNet>();
    size_t si = 0;
    for (const auto &[step, path] : snaps) {
        if (si++ % every != 0) continue;
        net->Load(path.string());

        std::ofstream f(ckpt_dir + "/jlens/jlens_" + std::to_string(step) + ".bin",
                        std::ios::binary);
        put(f, kMagic);
        put<uint64_t>(f, step);
        put<uint64_t>(f, kBounds);

        fit_boundary<1>(*net, fit_idx, f);
        fit_boundary<2>(*net, fit_idx, f);
        fit_boundary<3>(*net, fit_idx, f);
        fit_boundary<4>(*net, fit_idx, f);

        // Golden anchor at every snapshot: boundary-4 lens == unembed.W exactly,
        // dispersion 0. Re-fit tiny (8 contexts) and compare against the loaded W.
        {
            ActivationLensAccumulator<NandaNet, 4> acc;
            Tensor<1, SeqLen, Vocab> x;
            for (size_t i = 0; i < 8; ++i) {
                InputT Xs; OutputT Ys;
                Dataset::EncodeSample(fit_idx[i * (n_ctx / 8)], Xs, Ys);
                for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
                acc.Add(*net, x);
            }
            const auto L4 = acc.Mean();
            const auto &WU = std::get<0>(net->block<4>().all_params()).value; // Tensor<P, EmbDim>
            float worst = 0.f;
            for (size_t i = 0; i < WU.Size; ++i)
                worst = std::max(worst, std::abs(L4.flat(i) - WU.flat(i)));
            const float dis = acc.Dispersion();
            if (worst > 1e-5f || dis > 1e-4f)
                std::cout << "  !! golden anchor violated: |L4-W|=" << worst
                          << " dispersion=" << dis << "\n";
        }

        // Context block: indices, then per boundary the activations for the fit
        // and eval sets, then logits for both sets.
        put<uint64_t>(f, n_ctx);
        f.write(reinterpret_cast<const char *>(fit_idx.data()),
                static_cast<std::streamsize>(n_ctx * sizeof(uint32_t)));
        f.write(reinterpret_cast<const char *>(eval_idx.data()),
                static_cast<std::streamsize>(n_ctx * sizeof(uint32_t)));

        auto dump_acts = [&](const std::vector<uint32_t> &idxs) {
            for (const uint32_t idx : idxs) {
                InputT Xs; OutputT Ys;
                Dataset::EncodeSample(idx, Xs, Ys);
                Tensor<1, SeqLen, Vocab> x;
                for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
                const auto acts = net->template ForwardAll<1>(x);
                put_flats(f, acts.template get<1>(), SeqLen * EmbDim);
                put_flats(f, acts.template get<2>(), SeqLen * EmbDim);
                put_flats(f, acts.template get<3>(), SeqLen * EmbDim);
                put_flats(f, acts.template get<4>(), EmbDim);
                put_flats(f, acts.template get<5>(), P);
            }
        };
        dump_acts(fit_idx);
        dump_acts(eval_idx);

        std::cout << "step " << step << " done\n";
    }
    std::cout << "wrote " << ckpt_dir << "/jlens/jlens_<step>.bin\n";
    return 0;
}
