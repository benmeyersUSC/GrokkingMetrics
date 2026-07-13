// nanda_jlens_split.cpp
// The post-attention / pre-MLP boundary, exposed via NandaNetSplit (identical
// weights, transformer block split into its two residual sublayers).
//
// Per retained snapshot: verify functional equivalence with NandaNet (same
// logits from the same weights), fit the mean lens at attn-out (boundary 3 of
// the split net) over the SAME stride-sampled contexts as nanda_jlens, and dump
// the attn-out activations for the fit and eval sets.
//
// Emits <ckpt_dir>/jlens/jlens_split_<step>.bin:
//   u64 magic("JLSPLIT1") | u64 step | u64 1
//   u64 3 | u64 113 | u64 384 | f32 lens[113*384] | f32 rowcoh[113] | f32 coh | f32 dis
//   u64 n_ctx | u32 fit_idx[n] | u32 eval_idx[n]
//   f32 attn_acts fit [n*384] | f32 attn_acts eval [n*384]
//
// Usage: ./nanda_jlens_split <ckpt_dir> [n_ctx=256] [every=1]

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "nanda_net.hpp"

static constexpr uint64_t kMagic = 0x4A4C53504C495431ull; // "JLSPLIT1"
static constexpr size_t kAttnBoundary = 3;

template<typename T>
static void put(std::ofstream &f, const T &v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
static void put_flats(std::ofstream &f, const auto &t, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float v = t.flat(i);
        put(f, v);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nanda_jlens_split <ckpt_dir> [n_ctx=256] [every=1]\n";
        return 1;
    }
    const std::string ckpt_dir = argv[1];
    const size_t n_ctx = argc > 2 ? std::stoul(argv[2]) : 256;
    const size_t every = argc > 3 ? std::stoul(argv[3]) : 1;

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
    // descending: the final snapshot lands first so the site can bake early
    std::sort(snaps.rbegin(), snaps.rend());
    std::cout << snaps.size() << " snapshots (descending); " << n_ctx << " contexts\n";

    auto net  = std::make_unique<NandaNet>();
    auto snet = std::make_unique<NandaNetSplit>();
    size_t si = 0;
    for (const auto &[step, path] : snaps) {
        if (si++ % every != 0) continue;
        net->Load(path.string());
        snet->Load(path.string());

        // equivalence golden: same weights -> same logits through both nets
        {
            float worst = 0.f;
            for (size_t t = 0; t < 3; ++t) {
                InputT Xs; OutputT Ys;
                Dataset::EncodeSample(fit_idx[t * (n_ctx / 3)], Xs, Ys);
                Tensor<1, SeqLen, Vocab> x;
                for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
                const auto y1 = net->Forward<1>(x);
                const auto y2 = snet->Forward<1>(x);
                for (size_t c = 0; c < P; ++c)
                    worst = std::max(worst, std::abs(y1.flat(c) - y2.flat(c)));
            }
            if (worst > 1e-5f)
                std::cout << "  !! split-net equivalence violated: " << worst << "\n";
        }

        ActivationLensAccumulator<NandaNetSplit, kAttnBoundary> acc;
        Tensor<1, SeqLen, Vocab> x;
        for (const uint32_t idx : fit_idx) {
            InputT Xs; OutputT Ys;
            Dataset::EncodeSample(idx, Xs, Ys);
            for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
            acc.Add(*snet, x);
        }
        using Acc = ActivationLensAccumulator<NandaNetSplit, kAttnBoundary>;
        const auto mean = acc.Mean();
        const auto rows = acc.RowCoherence();

        std::ofstream f(ckpt_dir + "/jlens/jlens_split_" + std::to_string(step) + ".bin",
                        std::ios::binary);
        put(f, kMagic);
        put<uint64_t>(f, step);
        put<uint64_t>(f, 1);
        put<uint64_t>(f, kAttnBoundary);
        put<uint64_t>(f, Acc::TgtSize);
        put<uint64_t>(f, Acc::ActSize);
        put_flats(f, mean, Acc::TgtSize * Acc::ActSize);
        put_flats(f, rows, Acc::TgtSize);
        put(f, acc.Coherence());
        put(f, acc.Dispersion());

        put<uint64_t>(f, n_ctx);
        f.write(reinterpret_cast<const char *>(fit_idx.data()),
                static_cast<std::streamsize>(n_ctx * sizeof(uint32_t)));
        f.write(reinterpret_cast<const char *>(eval_idx.data()),
                static_cast<std::streamsize>(n_ctx * sizeof(uint32_t)));

        auto dump_acts = [&](const std::vector<uint32_t> &idxs) {
            for (const uint32_t idx : idxs) {
                InputT Xs; OutputT Ys;
                Dataset::EncodeSample(idx, Xs, Ys);
                for (size_t k = 0; k < InputT::Size; ++k) x.flat(k) = Xs.flat(k);
                const auto acts = snet->template ForwardAll<1>(x);
                put_flats(f, acts.template get<kAttnBoundary>(), SeqLen * EmbDim);
            }
        };
        dump_acts(fit_idx);
        dump_acts(eval_idx);
        std::cout << "step " << step << " done (dispersion "
                  << acc.Dispersion() << ")\n";
    }
    std::cout << "wrote " << ckpt_dir << "/jlens/jlens_split_<step>.bin\n";
    return 0;
}
