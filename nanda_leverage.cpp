// nanda_leverage.cpp
// Metric III + Metric II over training, for the Nanda mod-113 net.
//
// 1. StructuralPotential: mean per-parameter Jacobian L2 norm over K random
//    Xavier inits — architecture-only influence distribution, disentangled
//    from any particular init or training run. Computed once, cached.
// 2. Realized leverage: per-parameter Jacobian L2 norm at each retained weight
//    snapshot (snaps/snap_<step>.bin) of a training run.
//
// The ratio realized/potential per parameter points at which parameters carry
// the trained function relative to their architectural prior.
//
// Emits into <ckpt_dir>:
//   structural_potential.bin  — uint64 P, then P floats
//   leverage_realized.bin     — uint64 n_snaps, uint64 P, then per snap:
//                               uint64 step + P floats
//   param_manifest.txt        — one line per Param tensor: size
//
// Usage: ./nanda_leverage <ckpt_dir> [KInits=50]

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "nanda_net.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nanda_leverage <ckpt_dir> [KInits]\n";
        return 1;
    }
    const std::string ckpt_dir = argv[1];
    const size_t KInits = argc > 2 ? std::stoul(argv[2]) : 128;

    constexpr size_t Pc = NandaNet::TotalParamCount;
    std::cout << "TotalParamCount = " << Pc << "\n";

    // Full-token-coverage reference set: sample i is (a=i, b=(i+57) mod P) for
    // i in 0..56 — position a covers tokens 0..56, position b covers the rest,
    // EQ is always present. Every embedding row receives gradient signal, so
    // per-param leverage is fair for the embedding (unlike a small random batch,
    // where absent tokens' rows have identically zero Jacobian).
    constexpr size_t RefB = 57;
    typename PrependBatch<RefB, InputT>::type X_ref;
    {
        InputT Xb; OutputT Yb;
        for (size_t i = 0; i < RefB; ++i) {
            const uint32_t a = static_cast<uint32_t>(i);
            const uint32_t b = static_cast<uint32_t>((i + RefB) % P);
            Dataset::EncodeSample(a * P + b, Xb, Yb);
            for (size_t k = 0; k < InputT::Size; ++k)
                X_ref.flat(i * InputT::Size + k) = Xb.flat(k);
        }
    }

    // Param manifest: flat sizes in all_params() order (for downstream labeling).
    {
        auto net = std::make_unique<NandaNet>();
        std::ofstream mf(ckpt_dir + "/param_manifest.txt");
        std::apply([&](const auto &... ps) { ((mf << ps.Size << "\n"), ...); },
                   net->all_params());
    }

    // ── Metric III: StructuralPotential (cached) ─────────────────────────────
    const std::string sp_path = ckpt_dir + "/structural_potential.bin";
    if (!std::filesystem::exists(sp_path)) {
        std::cout << "computing StructuralPotential over " << KInits << " inits...\n";
        std::vector<float> sp(Pc, 0.f);
        for (size_t k = 0; k < KInits; ++k) {
            auto temp = std::make_unique<NandaNet>(); // fresh Xavier init
            const auto lev = ComputeJacobianNorms<NandaNet, RefB>(*temp, X_ref);
            for (size_t i = 0; i < Pc; ++i) sp[i] += lev[i];
            if ((k + 1) % 10 == 0) std::cout << "  init " << (k + 1) << "/" << KInits << "\n";
        }
        for (auto &v : sp) v /= static_cast<float>(KInits);
        std::ofstream f(sp_path, std::ios::binary);
        const uint64_t pc = Pc;
        f.write(reinterpret_cast<const char *>(&pc), sizeof(pc));
        f.write(reinterpret_cast<const char *>(sp.data()),
                static_cast<std::streamsize>(sp.size() * sizeof(float)));
        std::cout << "wrote " << sp_path << "\n";
    } else {
        std::cout << "StructuralPotential cached at " << sp_path << "\n";
    }

    // ── Metric II at each snapshot ───────────────────────────────────────────
    std::vector<std::pair<uint64_t, std::filesystem::path>> snaps;
    const std::regex pat(R"(snap_(\d+)\.bin)");
    for (const auto &e : std::filesystem::directory_iterator(ckpt_dir + "/snaps")) {
        std::smatch m;
        const std::string name = e.path().filename().string();
        if (std::regex_match(name, m, pat))
            snaps.emplace_back(std::stoull(m[1]), e.path());
    }
    std::sort(snaps.begin(), snaps.end());
    std::cout << snaps.size() << " weight snapshots found\n";

    std::ofstream f(ckpt_dir + "/leverage_realized.bin", std::ios::binary);
    const uint64_t n_snaps = snaps.size(), pc = Pc;
    f.write(reinterpret_cast<const char *>(&n_snaps), sizeof(n_snaps));
    f.write(reinterpret_cast<const char *>(&pc), sizeof(pc));

    auto net = std::make_unique<NandaNet>();
    std::vector<float> final_lev;
    for (const auto &[step, path] : snaps) {
        net->Load(path.string());
        const auto lev = ComputeJacobianNorms<NandaNet, RefB>(*net, X_ref);
        f.write(reinterpret_cast<const char *>(&step), sizeof(step));
        f.write(reinterpret_cast<const char *>(lev.data()),
                static_cast<std::streamsize>(lev.size() * sizeof(float)));
        float mx = 0.f, mean = 0.f;
        for (float v : lev) { mx = std::max(mx, v); mean += v; }
        std::cout << "step " << step << "  mean|J_i|=" << mean / Pc
                  << "  max|J_i|=" << mx << "\n";
        final_lev = lev;
    }
    std::cout << "wrote " << ckpt_dir << "/leverage_realized.bin\n";

    // ── Pass 2: full Jacobian columns for the top-TopK params ────────────────
    // The per-param |J_i| is the DIAGONAL of the Fisher metric. The off-diagonal
    // — how params share output directions — needs the columns themselves.
    // Dump the mean-J columns of the top params (by final realized leverage) at
    // every snapshot; downstream computes their pairwise-cosine collaboration
    // matrix over time.
    constexpr size_t TopK = 50;
    std::vector<size_t> top_idx(Pc);
    for (size_t i = 0; i < Pc; ++i) top_idx[i] = i;
    std::partial_sort(top_idx.begin(), top_idx.begin() + TopK, top_idx.end(),
                      [&](size_t a, size_t b) { return final_lev[a] > final_lev[b]; });
    top_idx.resize(TopK);

    {
        std::ofstream tf(ckpt_dir + "/top_params.txt");
        for (const size_t i : top_idx) tf << i << "\n";
    }

    std::ofstream jc(ckpt_dir + "/j_columns.bin", std::ios::binary);
    const uint64_t topk = TopK, outsize = OutputT::Size;
    jc.write(reinterpret_cast<const char *>(&n_snaps), sizeof(n_snaps));
    jc.write(reinterpret_cast<const char *>(&topk), sizeof(topk));
    jc.write(reinterpret_cast<const char *>(&outsize), sizeof(outsize));
    for (const auto &[step, path] : snaps) {
        net->Load(path.string());
        const auto J = ComputeJacobian<NandaNet, RefB>(*net, X_ref); // mean-J columns, all params
        jc.write(reinterpret_cast<const char *>(&step), sizeof(step));
        for (const size_t i : top_idx)
            for (size_t j = 0; j < OutputT::Size; ++j) {
                const float v = J[i].flat(j);
                jc.write(reinterpret_cast<const char *>(&v), sizeof(v));
            }
        std::cout << "J-columns @ step " << step << "\n";
    }
    std::cout << "wrote " << ckpt_dir << "/j_columns.bin\n";
    return 0;
}
