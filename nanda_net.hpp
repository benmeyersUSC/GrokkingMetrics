// nanda_net.hpp
// Shared definition of the Nanda-style mod-113 transformer: task constants,
// runner-local blocks, network type, loss, dataset, and batch building.
// Included by nanda_grokking.cpp (training runner) and nanda_leverage.cpp
// (structural-potential / realized-leverage analysis).

#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include "deps/TTTN/src/TTTN.hpp"

using namespace TTTN;

// ─── Task / architecture constants ───────────────────────────────────────────

static constexpr size_t P             = 113;              // modulus
static constexpr size_t Vocab         = P + 1;            // digits 0..P-1 plus EQ token (=P)
static constexpr size_t SeqLen        = 3;                // [a, b, EQ]
static constexpr size_t ReadPos       = 2;                // predict at EQ position
static constexpr size_t EmbDim        = 128;
static constexpr size_t Heads         = 4;
static constexpr size_t FFNHidden     = 512;
static constexpr size_t Total         = P * P;            // 12769
static constexpr size_t TrainSize     = 3840;             // ~30% of 12769
static constexpr size_t ValSize       = Total - TrainSize;
static constexpr size_t Batch         = 128;              // training / eval batch
static constexpr size_t JacobianBatch = 4;                // fixed reference samples for J
static constexpr uint32_t Seed        = 0xB1B1B1B1u;

// ─── Bespoke blocks (experiment-local — not part of TTTN core) ───────────────

// Adds a learned positional embedding of shape Tensor<SeqLen, EmbDim>.
template<size_t SL, size_t ED>
class PositionalEmbeddingBlock {
    Param<Tensor<SL, ED>> pos_;
public:
    using InputTensor  = Tensor<SL, ED>;
    using OutputTensor = Tensor<SL, ED>;
    template<size_t> using TrainingCache = std::tuple<>;

    auto all_params()       { return std::tie(pos_); }
    auto all_params() const { return std::tie(pos_); }

    PositionalEmbeddingBlock() {
        // Small-magnitude init for positional embeddings.
        // Honors TTTN_INIT_SEED (same convention as XavierInitMD) for reproducible inits.
        static std::mt19937 rng{[]() -> std::mt19937::result_type {
            if (const char *s = std::getenv("TTTN_INIT_SEED"))
                return static_cast<std::mt19937::result_type>(std::strtoul(s, nullptr, 10) ^ 0x9E3779B9u);
            return std::random_device{}();
        }()};
        std::uniform_real_distribution<float> dist{-0.02f, 0.02f};
        for (size_t i = 0; i < Tensor<SL, ED>::Size; ++i) pos_.value.flat(i) = dist(rng);
    }

    template<size_t B>
    auto Forward(const typename PrependBatch<B, InputTensor>::type &X) const
        -> typename PrependBatch<B, OutputTensor>::type {
        typename PrependBatch<B, OutputTensor>::type Y = X;
        for (size_t bi = 0; bi < B; ++bi)
            for (size_t i = 0; i < Tensor<SL, ED>::Size; ++i)
                Y.flat(bi * SL * ED + i) += pos_.value.flat(i);
        return Y;
    }

    template<size_t B>
    auto Forward(const typename PrependBatch<B, InputTensor>::type &X,
                 TrainingCache<B> &) const
        -> typename PrependBatch<B, OutputTensor>::type { return Forward<B>(X); }

    template<size_t B>
    auto Backward(const typename PrependBatch<B, OutputTensor>::type &dY,
                  const typename PrependBatch<B, OutputTensor>::type & /*a*/,
                  const typename PrependBatch<B, InputTensor>::type  & /*a_prev*/,
                  const TrainingCache<B> &)
        -> typename PrependBatch<B, InputTensor>::type {
        // dpos = mean_batch(dY) — batched backward normalises by Batch inside the trainer,
        // but here BackwardRange scales by 1/B already for grad accumulation; safe to sum.
        for (size_t bi = 0; bi < B; ++bi)
            for (size_t i = 0; i < Tensor<SL, ED>::Size; ++i)
                pos_.grad.flat(i) += dY.flat(bi * SL * ED + i) / static_cast<float>(B);
        return dY; // pass-through gradient
    }
};

// Extracts row `Pos` from a Tensor<SL, ED>, returning Tensor<ED>.
template<size_t SL, size_t ED, size_t Pos>
class SelectPositionBlock {
    static_assert(Pos < SL, "SelectPositionBlock Pos must be < SL");
public:
    using InputTensor  = Tensor<SL, ED>;
    using OutputTensor = Tensor<ED>;
    template<size_t> using TrainingCache = std::tuple<>;

    auto all_params()       { return std::tuple<>{}; }
    auto all_params() const { return std::tuple<>{}; }

    template<size_t B>
    auto Forward(const typename PrependBatch<B, InputTensor>::type &X) const
        -> typename PrependBatch<B, OutputTensor>::type {
        typename PrependBatch<B, OutputTensor>::type Y{};
        for (size_t bi = 0; bi < B; ++bi)
            for (size_t k = 0; k < ED; ++k)
                Y.flat(bi * ED + k) = X.flat(bi * SL * ED + Pos * ED + k);
        return Y;
    }

    template<size_t B>
    auto Forward(const typename PrependBatch<B, InputTensor>::type &X,
                 TrainingCache<B> &) const
        -> typename PrependBatch<B, OutputTensor>::type { return Forward<B>(X); }

    template<size_t B>
    auto Backward(const typename PrependBatch<B, OutputTensor>::type &dY,
                  const typename PrependBatch<B, OutputTensor>::type & /*a*/,
                  const typename PrependBatch<B, InputTensor>::type  & /*a_prev*/,
                  const TrainingCache<B> &)
        -> typename PrependBatch<B, InputTensor>::type {
        typename PrependBatch<B, InputTensor>::type dX{};
        for (size_t bi = 0; bi < B; ++bi)
            for (size_t k = 0; k < ED; ++k)
                dX.flat(bi * SL * ED + Pos * ED + k) = dY.flat(bi * ED + k);
        return dX;
    }
};

// ─── Network type ────────────────────────────────────────────────────────────

// Layer sequence:
//   0: MapDenseMDBlock<Tensor<3,Vocab>, Tensor<EmbDim>, N_map=1> — token embedding per position
//   1: PositionalEmbeddingBlock<3, EmbDim>
//   2: TransformerBlock<Tensor<3,EmbDim>, Heads, FFNHidden, PreNorm=true, Masked=false>
//   3: SelectPositionBlock<3, EmbDim, 2>
//   4: DenseMDBlock<Tensor<EmbDim>, Tensor<P>, Linear>                — unembed → logits
using EmbedBlock   = MapDenseMDBlock<Tensor<SeqLen, Vocab>, Tensor<EmbDim>, 1, Linear>;
using PosEmbBlock  = PositionalEmbeddingBlock<SeqLen, EmbDim>;
using TxfBlock     = TransformerBlock<Tensor<SeqLen, EmbDim>, Heads, FFNHidden, true, false>;
using SelectBlock  = SelectPositionBlock<SeqLen, EmbDim, ReadPos>;
using UnembedBlock = DenseMDBlock<Tensor<EmbDim>, Tensor<P>, Linear>;

using NandaNet = TrainableTensorNetwork<EmbedBlock, PosEmbBlock, TxfBlock, SelectBlock, UnembedBlock>;

// ─── Split variant: the transformer block's two residual sublayers as separate
// top-level blocks, exposing the post-attention / pre-MLP boundary to the lens.
// TransformerBlock is internally BlockSequence<Residual(LN→MHA), Residual(LN→FFN)>,
// so this is the SAME parameters in the SAME all_params() order — snap_*.bin
// files load into either net and the forward functions are identical.
//   boundaries: 0 in · 1 embed-out · 2 posemb-out · 3 attn-out · 4 mlp-out
//               (== NandaNet's txf-out) · 5 readout · 6 logits
using AttnHalfBlock = ResidualBlock<BlockSequence<
    LayerNormBlock<SeqLen, EmbDim>,
    MultiHeadAttentionBlock<SeqLen, Heads, false, EmbDim>>>;
using FFNHalfBlock = ResidualBlock<BlockSequence<
    LayerNormBlock<SeqLen, EmbDim>,
    MapDenseMDBlock<Tensor<SeqLen, EmbDim>, Tensor<FFNHidden>, 1, ReLU>,
    MapDenseMDBlock<Tensor<SeqLen, FFNHidden>, Tensor<EmbDim>, 1>>>;

using NandaNetSplit = TrainableTensorNetwork<EmbedBlock, PosEmbBlock,
    AttnHalfBlock, FFNHalfBlock, SelectBlock, UnembedBlock>;

static_assert(NandaNetSplit::TotalParamCount == NandaNet::TotalParamCount,
              "split net must be weight-compatible with NandaNet");

using InputT     = NandaNet::InputTensor;       // Tensor<3, Vocab>
using OutputT    = NandaNet::OutputTensor;      // Tensor<P>
using BatchInpT  = PrependBatch<Batch, InputT>::type;
using BatchOutT  = PrependBatch<Batch, OutputT>::type;
using RefInpT    = PrependBatch<JacobianBatch, InputT>::type;

// ─── Loss: Softmax + CE on the P classes ─────────────────────────────────────

struct SoftmaxCEL {
    template<size_t... Dims>
    static Tensor<> Loss(const Tensor<Dims...> &logits, const Tensor<Dims...> &target) {
        const auto probs = Softmax<Tensor<Dims...>::Rank - 1>(logits);
        return CEL::Loss(probs, target);
    }
    template<size_t... Dims>
    static Tensor<Dims...> Grad(const Tensor<Dims...> &logits, const Tensor<Dims...> &target) {
        return Softmax<Tensor<Dims...>::Rank - 1>(logits) - target;
    }
};

// ─── Dataset ─────────────────────────────────────────────────────────────────

struct Dataset {
    // Every (a, b) pair, in canonical order flattened as i = a*P + b.
    // train_indices / val_indices are seeded shuffles.
    std::vector<uint32_t> train_indices;
    std::vector<uint32_t> val_indices;

    void Build(uint32_t seed) {
        std::vector<uint32_t> perm(Total);
        for (uint32_t i = 0; i < Total; ++i) perm[i] = i;
        std::mt19937 rng{seed};
        std::shuffle(perm.begin(), perm.end(), rng);
        train_indices.assign(perm.begin(), perm.begin() + TrainSize);
        val_indices  .assign(perm.begin() + TrainSize, perm.end());
    }

    static void EncodeSample(uint32_t idx, InputT &X, OutputT &Y) {
        const uint32_t a = idx / P;
        const uint32_t b = idx % P;
        const uint32_t ans = (a + b) % P;
        // X: one-hot at [0, a], [1, b], [2, P]
        X.fill(0.f);
        X.flat(0 * Vocab + a) = 1.f;
        X.flat(1 * Vocab + b) = 1.f;
        X.flat(2 * Vocab + P) = 1.f;
        // Y: one-hot at ans
        Y.fill(0.f);
        Y.flat(ans) = 1.f;
    }
};

// Build a Batch from a slice of indices.
template<size_t B>
void BuildBatch(const std::vector<uint32_t> &idxs, size_t start,
                typename PrependBatch<B, InputT>::type &X,
                typename PrependBatch<B, OutputT>::type &Y)
{
    for (size_t b = 0; b < B; ++b) {
        InputT Xb; OutputT Yb;
        Dataset::EncodeSample(idxs[(start + b) % idxs.size()], Xb, Yb);
        for (size_t k = 0; k < InputT::Size;  ++k) X.flat(b * InputT::Size  + k) = Xb.flat(k);
        for (size_t k = 0; k < OutputT::Size; ++k) Y.flat(b * OutputT::Size + k) = Yb.flat(k);
    }
}
