// nanda_grokking.cpp
// Nanda-style modular arithmetic grokking runner with output-space trajectory instrumentation.
//
// Task: (a + b) mod P for P = 113, sequence [a, b, EQ] → predict answer.
// Architecture: 1-layer pre-norm transformer, EmbDim=128, 4 heads, FFN=512.
// Training: full-batch on 30% of P^2 examples, AdamW with weight decay = 1.0.
// Per-step instrumentation: Jacobian of output wrt each parameter (as an OutputTensor per param),
// contributed to a running output-space GROSS/NET/accord accumulator.
//
// Emits:
//   metrics.csv                — per-log-interval scalar signals
//   output_gross_net.bin       — per-checkpoint per-param output-space accumulators
//   ckpt.bin                   — weights + Adam state + Param::metrics (resumable)
//
// Build: `./build.sh` compiles main.cpp; run `g++ -std=c++20 -O3 ... nanda_grokking.cpp` separately.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "nanda_net.hpp"

// ─── Run config ──────────────────────────────────────────────────────────────

static constexpr float LR             = 1e-3f;
static constexpr float WD             = 1.0f;             // AdamW decoupled decay
static constexpr float Beta1          = 0.9f;
static constexpr float Beta2          = 0.98f;

static constexpr size_t TargetSteps   = 10000;
static constexpr size_t LogEvery      = 10;               // steps between CSV rows
static constexpr size_t JacobianEvery = 1;                // compute J every N steps
static constexpr size_t CkptEvery     = 500;              // steps between binary dumps
static constexpr size_t SnapEvery     = 250;              // steps between retained crystallization snapshots

// Rolling accord windows (steps).
static constexpr size_t W1 = 100;
static constexpr size_t W2 = 500;
static constexpr size_t W3 = 2000;

// Default output dir; override with argv[1]. v3 default is distinct so a
// stale launch can never clobber an earlier run's artifacts.
static constexpr const char* DefaultCkptDir = "checkpoints_nanda_grokking_v3";

// ─── Validation eval (accuracy + mean CE loss) ───────────────────────────────

struct ValStats { float loss; float acc; };

// Evaluate loss + accuracy over the first `count` indices of `idxs` (chunked by Batch).
ValStats EvalSet(NandaNet &net, const std::vector<uint32_t> &idxs, const size_t count) {
    float loss_sum = 0.f;
    size_t correct = 0;
    size_t seen = 0;
    for (size_t start = 0; start < count; start += Batch) {
        BatchInpT X; BatchOutT Y;
        const size_t chunk = std::min(Batch, count - start);
        for (size_t b = 0; b < Batch; ++b) {
            const size_t src = start + std::min(b, chunk - 1);
            InputT Xb; OutputT Yb;
            Dataset::EncodeSample(idxs[src], Xb, Yb);
            for (size_t k = 0; k < InputT::Size;  ++k) X.flat(b * InputT::Size  + k) = Xb.flat(k);
            for (size_t k = 0; k < OutputT::Size; ++k) Y.flat(b * OutputT::Size + k) = Yb.flat(k);
        }
        const auto logits = net.template Forward<Batch>(X);
        for (size_t b = 0; b < chunk; ++b) {
            OutputT lgt, tgt;
            for (size_t k = 0; k < OutputT::Size; ++k) {
                lgt.flat(k) = logits.flat(b * OutputT::Size + k);
                tgt.flat(k) = Y     .flat(b * OutputT::Size + k);
            }
            const auto probs = Softmax<0>(lgt);
            loss_sum += CEL::Loss(probs, tgt).flat(0);
            const size_t pred = Argmax(lgt);
            const size_t truth = Argmax(tgt);
            if (pred == truth) ++correct;
            ++seen;
        }
    }
    return {loss_sum / std::max<size_t>(seen, 1),
            100.f * static_cast<float>(correct) / std::max<size_t>(seen, 1)};
}

// Softmax+CE loss and gradient over a batch of logits.
float BatchLossAndGrad(const BatchOutT &logits, const BatchOutT &Y, BatchOutT &grad) {
    float loss_sum = 0.f;
    for (size_t b = 0; b < Batch; ++b) {
        OutputT lgt, tgt;
        for (size_t k = 0; k < OutputT::Size; ++k) {
            lgt.flat(k) = logits.flat(b * OutputT::Size + k);
            tgt.flat(k) = Y     .flat(b * OutputT::Size + k);
        }
        const auto probs = Softmax<0>(lgt);
        loss_sum += CEL::Loss(probs, tgt).flat(0);
        const auto g = probs - tgt; // combined softmax + CE grad
        for (size_t k = 0; k < OutputT::Size; ++k)
            grad.flat(b * OutputT::Size + k) = g.flat(k);
    }
    return loss_sum / static_cast<float>(Batch);
}

// Train batch: forward, loss, backward, Adam step. Captures the flat training
// gradient (post-backward, pre-update) into `g_train_out`. Returns mean batch loss.
float TrainStepCapture(NandaNet &net, NetworkTrainer<NandaNet, Batch> &tr,
                       const BatchInpT &X, const BatchOutT &Y, float lr,
                       std::vector<float> &g_train_out)
{
    const auto logits = tr.Forward(X);
    BatchOutT grad{};
    const float loss = BatchLossAndGrad(logits, Y, grad);
    tr.ZeroGrad();
    tr.Backward(grad);
    SnapshotFlatGrads(net, g_train_out);
    tr.Update(lr);
    return loss;
}

// Compute the flat gradient of the loss on a fixed (val) batch at current weights.
// Leaves Param::grad zeroed. Does not touch Adam state or trajectory metrics.
float ValGradCapture(NandaNet &net, NetworkTrainer<NandaNet, Batch> &tr,
                     const BatchInpT &Xv, const BatchOutT &Yv,
                     std::vector<float> &g_val_out)
{
    const auto logits = tr.Forward(Xv);
    BatchOutT grad{};
    const float loss = BatchLossAndGrad(logits, Yv, grad);
    tr.ZeroGrad();
    tr.Backward(grad);
    SnapshotFlatGrads(net, g_val_out);
    tr.ZeroGrad();
    return loss;
}

// ─── Flat-vector loss-space helpers ──────────────────────────────────────────

struct LossWork { float net; float gross; float accord; };

// First-order loss-space work of the step: net = Σ_i Δθ_i g_i (≈ ΔL to first
// order), gross = Σ_i |Δθ_i g_i|. accord = |net|/gross — the fraction of
// per-parameter loss-work that survives cross-parameter cancellation.
LossWork ComputeLossWork(const std::vector<float> &delta, const std::vector<float> &g) {
    float net = 0.f, gross = 0.f;
    for (size_t i = 0; i < delta.size(); ++i) {
        const float w = delta[i] * g[i];
        net   += w;
        gross += std::abs(w);
    }
    return {net, gross, gross > 0.f ? std::abs(net) / gross : 0.f};
}

float CosineFlat(const std::vector<float> &a, const std::vector<float> &b) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    const float d = std::sqrt(na) * std::sqrt(nb);
    return d > 0.f ? dot / d : 0.f;
}

// ─── Crystallization snapshots ───────────────────────────────────────────────
// Retained (never rotated) per-SnapEvery artifacts for post-hoc structure analysis:
//   snaps/snap_<step>.bin — weights only (net.Save format)
//   snaps/acts_<step>.bin — header, embedding weight matrix, then for all P*P
//                           inputs in canonical order (i = a*P + b): the 128-dim
//                           readout activation and the 113-dim logits.

void DumpActs(NandaNet &net, const std::string &path, const uint64_t step) {
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = Total, adim = EmbDim, ldim = P, vocab = Vocab, edim = EmbDim;
    f.write(reinterpret_cast<const char *>(&step),  sizeof(step));
    f.write(reinterpret_cast<const char *>(&n),     sizeof(n));
    f.write(reinterpret_cast<const char *>(&adim),  sizeof(adim));
    f.write(reinterpret_cast<const char *>(&ldim),  sizeof(ldim));
    f.write(reinterpret_cast<const char *>(&vocab), sizeof(vocab));
    f.write(reinterpret_cast<const char *>(&edim),  sizeof(edim));

    // Embedding weight (LearnedContraction layout: Tensor<EmbDim, Vocab>, row-major).
    const auto &embed_W = std::get<0>(net.block<0>().all_params()).value;
    embed_W.Save(f);

    // Readout activations + logits over the full canonical dataset, batched.
    std::vector<float> acts(Total * EmbDim), logits(Total * P);
    for (size_t start = 0; start < Total; start += Batch) {
        const size_t chunk = std::min(Batch, Total - start);
        BatchInpT X;
        for (size_t b = 0; b < Batch; ++b) {
            const size_t idx = start + std::min(b, chunk - 1);
            InputT Xb; OutputT Yb;
            Dataset::EncodeSample(static_cast<uint32_t>(idx), Xb, Yb);
            for (size_t k = 0; k < InputT::Size; ++k) X.flat(b * InputT::Size + k) = Xb.flat(k);
        }
        const auto A = net.template ForwardAll<Batch>(X);
        const auto &readout = A.template get<4>(); // post-SelectPositionBlock: [Batch, EmbDim]
        const auto &lgt     = A.template get<5>(); // logits: [Batch, P]
        for (size_t b = 0; b < chunk; ++b) {
            const size_t idx = start + b;
            for (size_t k = 0; k < EmbDim; ++k)
                acts[idx * EmbDim + k] = readout.flat(b * EmbDim + k);
            for (size_t k = 0; k < P; ++k)
                logits[idx * P + k] = lgt.flat(b * P + k);
        }
    }
    f.write(reinterpret_cast<const char *>(acts.data()),
            static_cast<std::streamsize>(acts.size() * sizeof(float)));
    f.write(reinterpret_cast<const char *>(logits.data()),
            static_cast<std::streamsize>(logits.size() * sizeof(float)));
}

void DumpIndices(const Dataset &ds, const std::string &path) {
    std::ofstream f(path, std::ios::binary);
    const uint64_t ntr = ds.train_indices.size(), nva = ds.val_indices.size();
    f.write(reinterpret_cast<const char *>(&ntr), sizeof(ntr));
    f.write(reinterpret_cast<const char *>(&nva), sizeof(nva));
    f.write(reinterpret_cast<const char *>(ds.train_indices.data()),
            static_cast<std::streamsize>(ntr * sizeof(uint32_t)));
    f.write(reinterpret_cast<const char *>(ds.val_indices.data()),
            static_cast<std::streamsize>(nva * sizeof(uint32_t)));
}

// ─── OutputSpaceTrajectory persistence helpers ───────────────────────────────

// Serialize the accumulator to a fresh binary file (rotates old to .prev).
void DumpTrajectory(const OutputSpaceTrajectory<NandaNet> &traj, const std::string &path) {
    const std::string prev = path + ".prev";
    if (std::filesystem::exists(path)) {
        std::error_code ec;
        std::filesystem::remove(prev, ec);
        std::filesystem::rename(path, prev, ec);
    }
    std::ofstream f(path, std::ios::binary);
    traj.SaveTo(f);
}

bool LoadTrajectory(OutputSpaceTrajectory<NandaNet> &traj, const std::string &path) {
    if (!std::filesystem::exists(path)) return false;
    std::ifstream f(path, std::ios::binary);
    traj.LoadFrom(f);
    return true;
}

// ─── main ────────────────────────────────────────────────────────────────────

// argv: [1] ckpt_dir  [2] data-split seed  [3] jacobian-every  [4] per-param accumulators (0/1)
int main(int argc, char **argv) {
    const std::string ckpt_dir = argc > 1 ? argv[1] : DefaultCkptDir;
    const uint32_t seed_run    = argc > 2 ? static_cast<uint32_t>(std::stoul(argv[2])) : Seed;
    const size_t jac_every     = argc > 3 ? std::stoul(argv[3]) : JacobianEvery;
    const bool per_param_accum = argc > 4 ? (std::string(argv[4]) == "1") : true;
    std::cout << "[cfg] dir=" << ckpt_dir << " seed=" << seed_run
              << " jac_every=" << jac_every << " per_param=" << per_param_accum << "\n";
    std::filesystem::create_directories(ckpt_dir);
    std::filesystem::create_directories(ckpt_dir + "/snaps");
    const std::string ckpt_path     = ckpt_dir + "/ckpt.bin";
    const std::string traj_tr_path  = ckpt_dir + "/output_gross_net_trainref.bin";
    const std::string traj_va_path  = ckpt_dir + "/output_gross_net_valref.bin";
    const std::string csv_path      = ckpt_dir + "/metrics.csv";
    const std::string cursor_path   = ckpt_dir + "/cursor.txt";

    // Dataset — deterministic split per run seed.
    Dataset ds; ds.Build(seed_run);
    DumpIndices(ds, ckpt_dir + "/indices.bin");

    // Two fixed reference batches for the Jacobian, stable across the run:
    //  - train refs: the function is pinned here once memorized — accord measures refinement
    //  - val refs:   where the function actually moves during grokking — the hypothesis lives here
    RefInpT X_ref_tr, X_ref_va;
    {
        typename PrependBatch<JacobianBatch, OutputT>::type Y_unused;
        BuildBatch<JacobianBatch>(ds.train_indices, 0, X_ref_tr, Y_unused);
        BuildBatch<JacobianBatch>(ds.val_indices,   0, X_ref_va, Y_unused);
    }

    // Fixed val batch for the per-step val-loss gradient (g_val).
    BatchInpT X_vgrad; BatchOutT Y_vgrad;
    BuildBatch<Batch>(ds.val_indices, 0, X_vgrad, Y_vgrad);

    // Network + trainer.
    auto net = std::make_unique<NandaNet>();
    NetworkTrainer<NandaNet, Batch> tr(*net);
    tr.adam().beta1 = Beta1;
    tr.adam().beta2 = Beta2;
    tr.adam().wd   = WD;

    // Output-space trajectory accumulators — one per reference batch.
    // Per-param storage (train ref only) is optional: disable for ensemble runs.
    OutputSpaceTrajectory<NandaNet> traj_tr(per_param_accum);
    OutputSpaceTrajectory<NandaNet> traj_va(/*per_param=*/false);

    // Rolling accord windows per ref. Not serialized — warm up over the first W
    // steps after any (re)start. Heap-backed: the W3 ring alone is ~1.8 MB.
    using Traj = OutputSpaceTrajectory<NandaNet>;
    auto roll_tr_w1_p = std::make_unique<Traj::RollingAccord<W1>>();
    auto roll_tr_w2_p = std::make_unique<Traj::RollingAccord<W2>>();
    auto roll_tr_w3_p = std::make_unique<Traj::RollingAccord<W3>>();
    auto roll_va_w1_p = std::make_unique<Traj::RollingAccord<W1>>();
    auto roll_va_w2_p = std::make_unique<Traj::RollingAccord<W2>>();
    auto roll_va_w3_p = std::make_unique<Traj::RollingAccord<W3>>();
    auto &roll_tr_w1 = *roll_tr_w1_p; auto &roll_tr_w2 = *roll_tr_w2_p; auto &roll_tr_w3 = *roll_tr_w3_p;
    auto &roll_va_w1 = *roll_va_w1_p; auto &roll_va_w2 = *roll_va_w2_p; auto &roll_va_w3 = *roll_va_w3_p;

    // Shape channel: rolling accord over the component of each step's val-ref
    // output movement PERPENDICULAR to the current mean logit direction F̂.
    // The parallel component (va_net_par) is the scale/confidence channel;
    // the perpendicular component is direction change — the algorithm forming.
    auto roll_va_shape_p = std::make_unique<Traj::RollingAccord<W2>>();
    auto &roll_va_shape  = *roll_va_shape_p;

    // Cumulative loss-space work (train-grad and val-grad flavors).
    float cum_lnet_tr = 0.f, cum_lgross_tr = 0.f;
    float cum_lnet_va = 0.f, cum_lgross_va = 0.f;

    // Resume support.
    size_t start_step = 0;
    if (std::filesystem::exists(ckpt_path)) {
        std::cout << "[resume] loading " << ckpt_path << "\n";
        tr.LoadTrainingState(ckpt_path);
        LoadTrajectory(traj_tr, traj_tr_path);
        LoadTrajectory(traj_va, traj_va_path);
        std::ifstream cur(cursor_path);
        if (cur) cur >> start_step;
        std::cout << "[resume] resuming at step " << start_step
                  << " (rolling windows + cumulative loss-work restart cold)\n";
    }

    // CSV header (only on fresh start).
    std::ofstream csv;
    if (!std::filesystem::exists(csv_path) || start_step == 0) {
        csv.open(csv_path, std::ios::trunc);
        csv << "step,train_loss,train_acc,val_loss,val_acc,"
               "param_gross,param_net_l2,param_eff,"
               "jtr_cum,jtr_inst,jtr_w100,jtr_w500,jtr_w2000,"
               "jva_cum,jva_inst,jva_w100,jva_w500,jva_w2000,"
               "ltr_net,ltr_accord,ltr_cum_accord,"
               "lva_net,lva_accord,lva_cum_accord,"
               "cos_gtr_gva,val_batch_loss,"
               "jtr_gross_l1,jva_gross_l1,"
               "va_net_par,va_shape_w500\n";
    } else {
        csv.open(csv_path, std::ios::app);
    }

    // Flat buffers: θ snapshot, Δθ, and the two per-step gradients.
    std::vector<float> theta_before, delta, g_train, g_val;
    theta_before.reserve(NandaNet::TotalParamCount);
    delta       .reserve(NandaNet::TotalParamCount);
    g_train     .reserve(NandaNet::TotalParamCount);
    g_val       .reserve(NandaNet::TotalParamCount);

    // Cached Jacobians (reused between recomputations if JacobianEvery > 1).
    std::vector<OutputT> J_tr, J_va;

    // RNG for minibatch sampling.
    std::mt19937 rng{seed_run + 1u};
    std::uniform_int_distribution<size_t> idx_dist{0, TrainSize - 1};

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t step = start_step; step < TargetSteps; ++step) {
        // ── Build minibatch ─────────────────────────────────────────────────
        BatchInpT X; BatchOutT Y;
        // random offset into the (shuffled) train indices — cheap "sampling without replacement"
        const size_t offset = idx_dist(rng);
        BuildBatch<Batch>(ds.train_indices, offset, X, Y);

        // ── Val-loss gradient at current weights (before the step) ──────────
        const float val_batch_loss = ValGradCapture(*net, tr, X_vgrad, Y_vgrad, g_val);

        // ── Snapshot θ_before ───────────────────────────────────────────────
        SnapshotFlatValues(*net, theta_before);

        // ── Training step (captures g_train between backward and update) ────
        TrainStepCapture(*net, tr, X, Y, LR, g_train);

        // ── Compute Δθ ──────────────────────────────────────────────────────
        ComputeFlatDelta(*net, theta_before, delta);

        // ── Jacobians at both reference batches ─────────────────────────────
        const bool recompute_J = (step % jac_every == 0) || J_tr.empty();
        if (recompute_J) {
            J_tr = ComputeJacobian<NandaNet, JacobianBatch>(*net, X_ref_tr);
            J_va = ComputeJacobian<NandaNet, JacobianBatch>(*net, X_ref_va);
        }

        // Mean logit direction at the val refs (post-update weights) — the axis
        // separating the scale channel from the shape channel.
        OutputT F_hat_va{};
        {
            const auto ref_logits = net->template Forward<JacobianBatch>(X_ref_va);
            F_hat_va.fill(0.f);
            for (size_t b = 0; b < JacobianBatch; ++b)
                for (size_t j = 0; j < P; ++j)
                    F_hat_va.flat(j) += ref_logits.flat(b * P + j);
            float norm = 0.f;
            for (size_t j = 0; j < P; ++j) norm += F_hat_va.flat(j) * F_hat_va.flat(j);
            norm = std::sqrt(norm);
            if (norm > 0.f)
                for (size_t j = 0; j < P; ++j) F_hat_va.flat(j) /= norm;
        }

        // ── Output-space accumulation: cumulative + instantaneous + windows ─
        const auto step_tr = traj_tr.Accumulate(delta, J_tr);
        const auto step_va = traj_va.Accumulate(delta, J_va);
        roll_tr_w1.add(step_tr); roll_tr_w2.add(step_tr); roll_tr_w3.add(step_tr);
        roll_va_w1.add(step_va); roll_va_w2.add(step_va); roll_va_w3.add(step_va);

        // Scale/shape split of the val-ref step movement.
        float va_net_par = 0.f;
        for (size_t j = 0; j < P; ++j) va_net_par += step_va.net.flat(j) * F_hat_va.flat(j);
        Traj::StepContribution step_va_shape{};
        step_va_shape.gross = step_va.gross;
        for (size_t j = 0; j < P; ++j)
            step_va_shape.net.flat(j) = step_va.net.flat(j) - va_net_par * F_hat_va.flat(j);
        roll_va_shape.add(step_va_shape);

        // ── Loss-space work (train-grad and val-grad flavors) ───────────────
        const LossWork lw_tr = ComputeLossWork(delta, g_train);
        const LossWork lw_va = ComputeLossWork(delta, g_val);
        cum_lnet_tr   += lw_tr.net;
        cum_lgross_tr += lw_tr.gross;
        cum_lnet_va   += lw_va.net;
        cum_lgross_va += lw_va.gross;
        const float cos_gv = CosineFlat(g_train, g_val);

        // ── Logging ─────────────────────────────────────────────────────────
        if (step % LogEvery == 0 || step + 1 == TargetSteps) {
            const auto trn = EvalSet(*net, ds.train_indices, TrainSize);
            const auto val = EvalSet(*net, ds.val_indices,   ValSize);
            const auto pt  = tr.Trajectory();
            csv << step << ","
                << trn.loss << ","
                << trn.acc  << ","
                << val.loss << ","
                << val.acc  << ","
                << pt.gross_path << ","
                << pt.net_norm   << ","
                << pt.efficiency_ratio << ","
                << traj_tr.AccordRatioL2() << ","
                << step_tr.AccordL2()      << ","
                << roll_tr_w1.accord()     << ","
                << roll_tr_w2.accord()     << ","
                << roll_tr_w3.accord()     << ","
                << traj_va.AccordRatioL2() << ","
                << step_va.AccordL2()      << ","
                << roll_va_w1.accord()     << ","
                << roll_va_w2.accord()     << ","
                << roll_va_w3.accord()     << ","
                << lw_tr.net    << ","
                << lw_tr.accord << ","
                << (cum_lgross_tr > 0.f ? std::abs(cum_lnet_tr) / cum_lgross_tr : 0.f) << ","
                << lw_va.net    << ","
                << lw_va.accord << ","
                << (cum_lgross_va > 0.f ? std::abs(cum_lnet_va) / cum_lgross_va : 0.f) << ","
                << cos_gv << ","
                << val_batch_loss << ","
                << traj_tr.NetworkGrossL1() << ","
                << traj_va.NetworkGrossL1() << ","
                << va_net_par << ","
                << roll_va_shape.accord()
                << "\n";
            csv.flush();

            const auto t_now = std::chrono::high_resolution_clock::now();
            const double s = std::chrono::duration<double>(t_now - t_start).count();
            std::cout << "step " << step
                      << "  train=" << trn.loss << " (" << trn.acc << "%)"
                      << "  val="   << val.loss << " (" << val.acc << "%)"
                      << "  jva_w500=" << roll_va_w2.accord()
                      << "  cos_gv="   << cos_gv
                      << "  wall="  << s << "s\n";
            std::cout.flush();
        }

        // ── Crystallization snapshot (retained, never rotated) ──────────────
        if (step % SnapEvery == 0 || step + 1 == TargetSteps) {
            net->Save(ckpt_dir + "/snaps/snap_" + std::to_string(step) + ".bin");
            DumpActs(*net, ckpt_dir + "/snaps/acts_" + std::to_string(step) + ".bin", step);
        }

        // ── Checkpoint ──────────────────────────────────────────────────────
        if ((step + 1) % CkptEvery == 0 || step + 1 == TargetSteps) {
            const std::string ckpt_new  = ckpt_path + ".new";
            const std::string ckpt_prev = ckpt_path + ".prev";
            tr.SaveTrainingState(ckpt_new);
            std::error_code ec;
            std::filesystem::remove(ckpt_prev, ec);
            if (std::filesystem::exists(ckpt_path))
                std::filesystem::rename(ckpt_path, ckpt_prev, ec);
            std::filesystem::rename(ckpt_new, ckpt_path, ec);
            DumpTrajectory(traj_tr, traj_tr_path);
            DumpTrajectory(traj_va, traj_va_path);
            std::ofstream cur(cursor_path);
            cur << (step + 1) << "\n";
        }
    }

    std::cout << "done. metrics -> " << csv_path << "\n";
    return 0;
}
