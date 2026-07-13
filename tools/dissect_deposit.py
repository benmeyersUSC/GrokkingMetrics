#!/usr/bin/env python3
"""Dissect the attn-out "="-slot deposit: why do antipodal same-class latents
still read as the same disposition?

Findings this script reproduces (final v3 snapshot, sum-0 class):
  1. The chord-midpoint geometry is real: centered =-slot STATES within a
     sum-class swing to near-antipodal (state cos p10 ~ -0.56), exactly as
     (mean of two circle points) = cosD * e(sigma) predicts, sigma defined
     only mod pi.
  2. The trained attention pattern at "=" is strongly content-dependent
     (head-1 a/b weights swing +-0.32 across the class): the deposit carries
     PRODUCT terms - attention already does part of the multiplication
     (value-composition), and products are class-pure (no sigma branch).
  3. The mean lens at attn-out reads almost only that product channel:
     - gate-free read (pure W_U through the residual identity): the
       antipodal pair (1,112)v(56,57) scores NEGATIVE (~ -0.51) - the flip
       is visible to a naive linear read;
     - the lens's FFN path carries ~all the energy and scores +0.74.
     Mechanism: for channels the MLP SQUARES (the additive midpoint
     features), the per-context Jacobian is ~ d(x^2)/dx = 2x - it spins with
     the context and cancels out of E[J] (spin-and-cancel in Jacobian
     space). ReLU's half-wave blindness is how the even/squaring features
     are built; even features have odd Jacobians; odd Jacobians average to
     zero. The mean lens is structurally deaf to whatever is about to be
     squared, and keen on what is already linear (attention's products).

Usage: python3 tools/dissect_deposit.py [ckpt_dir] [sum_class=0]
"""
import struct
import sys
from pathlib import Path

import numpy as np

P, V, SEQ, E, H, HD = 113, 114, 3, 128, 4, 32
KS = [5, 8, 17, 49]


def main() -> int:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints_nanda_grokking_v3")
    sclass = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    flat = np.fromfile(ckpt / "snaps/snap_9999.bin", dtype=np.float32)
    sizes = [14592, 128, 384, 128, 128, 16384, 16384, 16384, 16384,
             128, 128, 65536, 512, 65536, 128, 14464, 113]
    t, o = [], 0
    for s in sizes:
        t.append(flat[o:o + s].astype(np.float64))
        o += s
    embW, embB, pos, ln1g, ln1b, wq, wk, wv, wo = t[:9]
    wu = t[15]
    embW = embW.reshape(E, V)
    pos = pos.reshape(SEQ, E)
    wq = wq.reshape(H, HD, E); wk = wk.reshape(H, HD, E)
    wv = wv.reshape(H, HD, E); wo = wo.reshape(E, H, HD)
    wuP = wu.reshape(P, E)

    def lnf(x, g, b):
        m = x.mean(-1, keepdims=True)
        c = x - m
        return c / np.sqrt((c * c).mean(-1, keepdims=True) + 1e-8) * g + b

    def attn_eq(a, b):
        """=-slot row of attn-out + the '='-position attention pattern."""
        h = np.stack([embW[:, a], embW[:, b], embW[:, P]]) + embB + pos
        u = lnf(h, ln1g, ln1b)
        q = np.einsum('se,hde->shd', u, wq)
        k = np.einsum('se,hde->shd', u, wk)
        v = np.einsum('se,hde->shd', u, wv)
        sc = np.einsum('qhd,khd->hqk', q, k) / np.sqrt(HD)
        aw = np.exp(sc - sc.max(2, keepdims=True))
        aw /= aw.sum(2, keepdims=True)
        att = np.einsum('hqk,khd->qhd', aw, v)
        full = h + np.einsum('qhd,ehd->qe', att, wo)
        return full.ravel(), full[2], aw[:, 2, :]

    # split-lens (attn-out) + dataset mean from the dump
    buf = (ckpt / "jlens/jlens_split_9999.bin").read_bytes()
    off = 24
    _, tg, ac = struct.unpack_from("<QQQ", buf, off)
    off += 24
    LA = np.frombuffer(buf, np.float32, tg * ac, off).reshape(tg, ac).astype(np.float64)
    off += 4 * (tg * ac + tg + 2)
    n = struct.unpack_from("<Q", buf, off)[0]
    off += 8 + 8 * n
    acts = np.frombuffer(buf, np.float32, 2 * n * ac, off).reshape(2 * n, ac).astype(np.float64)
    mean_hA = acts.mean(0)
    mean_eq = mean_hA.reshape(SEQ, E)[2]
    dbar = LA @ mean_hA
    LAeq = LA.reshape(P, SEQ, E)[:, 2, :]

    # embedding frequency-plane bases per key k (orthonormalized)
    tok = np.arange(P)
    planes = {}
    Ecen = embW[:, :P].T - embW[:, :P].T.mean(0)
    for k in KS:
        C = np.exp(-2j * np.pi * k * tok / P) @ Ecen
        uk, vk = np.real(C), -np.imag(C)
        uk /= np.linalg.norm(uk)
        vk -= uk * (uk @ vk)
        vk /= np.linalg.norm(vk)
        planes[k] = (uk, vk)

    A = np.arange(P)
    B = (sclass - A) % P
    HF, US, AW = [], [], []
    for a, b in zip(A, B):
        hf, ue, aw = attn_eq(int(a), int(b))
        HF.append(hf); US.append(ue); AW.append(aw)
    HF, US, AW = np.stack(HF), np.stack(US), np.stack(AW)
    Uc = US - mean_eq
    DV = HF @ LA.T - dbar

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))

    i1, i2 = 1, 56  # the antipodal exemplars for sclass=0: (1,112) vs (56,57)
    print(f"=== sum-{sclass} class · pair ({i1},{int(B[i1])}) vs ({i2},{int(B[i2])}) ===")
    print(f"centered =-slot STATE cos:        {cos(Uc[i1], Uc[i2]):+.3f}")
    print(f"centered DISPOSITION cos:         {cos(DV[i1], DV[i2]):+.3f}")

    print("\nembed-plane content of the deposit (the additive/midpoint channel):")
    add = np.zeros((2, E))
    for k in KS:
        uk, vk = planes[k]
        p1 = np.array([Uc[i1] @ uk, Uc[i1] @ vk])
        p2 = np.array([Uc[i2] @ uk, Uc[i2] @ vk])
        add[0] += p1[0] * uk + p1[1] * vk
        add[1] += p2[0] * uk + p2[1] * vk
        print(f"  k={k:2d}: plane-cos between the pair = "
              f"{p1 @ p2 / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-30):+.2f}")
    res1, res2 = Uc[i1] - add[0], Uc[i2] - add[1]
    print(f"in-plane parts cos: {cos(add[0], add[1]):+.3f} · residual parts cos: {cos(res1, res2):+.3f}")
    d_add = LAeq @ add.T
    print(f"disposition energy from in-plane {np.linalg.norm(d_add[:, 0])**2:.1f} "
          f"vs residual {np.linalg.norm(LAeq @ res1)**2:.1f}")

    print("\ngate-free vs FFN-path reads (the ReLU question):")
    Ddir = Uc @ wuP.T
    Dffn = Uc @ (LAeq - wuP).T
    print(f"  pure W_U (residual identity, no ReLU): {cos(Ddir[i1], Ddir[i2]):+.3f}"
          f"   energy {np.linalg.norm(Ddir[i1])**2:.1f}")
    print(f"  lens FFN path (W_U·E[J_FFN]):          {cos(Dffn[i1], Dffn[i2]):+.3f}"
          f"   energy {np.linalg.norm(Dffn[i1])**2:.1f}")

    print("\nattention-at-'=' content dependence (std of per-head a/b/= weights):")
    for h in range(H):
        print(f"  head {h}: std={AW[:, h, :].std(0).round(3)}  mean={AW[:, h, :].mean(0).round(3)}")

    rng = np.random.default_rng(3)
    sc_, dc_ = [], []
    for _ in range(300):
        i, j = rng.integers(0, P, 2)
        if i != j:
            sc_.append(cos(Uc[i], Uc[j]))
            dc_.append(cos(DV[i], DV[j]))
    print(f"\nclass-wide =-slot state cos:  p10={np.percentile(sc_, 10):+.2f}"
          f" med={np.median(sc_):+.2f} p90={np.percentile(sc_, 90):+.2f}")
    print(f"class-wide disposition cos:   p10={np.percentile(dc_, 10):+.2f}"
          f" med={np.median(dc_):+.2f} p90={np.percentile(dc_, 90):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
