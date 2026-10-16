#!/usr/bin/env python3
"""
fit_gains.py — fit the gains g_1, g_2, ... of the linear relay model (Proposition prop:gains) to the measured
offset curves of an untrained model, layer by layer, using the exact recursion (no continuum approximation).

Model:  delta_l(i) = delta_{l-1}(i) + g_l/(P+i) * (P + sum_{j<=i} delta_{l-1}(j)),  delta_0 = 0,
measured signal S_l(t) = A_l^2 * (window mean of delta_l)^2, with a free amplitude A_l per layer (the
per-coordinate normalisation of S is layer dependent). Amplitudes are profiled out in log space, so every
layer is fitted on the SHAPE of its curve only.

  1. layer 1 fixes P (delta_1 is proportional to P/(P+i) whatever g_1);
  2. g_1, g_2, g_3 are fitted jointly on layers 2-3 (layer 2 alone constrains only g1 g2/(g1+g2));
  3. for l >= 4, g_l is fitted ALONE on layer l, with g_1..g_{l-1} fixed: one shape parameter predicts the
     whole curve of that layer;
  4. a joint fit of all gains on layers 2..L is reported for comparison.
Quality: R^2 of log S (dominated by the overall slope, so close to 1 for any decent fit) and, more
demanding, the largest deviation between measured and predicted local exponents over all octaves.

  python fit_gains.py real/p1b_step0_all
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

WIN = 16


def relay(gains, P, T):
    """exact linear relay recursion; returns delta[l][i] for l = 1..len(gains), i = 0..T (i=0 unused)."""
    i = np.arange(T + 1, dtype=float)
    d = np.zeros(T + 1)
    out = []
    for g in gains:
        S = np.cumsum(d)                     # sum_{j<=i} d(j), d(0)=0
        d = d + g / (P + i) * (P + S)
        d[0] = 0.0
        out.append(d.copy())
    return out


def window_means(d, W):
    return np.array([d[t - WIN + 1:t + 1].mean() for t in W])


def shape_resid(logS, logm):
    """residual of log S after profiling the amplitude (additive constant in log space)."""
    r = logS - logm
    return r - r.mean()


def local(logcurve, lt):
    return -np.diff(logcurve) / np.diff(lt) / 2


def main(d):
    Z = np.load(f"{d}/real.npz")
    m = int(Z["m"])
    sig = Z["sig"] - Z["flo"] * m / (m - 1)               # unbiased signal
    W = [int(x) for x in Z["windows"]]
    layers = [int(x) for x in Z["layers"]]
    if layers != list(range(1, len(layers) + 1)):
        sys.exit(f"need every layer from 1 on, got {layers}: rerun real_run.py with --layers 1,2,...,L-1")
    T = max(W)
    tc = np.array(W, float) - (WIN - 1) / 2
    lt = np.log(tc)
    logS = {l: np.log(sig[:, :, li].mean((0, 1))) for li, l in enumerate(layers)}
    L = len(layers)

    def model_log(gains, P, l):
        return 2 * np.log(window_means(relay(gains, P, T)[l - 1], W))

    # 1. P from layer 1
    resP = minimize_scalar(lambda P: np.sum(shape_resid(logS[1], model_log([1.0], P, 1)) ** 2),
                           bounds=(1, 300), method="bounded")
    P = resP.x

    # 2. g1, g2, g3 on layers 2-3
    def r23(lg):
        g = np.exp(lg)
        return np.concatenate([shape_resid(logS[l], model_log(g[:l], P, l)) for l in (2, 3)])
    starts = [[0.3, 0.3, 0.3], [0.1, 1.0, 0.3], [1.0, 0.1, 0.3], [0.05, 2.0, 0.2], [0.5, 0.5, 0.05], [2.0, 0.2, 0.5]]
    sols = [least_squares(r23, np.log(s0)) for s0 in starts]
    print("layers 2-3 joint fit of (g1, g2, g3) from several starting points:")
    for s0, so in zip(starts, sols):
        print(f"   start {s0} -> {np.round(np.exp(so.x), 3)}   cost {so.cost:.3e}")
    sol = min(sols, key=lambda so: so.cost)
    g_seq = list(np.exp(sol.x))

    # 3. one gain per layer from l = 4 on
    for l in range(4, L + 1):
        f = lambda lg: np.sum(shape_resid(logS[l], model_log(g_seq + [np.exp(lg)], P, l)) ** 2)
        r = minimize_scalar(f, bounds=(np.log(1e-4), np.log(10)), method="bounded")
        g_seq.append(float(np.exp(r.x)))

    # 4. joint fit of all gains on layers 2..L
    def rall(lg):
        g = np.exp(lg)
        return np.concatenate([shape_resid(logS[l], model_log(g[:l], P, l)) for l in range(2, L + 1)])
    sol_j = least_squares(rall, np.log(g_seq))
    g_joint = list(np.exp(sol_j.x))

    def quality(gains, l):
        lm = model_log(gains[:l], P, l)
        res = shape_resid(logS[l], lm)
        r2 = 1 - np.sum(res ** 2) / np.sum((logS[l] - logS[l].mean()) ** 2)
        dalpha = np.max(np.abs(local(logS[l], lt) - local(lm, lt)))
        return r2, dalpha

    print(f"\n{d}: P fitted from layer 1 = {P:.1f} (mean prompt length {float(Z['plen'][:, :, 0].mean()):.1f})")
    q1 = quality([1.0], 1)
    print(f"layer 1 (shape fixed by P alone): R2 = {q1[0]:.4f}, max |delta alpha_loc| = {q1[1]:.3f}")
    print(f"\n{'layer':>6s}{'g_l seq':>9s}{'R2':>8s}{'max dα':>8s}{'G_l seq':>9s}   {'g_l joint':>9s}{'R2':>8s}{'max dα':>8s}")
    out = []
    for l in range(1, L + 1):
        qs = quality(g_seq, l) if l >= 2 else (np.nan, np.nan)
        qj = quality(g_joint, l) if l >= 2 else (np.nan, np.nan)
        Gs = sum(g_seq[:l])
        print(f"{l:>6d}{g_seq[l - 1]:>9.3f}{qs[0]:>8.4f}{qs[1]:>8.3f}{Gs:>9.3f}   {g_joint[l - 1]:>9.3f}{qj[0]:>8.4f}{qj[1]:>8.3f}")
        out.append(dict(l=l, g_seq=g_seq[l - 1], r2_seq=qs[0], dalpha_seq=qs[1], G_seq=Gs,
                        g_joint=g_joint[l - 1], r2_joint=qj[0], dalpha_joint=qj[1]))
    ls = np.arange(4, L + 1)
    gam, c = np.polyfit(np.log(ls), np.log(np.array(g_seq[3:])), 1)
    print(f"\nsequential gains for l >= 4:  g_l ~ l^{gam:.2f}   (layers 1-3 are fitted jointly and not included)")
    print("R2 is of log S after removing the amplitude; 'max dα' is the largest gap between measured and predicted")
    print("local exponents over all octaves, the more demanding check.")
    json.dump(dict(P=P, gains=out, gamma=gam), open(f"{d}/gain_fit.json", "w"), indent=1, default=float)
    print("saved", f"{d}/gain_fit.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "real/p1b_step0_all")
