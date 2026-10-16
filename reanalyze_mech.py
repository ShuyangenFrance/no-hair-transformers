#!/usr/bin/env python3
"""reanalyze_mech.py — recompute the real_mech.py table from mech.npz with centre-labelled windows.
  python reanalyze_mech.py mech/p160_step0 mech/p160_step512 mech/p160_final
"""
import sys
import numpy as np

WIN = 16
for d in sys.argv[1:]:
    Z = np.load(f"{d}/mech.npz")
    W = np.array(Z["windows"], float); lt = np.log10(W - (WIN - 1) / 2)
    layers = [int(x) for x in Z["layers"]]
    alpha = lambda c: -np.polyfit(lt, np.log10(np.maximum(c, 1e-30)), 1)[0] / 2
    print(f"\n{d}  (window width {WIN}, centre labels; four axes pooled)")
    print(f"{'layer':>6s}{'measured':>10s}{'paths':>8s}{'direct':>8s}{'relay':>8s}{'share':>8s}{'uniform':>9s}{'per-head':>10s}")
    for li, l in enumerate(layers):
        pm = Z["pth"][:, :, li].mean((0, 1))
        share = pm[1, -1] / (pm[0, -1] + pm[1, -1])
        row = [alpha(Z[f"sig_{c}"][:, :, li].mean((0, 1))) for c in ("none",)] + \
              [alpha(pm[2]), alpha(pm[0]), alpha(pm[1]), share] + \
              [alpha(Z[f"sig_{c}"][:, :, li].mean((0, 1))) for c in ("uniform", "perhead")]
        print(f"{'L' + str(l):>6s}" + "".join(f"{v:>{w}.2f}" for v, w in zip(row, [10, 8, 8, 8, 8, 9, 10])))
