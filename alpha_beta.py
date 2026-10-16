#!/usr/bin/env python3
"""alpha_beta.py — split the teacher-forced hair SNR into signal and noise:
    offset^2 ~ t^(-2 alpha)   (prompt-induced shift, consistent across continuations)
    noise^2  ~ t^(-2 beta)    (spread of hidden states across continuations)
    SNR      ~ t^(-2 (alpha - beta))
Usage:  python alpha_beta.py run1 run1_rand
        python alpha_beta.py pythia/step*          (shell expands; checkpoints are sorted by step)
"""
import json, re, sys
import numpy as np

def key(d):
    m = re.search(r"step(\d+)", d)
    return (0, int(m.group(1))) if m else (1, d)

for run in sorted(sys.argv[1:], key=key):
    try:
        meta = json.load(open(f"{run}/meta.json")); Z = np.load(f"{run}/data.npz")
    except FileNotFoundError:
        print(f"{run}: missing files, skipped"); continue
    lt = np.log10(np.array(meta["windows"], float))
    for li, l in enumerate(meta["layers"]):
        sig, noi = [], []
        for ax in meta["axes"]:
            for si in range(meta["n_scenes"]):
                A = Z[f"tfA|{si}|{ax}"][:, :, li].astype(float)
                B = Z[f"tfB|{si}|{ax}"][:, :, li].astype(float)
                var = 0.5 * (A.var(0) + B.var(0))                    # (n_win, d)
                s0 = np.sqrt(var[0]) + 1e-8                           # fixed per-dim scale (first window)
                sig.append((((B - A).mean(0) / s0) ** 2).mean(-1))
                noi.append((var / s0 ** 2).mean(-1))
        a2 = np.polyfit(lt, np.log10(np.mean(sig, 0)), 1)[0]
        b2 = np.polyfit(lt, np.log10(np.mean(noi, 0)), 1)[0]
        print(f"{run:22s} L{l:<3d} alpha={-a2/2:5.2f}   beta={-b2/2:5.2f}   SNR slope {a2-b2:+.2f}")
    print()
