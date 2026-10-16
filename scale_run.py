#!/usr/bin/env python3
"""
scale_run.py — step 1 of the scale-up, Mac friendly: teacher-forced alpha on 20 scenes and long windows.

No generation is needed: the shared continuations are human-written text (WikiText-103), which earlier gave
the same alpha as the model's own generations. Each scene uses the same m continuations for the base prompt
and its four variants, so the base forward pass is shared across axes.

Outputs
  alpha / beta per layer with a 95% bootstrap interval over SCENES (the honest uncertainty)
  local alpha per octave (between consecutive windows): theory predicts that for an untrained network
      the local alpha creeps up towards 1 as t grows (log corrections fade), and decreases with depth
  alpha per axis
  raw curves in scale.npz so everything can be re-analysed without re-running the model

  python scale_run.py --model EleutherAI/pythia-160m --revision step0      --out scale/p160_step0
  python scale_run.py --model EleutherAI/pythia-160m --revision step143000 --out scale/p160_final
  python scale_run.py ... --n_scenes 4 --m 8                               (quick check first)
"""
import argparse, json, os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model          # noqa: E402
from tf_controls import human_chunks          # noqa: E402
from scenes20 import SCENES20                 # noqa: E402

AXES = ["detail", "para", "lang", "genre"]


def fit_alpha(curve, lt):
    return -np.polyfit(lt, np.log10(np.maximum(curve, 1e-30)), 1)[0] / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--revision", default=None)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n_scenes", type=int, default=20)
    p.add_argument("--m", type=int, default=16, help="shared continuations per scene")
    p.add_argument("--windows", default="16,32,64,128,256,512,1024,1920")
    p.add_argument("--win", type=int, default=16)
    p.add_argument("--layers", default="", help="default: 1,2,~L/4,~L/2,~3L/4,L-1")
    p.add_argument("--human_file", default="")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyze_only", action="store_true", help="re-analyse an existing scale.npz")
    p.add_argument("--estimator", choices=["unbiased", "plugin"], default="unbiased",
                   help="signal estimator: plug-in (mean offset)^2, or unbiased (subtract Var/(n_c) with ddof=1)")
    p.add_argument("--label", choices=["center", "end"], default="center",
                   help="time assigned to each window: its centre t-(win-1)/2 (default) or its end t")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = [int(x) for x in a.windows.split(",")]
    npz = os.path.join(a.out, "scale.npz")

    if not a.analyze_only:
        la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init,
                                   revision=a.revision, seed=a.seed)
        torch, model, tok, encode, _, _, dev, empty_cache = load_model(la)
        L = model.config.num_hidden_layers
        layers = [int(x) for x in a.layers.split(",")] if a.layers else \
            sorted(set([1, 2] + [max(1, round(L * f)) for f in (0.25, 0.5, 0.75)] + [L - 1]))
        scenes = SCENES20[:a.n_scenes]
        Tmax = max(W)
        maxpos = getattr(model.config, "max_position_embeddings", 10 ** 9)
        Pmax = max(len(encode(sc[k])) for sc in scenes for k in sc)
        if Pmax + Tmax > maxpos:
            sys.exit(f"prompt ({Pmax}) + longest window ({Tmax}) exceeds the model's context ({maxpos})")
        pool = human_chunks(tok, 2 * a.m, Tmax, a.human_file, model.config.vocab_size, a.seed)
        rng = np.random.default_rng(a.seed)

        def hid(ids, c):
            P = len(ids)
            x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
            with torch.no_grad():
                hs = model.base_model(x, output_hidden_states=True).hidden_states
            h = torch.stack([torch.stack([hs[l][0, P + t - a.win:P + t].float().mean(0) for l in layers])
                             for t in W]).cpu().numpy().astype(np.float64)
            del hs
            empty_cache()
            return h                                                   # (n_win, n_layers, d)

        S = len(scenes)
        sig = np.zeros((S, len(AXES), len(layers), len(W)))
        noi = np.zeros_like(sig)
        flo = np.zeros_like(sig)          # sampling floor of the squared mean offset (no-signal level)
        for si, sc in enumerate(scenes):
            C = pool[rng.choice(len(pool), a.m, replace=False)]
            HA = np.array([hid(encode(sc["base"]), c) for c in C])     # (m, n_win, n_layers, d)
            for ai, ax in enumerate(AXES):
                HB = np.array([hid(encode(sc[ax]), c) for c in C])
                for li in range(len(layers)):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    var = 0.5 * (A_.var(0) + B_.var(0))
                    s0 = np.sqrt(var[0]) + 1e-8
                    sig[si, ai, li] = (((B_ - A_).mean(0) / s0) ** 2).mean(-1)
                    noi[si, ai, li] = (var / s0 ** 2).mean(-1)
                    flo[si, ai, li] = ((B_ - A_).var(0) / len(C) / s0 ** 2).mean(-1)
            print(f"scene {si + 1}/{S} done", flush=True)
        name = a.model + (f"@{a.revision}" if a.revision else "") + (" [random init]" if a.random_init else "")
        np.savez_compressed(npz, sig=sig, noi=noi, flo=flo, windows=np.array(W), layers=np.array(layers),
                            model=name, m=a.m)

    D = np.load(npz)
    W = [int(x) for x in D["windows"]]
    tlab = np.array(W, float) - ((a.win - 1) / 2 if a.label == "center" else 0.0)
    lt = np.log10(tlab)
    sig, noi, layers, name = D["sig"], D["noi"], [int(x) for x in D["layers"]], str(D["model"])
    flo = D["flo"] if "flo" in D.files else None
    if a.estimator == "unbiased":
        if flo is None:
            print("note: this run has no sampling floor saved; using the plug-in estimator")
            a.estimator = "plugin"
        else:
            mC = int(D["m"]); sig = sig - flo * mC / (mC - 1)
    S = sig.shape[0]
    rng = np.random.default_rng(a.seed)
    print(f"\nmodel={name}  scenes={S}  continuations per scene={int(D['m'])}  windows={W}  "
          f"(window width {a.win}, labelled by its {a.label}, {a.estimator} signal)")

    print("\nalpha / beta per layer  (95% bootstrap interval over scenes)")
    print(f"{'layer':>7s}{'alpha':>8s}{'95% CI':>16s}{'beta':>8s}")
    res = {}
    for li, l in enumerate(layers):
        a0 = fit_alpha(sig[:, :, li].mean((0, 1)), lt)
        b0 = fit_alpha(noi[:, :, li].mean((0, 1)), lt)
        bs = [fit_alpha(sig[rng.integers(0, S, S), :, li].mean((0, 1)), lt) for _ in range(a.boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res[l] = dict(alpha=a0, lo=lo, hi=hi, beta=b0)
        print(f"{'L' + str(l):>7s}{a0:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>16s}{b0:>8.2f}")

    print("\nlocal alpha per octave (between consecutive windows), all axes pooled"
          + ("; '*' = signal below 3x the sampling floor at the octave's end" if flo is not None else ""))
    hdr = f"{'layer':>7s}" + "".join(f"{f'{W[i]}-{W[i + 1]}':>11s}" for i in range(len(W) - 1))
    print(hdr); print("-" * len(hdr))
    local = {}
    for li, l in enumerate(layers):
        c = np.log10(sig[:, :, li].mean((0, 1)))
        loc = [-(c[i + 1] - c[i]) / (lt[i + 1] - lt[i]) / 2 for i in range(len(W) - 1)]
        local[l] = loc
        bad = [False] * (len(W) - 1)
        if flo is not None:
            ratio = sig[:, :, li].mean((0, 1)) / flo[:, :, li].mean((0, 1))
            bad = [ratio[i + 1] < 3 for i in range(len(W) - 1)]
        print(f"{'L' + str(l):>7s}" + "".join(f"{v:>10.2f}{'*' if b else ' '}" for v, b in zip(loc, bad)))

    print("\nalpha per axis")
    print(f"{'layer':>7s}" + "".join(f"{ax:>9s}" for ax in AXES))
    for li, l in enumerate(layers):
        print(f"{'L' + str(l):>7s}" + "".join(f"{fit_alpha(sig[:, ai, li].mean(0), lt):>9.2f}"
                                              for ai in range(len(AXES))))

    tm = np.sqrt(tlab[:-1] * tlab[1:])
    print("\nlog-power fit of the local exponents: 1/(1-alpha_loc) = ln t / k + c")
    for l in layers:
        oma = 1 - np.array(local[l])
        if np.any(oma <= 0):
            print(f"{'L' + str(l):>7s}  (local exponent >= 1 somewhere)"); continue
        y, x = 1 / oma, np.log(tm); b, c = np.polyfit(x, y, 1)
        print(f"{'L' + str(l):>7s}  k = {1 / b:.2f}  R2 = {1 - np.var(y - (c + b * x)) / np.var(y):.3f}")
    lnl = np.log(np.array(layers, float)); al = np.array([res[l]["alpha"] for l in layers])
    b_, a_ = np.polyfit(lnl, al, 1)
    print(f"\nfit alpha(l) = {a_:.2f} {b_:+.3f} ln l   (residuals: "
          + " ".join(f"{v:+.2f}" for v in al - (a_ + b_ * lnl)) + ")")
    json.dump(dict(model=name, layers=layers, windows=W, alpha={str(k): v for k, v in res.items()},
                   local={str(k): v for k, v in local.items()}, lnl_fit=[a_, b_]),
              open(os.path.join(a.out, "scale.json"), "w"), indent=1, default=float)
    print("saved", npz, "and scale.json")


if __name__ == "__main__":
    main()
