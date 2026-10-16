#!/usr/bin/env python3
"""
real_mech.py — item 3: mechanism and causality on the real-corpus prompt pairs used by real_run.py.

For one checkpoint, on a subset of the pairs in <pairs_from>/pairs.json (same sentences as real_run.py):
  measured  : alpha of the teacher-forced offset (as real_run.py)
  paths     : exact split of the attention writes into direct (keys = prompt) and relay (keys = continuation),
              accumulated over blocks 0..l-1, so alpha_direct / alpha_relay / alpha_both can be set against
              the measured alpha at layer l, plus the relay share at the last window
  uniform   : alpha with  + log((q+1)/P)  on logits from continuation queries to prompt tokens 1..P-1
  per-head  : the same with per-head exponents gamma[l,h], calibrated (3 rounds) so that each head's
              attention to a typical prompt token stops decaying
Continuations: WikiText-103 validation chunks. Intervals: bootstrap over prompt pairs.

  python real_mech.py --model EleutherAI/pythia-160m --revision step0      --pairs_from real/p160_step0 --out mech2/p160_step0
  python real_mech.py --model EleutherAI/pythia-160m --revision step512    --pairs_from real/p160_step0 --out mech2/p160_step512
  python real_mech.py --model EleutherAI/pythia-160m --revision step143000 --pairs_from real/p160_step0 --out mech2/p160_final
  python real_mech.py ... --n_pairs 4 --m 4                                                     (quick check)
"""
import argparse, json, os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model, attach_value_norms   # noqa: E402
from tf_controls import human_chunks                       # noqa: E402

AXES = ["detail", "para", "lang", "unrelated"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--revision", default=None)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--pairs_from", required=True, help="a real_run.py output directory (pairs.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--n_pairs", type=int, default=25, help="pairs per axis (first n of pairs.json)")
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--windows", default="16,32,64,128,256,512,1024")
    p.add_argument("--win", type=int, default=16)
    p.add_argument("--layers", default="")
    p.add_argument("--cal_rounds", type=int, default=3)
    p.add_argument("--n_cal", type=int, default=8, help="base sentences used for calibration")
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = [int(x) for x in a.windows.split(",")]; Tmax = max(W)
    lt = np.log10(np.array(W, float) - (a.win - 1) / 2)           # windows labelled by their centre
    pairs = json.load(open(os.path.join(a.pairs_from, "pairs.json")))
    pairs = {ax: pairs[ax][:a.n_pairs] for ax in AXES}

    la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init, revision=a.revision, seed=a.seed)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(la, eager=True)
    dtype = next(model.parameters()).dtype
    L, H, d = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size
    layers = [int(x) for x in a.layers.split(",")] if a.layers else \
        sorted(set([1, 2, 3] + [max(1, round(L * f)) for f in (0.25, 0.5, 0.75)] + [L - 1]))
    pool = human_chunks(tok, 64, Tmax, "", model.config.vocab_size, a.seed)
    rng = np.random.default_rng(a.seed)

    vstore, vhandles = attach_value_norms(model, torch, full=True)
    gamma = np.zeros((L, H)); ctx = {"on": False, "P": 0}
    def make_hook(l):
        def hook(mod, args, kwargs):
            if not ctx["on"] or not gamma[l].any():
                return None
            m = kwargs["attention_mask"]; N, P = m.shape[-1], ctx["P"]
            q = torch.arange(P, N, dtype=torch.float32, device=m.device)
            B = torch.zeros((H, N, N), dtype=torch.float32, device=m.device)
            B[:, P:, 1:P] = torch.tensor(gamma[l], dtype=torch.float32, device=m.device)[:, None, None] \
                * torch.log((q + 1) / P)[None, :, None]
            kwargs["attention_mask"] = (m.float() + B[None]).to(m.dtype)
            return args, kwargs
        return hook
    mods = [ly.self_attn if hasattr(ly, "self_attn") else ly.attention for ly in model.base_model.layers]
    hhandles = [m_.register_forward_pre_hook(make_hook(l), with_kwargs=True) for l, m_ in enumerate(mods)]

    def mask(P, N, uniform):
        m = torch.triu(torch.full((N, N), torch.finfo(dtype).min, dtype=torch.float32), diagonal=1)
        if uniform:
            q = torch.arange(P, N, dtype=torch.float32)
            m[P:, 1:P] += torch.log((q + 1) / P)[:, None]
        return m.to(dtype).to(dev)[None, None]

    def forward(ids, c, cond, want_paths=False, want_attn=False):
        P = len(ids); ctx["P"] = P; ctx["on"] = cond == "perhead"
        x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        with torch.no_grad():
            o = model.base_model(x, attention_mask=mask(P, x.shape[1], cond == "uniform"),
                                 output_hidden_states=True, output_attentions=(want_paths or want_attn))
        hs = o.hidden_states
        h = torch.stack([torch.stack([hs[l][0, P + t - a.win:P + t].float().mean(0) for l in layers])
                         for t in W]).cpu().numpy().astype(np.float64)
        paths = None
        if want_paths:                                  # (L, n_win, 2, d): direct / relay writes of each block
            paths = np.zeros((L, len(W), 2, d))
            for k, A in enumerate(o.attentions):
                A = A[0].float(); fv = vstore[k].to(A.device)
                for j, t in enumerate(W):
                    rows = A[:, P + t - a.win:P + t, :]
                    paths[k, j, 0] = torch.einsum("hqn,hnd->d", rows[:, :, :P], fv[:, :P]).cpu().numpy() / a.win
                    paths[k, j, 1] = torch.einsum("hqn,hnd->d", rows[:, :, P:], fv[:, P:]).cpu().numpy() / a.win
        tokshare = None
        if want_attn:                                   # (L, H, n_win, P-1) attention on each prompt token
            tokshare = np.stack([np.stack([A[0, :, P + t - a.win:P + t, 1:P].float().mean(1).cpu().numpy()
                                           for t in W], 1) for A in o.attentions])
        del o, hs
        empty_cache()
        return h, paths, tokshare

    # ---- per-head calibration
    cal_sents = [pairs[ax][i][0] for i in range(min(a.n_cal, a.n_pairs)) for ax in AXES][:a.n_cal]
    cal_conts = pool[:4]
    cal_info = []
    for r in range(a.cal_rounds + 1):
        meds = []
        for s_ in cal_sents:
            for c in cal_conts:
                _, _, sh = forward(encode(s_), c, "perhead", want_attn=True)
                ts = np.polyfit(lt, np.log10(np.maximum(sh, 1e-12)).transpose(2, 0, 1, 3).reshape(len(W), -1),
                                1)[0].reshape(L, H, -1)
                meds.append(np.median(ts, -1))
        s_lh = np.mean(meds, 0)
        cal_info.append((float(np.median(s_lh)), float(np.percentile(s_lh, 10)), float(np.percentile(s_lh, 90))))
        print(f"[calibration {r}] head slopes: median {cal_info[-1][0]:+.2f}, 10-90% "
              f"[{cal_info[-1][1]:+.2f}, {cal_info[-1][2]:+.2f}]", flush=True)
        if r < a.cal_rounds:
            gamma[:] = np.clip(gamma - s_lh, 0.0, 3.0)

    # ---- measurement
    conds = ["none", "uniform", "perhead"]
    n = a.n_pairs
    sig = {c: np.zeros((len(AXES), n, len(layers), len(W))) for c in conds}
    pth = np.zeros((len(AXES), n, len(layers), 3, len(W)))            # direct / relay / both, cumulative
    for ai, ax in enumerate(AXES):
        for pi, (s1, s2) in enumerate(pairs[ax]):
            i1, i2 = encode(s1), encode(s2)
            C = pool[rng.choice(len(pool), a.m, replace=False)]
            for cond in conds:
                outA = [forward(i1, c, cond, want_paths=(cond == "none")) for c in C]
                outB = [forward(i2, c, cond, want_paths=(cond == "none")) for c in C]
                HA = np.array([o_[0] for o_ in outA]); HB = np.array([o_[0] for o_ in outB])
                for li in range(len(layers)):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    s0 = np.sqrt(0.5 * (A_.var(0) + B_.var(0)))[0] + 1e-8
                    D_ = B_ - A_
                    sig[cond][ai, pi, li] = ((D_.mean(0) ** 2 - D_.var(0, ddof=1) / len(D_)) / s0 ** 2).mean(-1)
                if cond == "none":
                    PA = np.array([o_[1] for o_ in outA]); PB = np.array([o_[1] for o_ in outB])   # (m,L,nW,2,d)
                    for li, l in enumerate(layers):
                        DA, DB = PA[:, :l].sum(1), PB[:, :l].sum(1)                  # cumulative over blocks
                        s0 = DA.sum(2)[:, 0].std(0) + 1e-8                           # (d,)
                        D = DB - DA
                        for g, Dg in enumerate([D[:, :, 0], D[:, :, 1], D.sum(2)]):
                            pth[ai, pi, li, g] = ((Dg.mean(0) ** 2 - Dg.var(0, ddof=1) / len(Dg)) / s0 ** 2).mean(-1)
            print(f"[{ax}] pair {pi + 1}/{n}", flush=True)
    for h_ in vhandles + hhandles:
        h_.remove()

    def alpha(curve):
        return -np.polyfit(lt, np.log10(np.maximum(curve, 1e-30)), 1)[0] / 2
    def boot(arr):                                       # arr: (axes, n, n_win) -> interval of the pooled alpha
        bs = [alpha(arr[:, rng.integers(0, n, n)].mean((0, 1))) for _ in range(a.boot)]
        return np.percentile(bs, [2.5, 97.5])

    name = a.model + (f"@{a.revision}" if a.revision else "") + (" [random init]" if a.random_init else "")
    print(f"\nmodel={name}  pairs per axis={n}  continuations={a.m}  windows={W}  (all four axes pooled)")
    print(f"per-head calibration, residual head slopes: median {cal_info[-1][0]:+.2f}, "
          f"10-90% [{cal_info[-1][1]:+.2f}, {cal_info[-1][2]:+.2f}], mean gamma {gamma.mean():.2f}")
    print(f"\n{'layer':>6s}{'measured':>20s}{'paths both':>12s}{'direct':>9s}{'relay':>8s}{'relay share':>13s}"
          f"{'uniform':>20s}{'per-head':>20s}")
    out = {}
    for li, l in enumerate(layers):
        am = alpha(sig["none"][:, :, li].mean((0, 1))); lo, hi = boot(sig["none"][:, :, li])
        au = alpha(sig["uniform"][:, :, li].mean((0, 1))); ulo, uhi = boot(sig["uniform"][:, :, li])
        ap = alpha(sig["perhead"][:, :, li].mean((0, 1))); plo, phi = boot(sig["perhead"][:, :, li])
        pm = pth[:, :, li].mean((0, 1))                                    # (3, n_win)
        share = pm[1, -1] / (pm[0, -1] + pm[1, -1])
        out[l] = dict(measured=[am, lo, hi], both=alpha(pm[2]), direct=alpha(pm[0]), relay=alpha(pm[1]),
                      relay_share=share, uniform=[au, ulo, uhi], perhead=[ap, plo, phi])
        print(f"{'L' + str(l):>6s}{f'{am:.2f} [{lo:.2f},{hi:.2f}]':>20s}{alpha(pm[2]):>12.2f}{alpha(pm[0]):>9.2f}"
              f"{alpha(pm[1]):>8.2f}{share:>13.2f}{f'{au:.2f} [{ulo:.2f},{uhi:.2f}]':>20s}"
              f"{f'{ap:.2f} [{plo:.2f},{phi:.2f}]':>20s}")
    print("\npaths: cumulative attention writes of blocks 0..l-1; 'both' should match 'measured' (MLPs act position-wise).")
    print("relay share = relay / (direct + relay) of the squared offset at the last window.")
    np.savez_compressed(os.path.join(a.out, "mech.npz"), pth=pth, gamma=gamma,
                        **{f"sig_{c}": v for c, v in sig.items()}, layers=np.array(layers), windows=np.array(W))
    json.dump(dict(model=name, calibration=cal_info, result={str(k): v for k, v in out.items()}),
              open(os.path.join(a.out, "mech.json"), "w"), indent=1, default=float)
    print("saved", a.out)


if __name__ == "__main__":
    main()
