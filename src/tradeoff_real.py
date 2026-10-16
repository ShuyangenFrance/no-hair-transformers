#!/usr/bin/env python3
"""
tradeoff_real.py — tradeoff.py on the real-corpus prompt pairs of real_run.py.

For each target (prompt / placebo) and strength s, every logit from a late query q to the target span gets
+ s*log((q+1)/P). prompt = prompt tokens 1..P-1; placebo = continuation tokens P+1..2P-1 (same length),
for queries q >= 2P. Reports alpha (pooled over the four axes, bootstrap over pairs) and the perplexity
change on WikiText continuations (overall and steps 256..1024) relative to the unmodified model.

  python tradeoff_real.py --model EleutherAI/pythia-160m --revision step143000 \
      --pairs_from real/p160_step0 --out tradeoff2/p160_final
"""
import argparse, json, os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model     # noqa: E402
from tf_controls import human_chunks     # noqa: E402

AXES = ["detail", "para", "lang", "unrelated"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--revision", default=None)
    p.add_argument("--pairs_from", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strengths", default="0.25,0.5,0.75,1")
    p.add_argument("--n_pairs", type=int, default=25)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n_ppl", type=int, default=40, help="prompts used for perplexity")
    p.add_argument("--windows", default="16,32,64,128,256,512,1024")
    p.add_argument("--win", type=int, default=16)
    p.add_argument("--layers", default="")
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = [int(x) for x in a.windows.split(",")]; Tmax = max(W)
    lt = np.log10(np.array(W, float) - (a.win - 1) / 2)          # windows labelled by their centre
    pairs = json.load(open(os.path.join(a.pairs_from, "pairs.json")))
    pairs = {ax: pairs[ax][:a.n_pairs] for ax in AXES}

    la = types.SimpleNamespace(smoke=False, model=a.model, random_init=False, revision=a.revision, seed=a.seed)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(la, eager=True)
    dtype = next(model.parameters()).dtype
    L = model.config.num_hidden_layers
    layers = [int(x) for x in a.layers.split(",")] if a.layers else \
        sorted(set([2, max(1, round(L * 0.5)), L - 1]))
    pool = human_chunks(tok, 64, Tmax, "", model.config.vocab_size, a.seed)
    rng = np.random.default_rng(a.seed)

    def mask(P, N, target, s):
        m = torch.triu(torch.full((N, N), torch.finfo(dtype).min, dtype=torch.float32), diagonal=1)
        if s > 0:
            k0, k1, q0 = (1, P, P) if target == "prompt" else (P + 1, 2 * P, 2 * P)
            q = torch.arange(q0, N, dtype=torch.float32)
            m[q0:, k0:k1] += s * torch.log((q + 1) / P)[:, None]
        return m.to(dtype).to(dev)[None, None]

    def hidden(ids, c, target, s):
        P = len(ids); x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        with torch.no_grad():
            hs = model.base_model(x, attention_mask=mask(P, x.shape[1], target, s), output_hidden_states=True).hidden_states
        h = torch.stack([torch.stack([hs[l][0, P + t - a.win:P + t].float().mean(0) for l in layers]) for t in W])
        h = h.cpu().numpy().astype(np.float64); del hs; empty_cache(); return h

    def nll(ids, c, target, s):
        P = len(ids); x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        with torch.no_grad():
            h = model.base_model(x, attention_mask=mask(P, x.shape[1], target, s)).last_hidden_state[0, P - 1:-1]
            tgt = x[0, P:]; out = np.zeros(len(tgt))
            for c0 in range(0, len(tgt), 256):
                lp = torch.log_softmax(model.get_output_embeddings()(h[c0:c0 + 256]).float(), -1)
                out[c0:c0 + 256] = (-lp.gather(-1, tgt[c0:c0 + 256][:, None])[:, 0]).cpu().numpy()
        del h; empty_cache(); return out

    conts = {(ax, pi): pool[rng.choice(len(pool), a.m, replace=False)] for ax in AXES for pi in range(a.n_pairs)}
    ppl_prompts = [pairs[ax][i][0] for i in range(a.n_pairs) for ax in AXES][:a.n_ppl]
    ppl_conts = pool[rng.choice(len(pool), len(ppl_prompts), replace=True)]

    def run(target, s):
        sig = np.zeros((len(AXES), a.n_pairs, len(layers), len(W)))
        for ai, ax in enumerate(AXES):
            for pi, (s1, s2) in enumerate(pairs[ax]):
                C = conts[(ax, pi)]
                HA = np.array([hidden(encode(s1), c, target, s) for c in C])
                HB = np.array([hidden(encode(s2), c, target, s) for c in C])
                for li in range(len(layers)):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    s0 = np.sqrt(0.5 * (A_.var(0) + B_.var(0)))[0] + 1e-8
                    D_ = B_ - A_                                   # unbiased: (mean)^2 - var(ddof=1)/n
                    sig[ai, pi, li] = ((D_.mean(0) ** 2 - D_.var(0, ddof=1) / len(D_)) / s0 ** 2).mean(-1)
        nl = np.array([nll(encode(pr), c, target, s) for pr, c in zip(ppl_prompts, ppl_conts)])
        return sig, nl.mean(0)

    def alpha(curve):
        return -np.polyfit(lt, np.log10(np.maximum(curve, 1e-30)), 1)[0] / 2

    settings = [("none", 0.0)] + [(tg, float(s)) for tg in ("prompt", "placebo") for s in a.strengths.split(",")]
    res = {}
    for tg, s in settings:
        sig, nl = run(tg, s)
        al = []
        for li in range(len(layers)):
            a0 = alpha(sig[:, :, li].mean((0, 1)))
            bs = [alpha(sig[:, rng.integers(0, a.n_pairs, a.n_pairs), li].mean((0, 1))) for _ in range(a.boot)]
            al.append((a0, *np.percentile(bs, [2.5, 97.5])))
        res[(tg, s)] = (al, nl)
        print(f"[{tg} s={s}] alpha " + " ".join(f"{v[0]:.2f}" for v in al), flush=True)

    base = res[("none", 0.0)][1]; late = slice(255, Tmax)
    d = lambda nl, sl=slice(None): (np.exp(nl[sl].mean() - base[sl].mean()) - 1) * 100
    print(f"\nmodel={a.model}@{a.revision}  pairs per axis={a.n_pairs}  layers={layers}  (alpha pooled over axes)")
    print(f"{'target':9s}{'s':>6s}   " + "".join(f"{'L' + str(l):>20s}" for l in layers) + f"{'dPPL all':>11s}{'dPPL late':>11s}")
    for (tg, s), (al, nl) in res.items():
        print(f"{tg:9s}{s:>6.2f}   " + "".join(f"{f'{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]':>20s}" for v in al)
              + f"{d(nl):>10.1f}%{d(nl, late):>10.1f}%")
    print("\ndPPL: perplexity change on WikiText continuations vs. the unmodified model; late = steps 256-1024.")
    json.dump({f"{tg}|{s}": {"alpha": al, "nll": nl.tolist()} for (tg, s), (al, nl) in res.items()},
              open(os.path.join(a.out, "tradeoff.json"), "w"), default=float)


if __name__ == "__main__":
    main()
