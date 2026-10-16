#!/usr/bin/env python3
"""
tradeoff.py — how much forgetting can be removed, at what perplexity cost, and is the cost about
remembering the prompt or about distorting attention at all?

For a strength s, every attention logit from a late query q to a TARGET span of keys gets
    + s * log((q+1) / P)        (all layers and heads; s = 1 is the 'uniform' boost of tf_controls.py)
Targets:
  prompt  : prompt tokens 1..P-1                         (the intervention we care about)
  placebo : continuation tokens P+1..2P-1, same length,  (equally strong distortion of attention,
            boosted for queries q >= 2P                    but towards text that is not the prompt)
If the perplexity cost of 'placebo' is as large as that of 'prompt', the cost comes from distorting
attention, not from remembering the prompt.

Measures, for each (target, strength):
  alpha : decay exponent of the teacher-forced prompt offset (as in alpha_beta.py / tf_controls.py)
  ppl   : perplexity change vs. the unmodified model on human text (WikiText) and on the model's own text

  python tradeoff.py --out pythia/step143000
  python tradeoff.py --out pythia/step143000 --strengths 0.25,0.5,0.75,1 --m 16
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model, restore_run_args   # noqa: E402
from tf_controls import human_chunks                      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--strengths", default="0.25,0.5,0.75,1")
    p.add_argument("--targets", default="prompt,placebo")
    p.add_argument("--m", type=int, default=16, help="shared continuations per prompt pair (alpha)")
    p.add_argument("--m_ppl", type=int, default=8, help="sequences per scene and text type (perplexity)")
    p.add_argument("--human_file", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None); p.add_argument("--smoke", action="store_true")
    p.add_argument("--random_init", action="store_true"); p.add_argument("--revision", default=None)
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    scenes = restore_run_args(a, meta)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(a, eager=True)
    dtype = next(model.parameters()).dtype
    W, win, axes, layers = meta["windows"], meta["win"], meta["axes"], meta["layers"]
    Tmax = max(W)
    lt = np.log10(np.array(W, float))

    def mask(P, N, target, s):
        """causal additive mask (1,1,N,N) with the boost on the target span."""
        m = torch.triu(torch.full((N, N), torch.finfo(dtype).min, dtype=torch.float32), diagonal=1)
        if s > 0:
            if target == "prompt":
                k0, k1, q0 = 1, P, P
            else:                                                   # placebo: same length, inside the continuation
                k0, k1, q0 = P + 1, 2 * P, 2 * P
            q = torch.arange(q0, N, dtype=torch.float32)
            m[q0:, k0:k1] += s * torch.log((q + 1) / P)[:, None]
        return m.to(dtype).to(dev)[None, None]

    def hidden(ids, c, target, s):
        P = len(ids)
        x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        with torch.no_grad():
            hs = model.base_model(x, attention_mask=mask(P, x.shape[1], target, s),
                                  output_hidden_states=True).hidden_states
        h = torch.stack([torch.stack([hs[l][0, P + t - win:P + t].float().mean(0) for l in layers])
                         for t in W]).cpu().numpy()
        del hs
        empty_cache()
        return h

    def nll(ids, c, target, s):
        P = len(ids)
        x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        with torch.no_grad():
            logits = model(x, attention_mask=mask(P, x.shape[1], target, s)).logits[0].float()
        lp = torch.log_softmax(logits[P - 1:-1], -1)
        out = -lp.gather(-1, x[0, P:][:, None])[:, 0].cpu().numpy()
        del logits, lp
        empty_cache()
        return out

    def alpha_of(target, s):
        sig = {l: [] for l in layers}
        for si, sc in enumerate(scenes):
            ib = encode(sc["base"])
            for ax in axes:
                ia = encode(sc[ax]); h = a.m // 2
                C = np.concatenate([Z[f"conts|{si}|base"][:h], Z[f"conts|{si}|{ax}"][:h]])
                HA = np.array([hidden(ib, c, target, s) for c in C], np.float64)
                HB = np.array([hidden(ia, c, target, s) for c in C], np.float64)
                for li, l in enumerate(layers):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    s0 = np.sqrt(0.5 * (A_.var(0) + B_.var(0)))[0] + 1e-8
                    sig[l].append((((B_ - A_).mean(0) / s0) ** 2).mean(-1))
        return [-np.polyfit(lt, np.log10(np.mean(sig[l], 0)), 1)[0] / 2 for l in layers]

    human = human_chunks(tok, a.m_ppl, Tmax, a.human_file, model.config.vocab_size, a.seed)
    def ppl_of(target, s):
        out = {}
        for txt in ("human", "gen"):
            rows = []
            for si, sc in enumerate(scenes):
                ids = encode(sc["base"])
                C = human if txt == "human" else Z[f"conts|{si}|base"][:a.m_ppl]
                rows += [nll(ids, c, target, s) for c in C]
            out[txt] = np.array(rows).mean(0)                     # mean NLL at each step t
        return out

    settings = [("none", 0.0)] + [(tg, float(s)) for tg in a.targets.split(",")
                                  for s in a.strengths.split(",")]
    res = {}
    for tg, s in settings:
        al = alpha_of(tg, s)
        pp = ppl_of(tg, s)
        res[(tg, s)] = (al, pp)
        print(f"[{tg} s={s}] alpha " + " ".join(f"{v:.2f}" for v in al), flush=True)

    base = res[("none", 0.0)][1]
    late = slice(255, Tmax)                                       # steps 256..512
    def dppl(pp, txt, sl=slice(None)):
        return (np.exp(pp[txt][sl].mean() - base[txt][sl].mean()) - 1) * 100

    print(f"\nmodel={meta['model']}   alpha per layer {layers}; perplexity change vs. unmodified model")
    hdr = (f"{'target':9s}{'s':>6s}   {'alpha (layers)':24s}{'mean a':>8s}"
           f"{'human all':>11s}{'human late':>12s}{'gen all':>10s}{'gen late':>10s}")
    print(hdr); print("-" * len(hdr))
    for (tg, s), (al, pp) in res.items():
        print(f"{tg:9s}{s:>6.2f}   {' '.join(f'{v:5.2f}' for v in al):24s}{np.mean(al):>8.2f}"
              f"{dppl(pp, 'human'):>10.1f}%{dppl(pp, 'human', late):>11.1f}%"
              f"{dppl(pp, 'gen'):>9.1f}%{dppl(pp, 'gen', late):>9.1f}%")
    print("\nlate = generation steps 256-512. Placebo boosts a same-length span at the start of the continuation;")
    print("its alpha can also move (that span carries relayed prompt information), so read placebo mainly for perplexity.")
    json.dump({f"{tg}|{s}": {"alpha": al, "nll_human": pp["human"].tolist(), "nll_gen": pp["gen"].tolist()}
               for (tg, s), (al, pp) in res.items()}, open(os.path.join(a.out, "tradeoff.json"), "w"))
    print("saved", os.path.join(a.out, "tradeoff.json"))


if __name__ == "__main__":
    main()
