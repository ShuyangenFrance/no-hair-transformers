#!/usr/bin/env python3
"""
ppl_test.py — does the dilution compensation hurt the model? Perplexity with and without the boost.

Conditions (bias on the logits from continuation queries q >= P to prompt tokens 1..P-1):
  none      : the unmodified model
  uniform   : + log((q+1)/P)                      (same for every layer and head)
  cal       : + gamma[l,h] * log((q+1)/P)         (per-head exponents fitted by tf_controls.py)
  cal_half  : + 0.5 * gamma[l,h] * log((q+1)/P)   (a milder setting, to see the trade-off)

Texts scored (prompt = each scene's base prompt, scored tokens = the continuation only):
  human : WikiText-103 chunks (or --human_file)      -> does the boost damage ordinary prediction?
  gen   : the model's own generations from data.npz  -> same, on text that does follow the prompt

Needs tfx_gamma.npy in --out (written by: python tf_controls.py --out <run> --conditions boost_cal).
  python ppl_test.py --out pythia/step143000
Prints mean NLL and perplexity per generation-step band and overall, and the change relative to 'none'.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model, restore_run_args   # noqa: E402
from tf_controls import human_chunks                      # noqa: E402

BANDS = [1, 16, 32, 64, 128, 256, 513]          # continuation positions t, bins [a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", default="none,uniform,cal,cal_half")
    p.add_argument("--texts", default="human,gen")
    p.add_argument("--m", type=int, default=8, help="sequences per scene and text type")
    p.add_argument("--human_file", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None); p.add_argument("--smoke", action="store_true")
    p.add_argument("--random_init", action="store_true"); p.add_argument("--revision", default=None)
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    scenes = restore_run_args(a, meta)
    gpath = os.path.join(a.out, "tfx_gamma.npy")
    conds = a.conditions.split(",")
    if any(c.startswith("cal") for c in conds) and not os.path.exists(gpath):
        sys.exit(f"{gpath} not found: run  python tf_controls.py --out {a.out} --conditions boost_cal  first")
    gamma_cal = np.load(gpath) if os.path.exists(gpath) else None

    torch, model, tok, encode, _, _, dev, empty_cache = load_model(a, eager=True)
    dtype = next(model.parameters()).dtype
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    Tmax = max(meta["windows"])

    # per-layer bias through pre-hooks on the attention modules (same mechanism as tf_controls.py)
    gamma = np.zeros((L, H)); ctx = {"P": 0}
    def make_hook(l):
        def hook(mod, args, kwargs):
            if not gamma[l].any():
                return None
            m = kwargs["attention_mask"]
            N, P = m.shape[-1], ctx["P"]
            q = torch.arange(P, N, dtype=torch.float32, device=m.device)
            lq = torch.log((q + 1) / P)
            g = torch.tensor(gamma[l], dtype=torch.float32, device=m.device)
            B = torch.zeros((H, N, N), dtype=torch.float32, device=m.device)
            B[:, P:, 1:P] = g[:, None, None] * lq[None, :, None]
            kwargs["attention_mask"] = (m.float() + B[None]).to(m.dtype)
            return args, kwargs
        return hook
    mods = [ly.self_attn if hasattr(ly, "self_attn") else ly.attention for ly in model.base_model.layers]
    handles = [m_.register_forward_pre_hook(make_hook(l), with_kwargs=True) for l, m_ in enumerate(mods)]

    def causal(N):
        m = torch.triu(torch.full((N, N), torch.finfo(dtype).min, dtype=torch.float32), diagonal=1)
        return m.to(dtype).to(dev)[None, None]

    def nll(ids, c):
        """per-token NLL of the continuation tokens c[0..Tmax-1], each predicted from everything before it."""
        P = len(ids)
        x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        ctx["P"] = P
        with torch.no_grad():
            logits = model(x, attention_mask=causal(x.shape[1])).logits[0].float()
        lp = torch.log_softmax(logits[P - 1:-1], -1)                # predicts tokens P .. P+Tmax-1
        tgt = x[0, P:]
        out = -lp.gather(-1, tgt[:, None])[:, 0].cpu().numpy()     # (Tmax,), index t-1 for step t
        del logits, lp
        empty_cache()
        return out

    texts = {}
    if "human" in a.texts:
        texts["human"] = human_chunks(tok, a.m, Tmax, a.human_file, model.config.vocab_size, a.seed)
    res = {}
    for txt in a.texts.split(","):
        for cond in conds:
            if cond == "none":
                gamma[:] = 0
            elif cond == "uniform":
                gamma[:] = 1.0
            elif cond == "cal":
                gamma[:] = gamma_cal
            elif cond == "cal_half":
                gamma[:] = 0.5 * gamma_cal
            rows = []
            for si, sc in enumerate(scenes):
                ids = encode(sc["base"])
                C = texts["human"] if txt == "human" else Z[f"conts|{si}|base"][:a.m]
                rows += [nll(ids, c) for c in C]
            res[(txt, cond)] = np.array(rows)                     # (n_seq, Tmax)
            print(f"[{txt}] {cond} done", flush=True)
    for h_ in handles:
        h_.remove()

    t = np.arange(1, Tmax + 1)
    bands = [(lo, hi) for lo, hi in zip(BANDS[:-1], BANDS[1:]) if lo <= Tmax]
    print(f"\nmodel={meta['model']}   mean NLL per token (nats); in brackets: change vs 'none' in %, of perplexity")
    for txt in a.texts.split(","):
        print(f"\n[{txt}]")
        hdr = f"{'condition':10s}" + "".join(f"{f't {lo}-{min(hi, Tmax + 1) - 1}':>18s}" for lo, hi in bands) \
              + f"{'all':>18s}"
        print(hdr); print("-" * len(hdr))
        base = res[(txt, "none")] if (txt, "none") in res else None
        for cond in conds:
            R = res[(txt, cond)]
            cells = []
            for lo, hi in bands + [(1, Tmax + 1)]:
                sel = (t >= lo) & (t < hi)
                v = R[:, sel].mean()
                d = "" if base is None or cond == "none" else f"({(np.exp(v - base[:, sel].mean()) - 1) * 100:+.1f}%)"
                cells.append(f"{v:7.3f} {d:>9s}")
            print(f"{cond:10s}" + "".join(f"{c_:>18s}" for c_ in cells))
    json.dump({f"{k[0]}|{k[1]}": v.mean(0).tolist() for k, v in res.items()},
              open(os.path.join(a.out, "ppl.json"), "w"))
    print("\nsaved", os.path.join(a.out, "ppl.json"), "(mean NLL at each step t, per text and condition)")


if __name__ == "__main__":
    main()
