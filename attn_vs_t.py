#!/usr/bin/env python3
"""
attn_vs_t.py — how attention to the prompt depends on distance, out to long contexts (late-slowdown diagnostic).

For the first few blocks of a model, prompt + WikiText continuation of length T: attention mass that continuation
queries at distance t pay to the prompt tokens (all, and without token 0), averaged over heads and prompts, in
the same 16-token windows as the offset measurements. Pure 1/t dilution means (P+t) * mass is constant; a flat or
rising (P+t) * mass beyond ~1000 tokens is the positional effect suspected in trained Qwen.

  python attn_vs_t.py --model Qwen/Qwen2.5-0.5B --pairs_from real/qwen05_final
  python attn_vs_t.py --model Qwen/Qwen2.5-0.5B --pairs_from real/qwen05_final --n_prompts 4 --m 1   (quick)
"""
import argparse, json, os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model     # noqa: E402
from tf_controls import human_chunks     # noqa: E402

WIN = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default=None)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--pairs_from", required=True)
    p.add_argument("--n_prompts", type=int, default=25)
    p.add_argument("--m", type=int, default=4, help="continuations per prompt")
    p.add_argument("--windows", default="16,32,64,128,256,512,1024,2048")
    p.add_argument("--blocks", type=int, default=3, help="analyse blocks 1..this (the rest is cut off)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    W = [int(x) for x in a.windows.split(",")]; T = max(W)

    la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init, revision=a.revision, seed=a.seed)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(la, eager=True)
    base = model.base_model
    base.layers = base.layers[:a.blocks]                      # only the first blocks are needed
    model.config.num_hidden_layers = a.blocks
    pairs = json.load(open(os.path.join(a.pairs_from, "pairs.json")))
    prompts = [pr[0] for ax in ("detail", "para", "lang", "unrelated") for pr in pairs[ax]][:a.n_prompts]
    pool = human_chunks(tok, min(64, len(prompts) * a.m), T, "", model.config.vocab_size, a.seed)
    rng = np.random.default_rng(a.seed)

    mass = np.zeros((a.blocks, len(W))); mass1 = np.zeros_like(mass); Ps = []; n = 0
    for s in prompts:
        ids = encode(s); P = len(ids); Ps.append(P)
        for c in pool[rng.choice(len(pool), a.m, replace=False)]:
            x = torch.tensor([ids + c[:T].tolist()], device=dev)
            with torch.no_grad():
                att = base(x, output_attentions=True).attentions
            for b, A in enumerate(att):
                A = A[0].float()                                  # H x N x N
                for j, t in enumerate(W):
                    rows = A[:, P + t - WIN:P + t, :]
                    mass[b, j] += rows[:, :, :P].sum(-1).mean().item()
                    mass1[b, j] += rows[:, :, 1:P].sum(-1).mean().item()
            n += 1
            del att
            empty_cache()
    mass /= n; mass1 /= n
    P = float(np.mean(Ps)); tc = np.array(W, float) - (WIN - 1) / 2
    name = a.model + (f"@{a.revision}" if a.revision else "") + (" [random init]" if a.random_init else "")
    print(f"\nmodel={name}  prompts={len(prompts)} x {a.m} continuations  mean P={P:.1f}")
    for lab, M in (("attention on the prompt", mass), ("attention on prompt tokens 1..P-1 (token 0 excluded)", mass1)):
        print(f"\n{lab}: mass, and (P+t)*mass (constant under pure 1/t dilution)")
        print(f"{'block':>6s}" + "".join(f"{f't={t}':>16s}" for t in W))
        for b in range(a.blocks):
            print(f"{b + 1:>6d}" + "".join(f"{M[b, j]:>8.4f}/{(P + tc[j]) * M[b, j]:<7.2f}" for j in range(len(W))))
        print(f"{'slope':>6s}" + "".join(f"{'':>16s}" if j == 0 else
              f"{np.log(M[0, j] / M[0, j - 1]) / np.log(tc[j] / tc[j - 1]):>16.2f}" for j in range(len(W)))
              + "   (block 1, d ln mass / d ln t between consecutive windows)")
    json.dump(dict(model=name, windows=W, P=P, mass=mass.tolist(), mass_excl0=mass1.tolist()),
              open("attn_vs_t_" + a.model.split("/")[-1].replace(".", "") + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
