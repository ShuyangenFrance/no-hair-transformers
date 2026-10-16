#!/usr/bin/env python3
"""
prenorm_check.py — empirical check of Assumption 1 (non-degenerate pre-norm activations).

For every residual-stream state that enters a LayerNorm (inputs of every block and of the final norm),
computes ||Pi h|| / sqrt(d), where Pi removes the mean over features (RMSNorm models: use --rms, Pi = I).
Runs prompt + continuation sequences from a real_run.py pairs.json and WikiText continuations,
and reports per layer the minimum, the 0.01% quantile and the median over all positions and sequences.
The theorem needs r = min > 0; the reported minimum is the r used for the constants.

  python prenorm_check.py --model EleutherAI/pythia-160m --revision step143000 --pairs_from real/p160_step0
"""
import argparse, json, os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model     # noqa: E402
from tf_controls import human_chunks     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--revision", default=None)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--pairs_from", required=True)
    p.add_argument("--n_prompts", type=int, default=40)
    p.add_argument("--m", type=int, default=2, help="continuations per prompt")
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--rms", action="store_true", help="RMSNorm model: do not subtract the mean")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init, revision=a.revision, seed=a.seed)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(la)
    pairs = json.load(open(os.path.join(a.pairs_from, "pairs.json")))
    prompts = [s for ax in pairs for pr in pairs[ax] for s in pr][:a.n_prompts]
    pool = human_chunks(tok, 16, a.T, "", model.config.vocab_size, a.seed)
    rng = np.random.default_rng(a.seed)
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    vals = [[] for _ in range(L + 1)]          # hidden_states[0..L-1] enter blocks; index L = input of the final norm
    final_norm = getattr(model.base_model, "final_layer_norm", None) or getattr(model.base_model, "norm")
    cap = {}
    hk = final_norm.register_forward_pre_hook(lambda mod, args: cap.__setitem__("x", args[0].detach()))

    for k, s in enumerate(prompts):
        for c in pool[rng.choice(len(pool), a.m, replace=False)]:
            x = torch.tensor([encode(s) + c[:a.T].tolist()], device=dev)
            with torch.no_grad():
                out = model.base_model(x, output_hidden_states=True)
            hs = list(out.hidden_states[:L]) + [cap["x"]]
            for l, h in enumerate(hs):
                h = h[0].float()
                if not a.rms:
                    h = h - h.mean(-1, keepdim=True)
                vals[l].append((h.norm(dim=-1) / d ** 0.5).cpu().numpy())
            del out
            empty_cache()
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(prompts)} prompts", flush=True)

    print(f"\nmodel={a.model}{'@' + a.revision if a.revision else ''}  sequences={len(prompts) * a.m}  T={a.T}")
    print("||Pi h|| / sqrt(d) at the input of each block (block 0 input = embeddings) and of the final norm")
    print(f"{'block':>6s}{'min':>10s}{'0.01%':>10s}{'median':>10s}")
    rmin = np.inf
    for l in range(L + 1):
        v = np.concatenate(vals[l])
        rmin = min(rmin, v.min())
        lab = str(l) if l < L else "final"
        print(f"{lab:>6s}{v.min():>10.4f}{np.quantile(v, 1e-4):>10.4f}{np.median(v):>10.4f}")
    hk.remove()
    print(f"\nr / sqrt(d) = {rmin:.4f}  (minimum over all blocks, positions and sequences)")


if __name__ == "__main__":
    main()
