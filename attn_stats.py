#!/usr/bin/env python3
"""
attn_stats.py — attention statistics for the remaining TODOs of the paper.

Per layer, over late queries (continuation positions >= 64) of real-corpus prompts + WikiText continuations:
  sink     : mean attention weight on token 0                                   (TODO: sink mass across training)
  entropy  : attention entropy divided by log(number of visible keys)            (TODO: untrained Qwen, layers 1-3)
  max_w    : mean of the largest attention weight of each query                 (TODO: untrained Qwen, layers 1-3)
  spread   : logit spread max_k s_ik - min_k s_ik = ln(max alpha / min alpha)   (TODO: Delta in Remark rem:constant)
             reported as median and maximum over queries and heads
  mid_min  : (sequential blocks only, e.g. Qwen) min of ||h||/sqrt(d) at the input of the post-attention norm
             (TODO: post-attention residual of the sequential Qwen blocks; RMSNorm, so no centring)

  python attn_stats.py --model EleutherAI/pythia-160m --revision step0 --pairs_from real/p160_step0
  python attn_stats.py --model Qwen/Qwen2.5-0.5B --random_init --pairs_from real/p160_step0
  for s in 0 128 256 512 1000 4000 16000 143000; do
    python attn_stats.py --model EleutherAI/pythia-160m --revision step$s --pairs_from real/p160_step0 --quiet
  done
"""
import argparse, json, math, os, sys, types
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
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--q0", type=int, default=64, help="only continuation queries at or after this step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="print one summary line per layer group only")
    a = p.parse_args()

    la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init, revision=a.revision, seed=a.seed)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(la, eager=True)
    L = model.config.num_hidden_layers
    pairs = json.load(open(os.path.join(a.pairs_from, "pairs.json")))
    prompts = [s for ax in pairs for pr in pairs[ax] for s in pr][:a.n_prompts]
    n_pool = min(len(prompts), 200)              # WikiText-103 validation holds about 200 disjoint 1024-token chunks
    pool = human_chunks(tok, n_pool, a.T, "", model.config.vocab_size, a.seed)

    # mid-block residual (sequential blocks): input of the post-attention norm
    mids, handles = {l: [] for l in range(L)}, []
    for l, ly in enumerate(model.base_model.layers):
        if hasattr(ly, "post_attention_layernorm") and not getattr(model.config, "use_parallel_residual", False):
            handles.append(ly.post_attention_layernorm.register_forward_pre_hook(
                lambda mod, args, l=l: mids[l].append(args[0][0].float().norm(dim=-1).min().item()
                                                      / math.sqrt(args[0].shape[-1]))))

    acc = {k: [[] for _ in range(L)] for k in ("sink", "entropy", "max_w", "spread")}
    for k, s in enumerate(prompts):
        c = pool[k % n_pool]                     # chunks are reused cyclically when prompts outnumber them
        ids = encode(s); P = len(ids)
        x = torch.tensor([ids + c[:a.T].tolist()], device=dev)
        with torch.no_grad():
            att = model.base_model(x, output_attentions=True).attentions
        for l, A in enumerate(att):
            A = A[0].float()[:, P + a.q0:, :]                       # H x Q x N (late continuation queries)
            n = torch.arange(P + a.q0 + 1, x.shape[1] + 1, device=A.device, dtype=torch.float32)  # visible keys
            acc["sink"][l].append(A[:, :, 0].mean().item())
            ent = -(A * torch.log(A.clamp(min=1e-30))).sum(-1) / torch.log(n)[None, :]
            acc["entropy"][l].append(ent.mean().item())
            acc["max_w"][l].append(A.max(-1).values.mean().item())
            mask = torch.arange(A.shape[-1], device=A.device)[None, :] < n[:, None]  # causal: visible keys only
            Amin = torch.where(mask[None], A, torch.full_like(A, float("inf"))).min(-1).values.clamp(min=1e-30)
            spread = torch.log(A.max(-1).values / Amin)              # = max_k s - min_k s; saturates at ln(1e30)=69.08
                                                                     # when the smallest weight underflows in float32
            acc["spread"][l].append((spread.median().item(), spread.max().item()))
        del att
        empty_cache()
    for h in handles:
        h.remove()

    name = a.model + (f"@{a.revision}" if a.revision else "") + (" [random init]" if a.random_init else "")
    print(f"\nmodel={name}  prompts={len(prompts)}  T={a.T}  queries from continuation step {a.q0}")
    print(f"{'layer':>6s}{'sink':>8s}{'entropy':>9s}{'max_w':>8s}{'spread med':>12s}{'spread max':>12s}"
          + (f"{'mid_min':>10s}" if handles else ""))
    out = {}
    for l in range(L):
        sp = np.array(acc["spread"][l])
        row = dict(sink=float(np.mean(acc["sink"][l])), entropy=float(np.mean(acc["entropy"][l])),
                   max_w=float(np.mean(acc["max_w"][l])), spread_med=float(np.median(sp[:, 0])),
                   spread_max=float(sp[:, 1].max()), mid_min=float(min(mids[l])) if mids[l] else None)
        out[l] = row
        if a.quiet and l not in (0, 1, 2, L // 2, L - 1):
            continue
        print(f"{l + 1:>6d}{row['sink']:>8.3f}{row['entropy']:>9.3f}{row['max_w']:>8.3f}"
              f"{row['spread_med']:>12.2f}{row['spread_max']:>12.2f}"
              + (f"{row['mid_min']:>10.3f}" if row["mid_min"] is not None else ""))
    print("layer = block index + 1 (so 'layer 1' is the first block, as in the paper's L1 after block 0)")
    tag = (a.model.split("/")[-1] + "_" + (a.revision or ("random" if a.random_init else "final"))).replace(".", "")
    os.makedirs("attn_stats", exist_ok=True)
    json.dump(out, open(f"attn_stats/{tag}.json", "w"), indent=1)
    print(f"saved attn_stats/{tag}.json")


if __name__ == "__main__":
    main()
