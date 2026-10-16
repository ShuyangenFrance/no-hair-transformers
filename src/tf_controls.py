#!/usr/bin/env python3
"""
tf_controls.py — two teacher-forced controls for the decay exponent alpha (offset ~ t^-alpha).

  gen    : baseline. Shared continuations = the model's own generations (as in gen/analyze).
  human  : shared continuations = human-written text (WikiText-103 by default, or --human_file).
           Same alpha -> forgetting is a property of the network, not of the statistics of generated text.
  boost  : model generations as continuations, but every attention logit from a continuation query to a
           prompt key gets  + p * log((q+1)/P)  (q = query position, P = prompt length). With p = 1 this
           cancels the ~1/t dilution of attention to the prompt (a stronger, t-dependent split-softmax).
           If forgetting were just attention dilution, alpha should drop to ~0.
  boost_cal: the same idea, calibrated per layer and head. Each head gets its own exponent gamma[l,h]
           (bias gamma * log((q+1)/P) on prompt tokens 1..P-1), fitted so that the attention that head pays
           to a typical prompt token (median over tokens, so in-prompt sinks such as '.' do not dominate)
           stops decaying with t. Fitted iteratively (--cal_rounds), because boosting one layer changes the
           attention of the layers above it.

Reuses the run in --out (meta.json, data.npz from nohair_gen_v2.py gen); no new generation.
  python tf_controls.py --out pythia/step512
  python tf_controls.py --out pythia/step512 --human_file my_text.txt      # local text instead of WikiText
  python tf_controls.py --out pythia/step512 --conditions gen,boost,boost_cal
Writes tfx.json and prints alpha / beta per condition and layer, plus how attention to the prompt
decays with and without the boost (to check the compensation worked).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model, restore_run_args   # noqa: E402


def human_chunks(tok, n, L, path, vocab, seed=0):
    """n chunks of L token ids of human-written text, spread evenly through the source."""
    if tok is None:                                     # smoke run: no tokenizer
        return np.random.default_rng(seed).integers(0, vocab, (n, L))
    if path:
        text = open(path, encoding="utf-8").read()
        src = path
    else:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        text = "".join(ds["text"])
        src = "WikiText-103 validation"
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) < n * L:
        sys.exit(f"human text too short: {len(ids)} tokens, need {n * L}")
    stride = len(ids) // n
    print(f"human continuations: {n} x {L} tokens from {src} ({len(ids)} tokens)", flush=True)
    return np.array([ids[i * stride: i * stride + L] for i in range(n)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", default="gen,human,boost")
    p.add_argument("--m", type=int, default=16, help="shared continuations per prompt pair")
    p.add_argument("--boost_power", type=float, default=1.0)
    p.add_argument("--human_file", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cal_rounds", type=int, default=3)
    p.add_argument("--n_cal", type=int, default=4, help="continuations per scene used for calibration")
    p.add_argument("--model", default=None); p.add_argument("--smoke", action="store_true")
    p.add_argument("--random_init", action="store_true"); p.add_argument("--revision", default=None)
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    scenes = restore_run_args(a, meta)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(a, eager=True)
    W, win, S, axes, layers = meta["windows"], meta["win"], meta["n_scenes"], meta["axes"], meta["layers"]
    Tmax = max(W)
    conds = a.conditions.split(",")
    dtype = next(model.parameters()).dtype
    if meta.get("scenes", "zh") == "zh" and "human" in conds and not a.human_file:
        print("note: Chinese prompt set with English WikiText continuations; pass --human_file for Chinese text")

    human = human_chunks(tok, a.m, Tmax, a.human_file, model.config.vocab_size, a.seed) if "human" in conds else None

    lt = np.log10(np.array(W, float))

    def mask4d(P, N, boost):
        """additive mask: causal, plus the prompt boost for continuation queries."""
        neg = torch.finfo(dtype).min
        m = torch.full((N, N), neg, dtype=torch.float32)
        m = torch.triu(m, diagonal=1)                                   # 0 on and below the diagonal
        if boost:
            q = torch.arange(P, N, dtype=torch.float32)
            m[P:, :P] += a.boost_power * torch.log((q + 1) / P)[:, None]
        return m.to(dtype).to(dev)[None, None]

    # ---- per-layer, per-head boost (boost_cal): replace each attention module's mask through a pre-hook
    L = model.config.num_hidden_layers; H = model.config.num_attention_heads
    gamma = np.zeros((L, H))
    ctx = {"active": False, "P": 0}
    def make_hook(l):
        def hook(mod, args, kwargs):
            if not ctx["active"] or not gamma[l].any():
                return None
            m = kwargs["attention_mask"]                                     # (1, 1, N, N) additive
            N, P = m.shape[-1], ctx["P"]
            q = torch.arange(P, N, dtype=torch.float32, device=m.device)
            lq = torch.log((q + 1) / P)                                      # (N-P,)
            g = torch.tensor(gamma[l], dtype=torch.float32, device=m.device)
            B = torch.zeros((H, N, N), dtype=torch.float32, device=m.device)
            B[:, P:, 1:P] = g[:, None, None] * lq[None, :, None]
            kwargs["attention_mask"] = (m.float() + B[None]).to(m.dtype)
            return args, kwargs
        return hook
    attn_mods = [ly.self_attn if hasattr(ly, "self_attn") else ly.attention for ly in model.base_model.layers]
    handles = [m_.register_forward_pre_hook(make_hook(l), with_kwargs=True) for l, m_ in enumerate(attn_mods)]

    def head_slopes(cal_set):
        """slope in log t of the attention each head pays to a typical prompt token (median over tokens)."""
        acc = None
        for ids, c in cal_set:
            P = len(ids)
            x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
            ctx["P"] = P
            with torch.no_grad():
                o = model.base_model(x, attention_mask=mask4d(P, x.shape[1], False), output_attentions=True)
            # (L, H, n_win, P-1): mean over the window's queries of the attention on each prompt token
            sh = np.stack([np.stack([A[0, :, P + t - win:P + t, 1:P].float().mean(1).cpu().numpy() for t in W], 1)
                           for A in o.attentions])
            del o
            empty_cache()
            tok_slopes = np.polyfit(lt, np.log10(np.maximum(sh, 1e-12)).transpose(2, 0, 1, 3).reshape(len(W), -1),
                                    1)[0].reshape(L, H, P - 1)
            med = np.median(tok_slopes, -1)
            acc = med[None] if acc is None else np.concatenate([acc, med[None]])
        return acc.mean(0)                             # (L, H)

    def run(ids, c, boost, want_attn):
        """hidden states (n_win, n_layers, d) averaged over each window; optionally prompt attention share."""
        P = len(ids)
        x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
        ctx["P"] = P
        with torch.no_grad():
            o = model.base_model(x, attention_mask=mask4d(P, x.shape[1], boost == "uniform"),
                                 output_hidden_states=True, output_attentions=want_attn)
        hs = o.hidden_states
        h = torch.stack([torch.stack([hs[l][0, P + t - win:P + t].float().mean(0) for l in layers])
                         for t in W]).cpu().numpy()
        att = None
        if want_attn:   # share of attention on prompt tokens 1..P-1, mean over layers, heads, window queries
            att = np.array([np.mean([A[0, :, P + t - win:P + t, 1:P].float().sum(-1).mean().item()
                                     for A in o.attentions]) for t in W])
        del o, hs
        empty_cache()
        return h, att

    results, attn_slopes, cal_info = {}, {}, {}
    for cond in conds:
        boost = {"boost": "uniform", "boost_cal": "cal"}.get(cond, "none")
        ctx["active"] = boost == "cal"
        if boost == "cal" and not cal_info:
            cal_set = [(encode(sc["base"]), c) for si, sc in enumerate(scenes)
                       for c in Z[f"conts|{si}|base"][:a.n_cal]]
            for r in range(a.cal_rounds):
                s_lh = head_slopes(cal_set)
                print(f"[calibration round {r}] head slopes before update: median {np.median(s_lh):+.2f}, "
                      f"10-90% [{np.percentile(s_lh, 10):+.2f}, {np.percentile(s_lh, 90):+.2f}]", flush=True)
                gamma[:] = np.clip(gamma - s_lh, 0.0, 3.0)
            s_lh = head_slopes(cal_set)
            cal_info = dict(final_median=float(np.median(s_lh)), p10=float(np.percentile(s_lh, 10)),
                            p90=float(np.percentile(s_lh, 90)), gamma_mean=float(gamma.mean()))
            print(f"[calibrated] residual head slopes: median {cal_info['final_median']:+.2f}, "
                  f"10-90% [{cal_info['p10']:+.2f}, {cal_info['p90']:+.2f}];  mean gamma {gamma.mean():.2f}",
                  flush=True)
        sig = {l: [] for l in layers}; noi = {l: [] for l in layers}; att_curves = []
        for si, sc in enumerate(scenes):
            ib = encode(sc["base"])
            for ax in axes:
                ia = encode(sc[ax])
                if cond == "human":
                    C = human
                else:
                    h = a.m // 2
                    C = np.concatenate([Z[f"conts|{si}|base"][:h], Z[f"conts|{si}|{ax}"][:h]])
                want = (cond in ("gen", "boost", "boost_cal")) and ax == axes[0]   # attention check: one axis
                HA, HB = [], []
                for c in C:
                    hA, at = run(ib, c, boost, want)
                    hB, _ = run(ia, c, boost, False)
                    HA.append(hA); HB.append(hB)
                    if at is not None:
                        att_curves.append(at)
                HA, HB = np.array(HA, np.float64), np.array(HB, np.float64)    # (m, n_win, n_layers, d)
                for li, l in enumerate(layers):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    var = 0.5 * (A_.var(0) + B_.var(0))
                    s0 = np.sqrt(var[0]) + 1e-8
                    sig[l].append((((B_ - A_).mean(0) / s0) ** 2).mean(-1))
                    noi[l].append((var / s0 ** 2).mean(-1))
                print(f"[{cond}] scene {si} {ax} done", flush=True)
        for l in layers:
            a2 = np.polyfit(lt, np.log10(np.mean(sig[l], 0)), 1)[0]
            b2 = np.polyfit(lt, np.log10(np.mean(noi[l], 0)), 1)[0]
            results[(cond, l)] = (-a2 / 2, -b2 / 2)
        if att_curves:
            attn_slopes[cond] = float(np.polyfit(lt, np.log10(np.mean(att_curves, 0)), 1)[0])

    print(f"\nmodel={meta['model']}   alpha (offset ~ t^-alpha) / beta (noise ~ t^-beta)")
    print(f"{'layer':8s}" + "".join(f"{c:>18s}" for c in conds))
    for l in layers:
        print(f"L{l:<7d}" + "".join(f"{results[(c, l)][0]:>9.2f} / {results[(c, l)][1]:<6.2f}" for c in conds))
    if attn_slopes:
        print("\nattention to the prompt (tokens 1..P-1), slope in log t:  "
              + "   ".join(f"{c}: {v:+.2f}" for c, v in attn_slopes.items())
              + "   (boost should be ~0 if the compensation worked)")
    if cal_info:
        print(f"boost_cal: per-head residual slopes of attention to a typical prompt token after calibration: "
              f"median {cal_info['final_median']:+.2f}, 10-90% [{cal_info['p10']:+.2f}, {cal_info['p90']:+.2f}]")
    for h_ in handles:
        h_.remove()
    np.save(os.path.join(a.out, "tfx_gamma.npy"), gamma)
    json.dump({"alpha_beta": {f"{c}|{l}": v for (c, l), v in results.items()}, "attn_slope": attn_slopes,
               "calibration": cal_info,
               "m": a.m, "boost_power": a.boost_power, "human_file": a.human_file},
              open(os.path.join(a.out, "tfx.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, "tfx.json"))


if __name__ == "__main__":
    main()