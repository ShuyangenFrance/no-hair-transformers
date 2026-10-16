#!/usr/bin/env python3
"""
multiturn_test.py — does a system instruction survive a long conversation, with and without the boost?

Setup: system prompt "answer only in French". Turn 1: the user asks (in English) for a long story.
Every later turn the user says, in English, "continue with the next chapter". The model answers each turn
until it stops by itself (no forced length), so every reply is a natural one. The English user turns keep
pulling towards English, which is where drift should show up.

Conditions (bias on every attention logit from a later position to the system-prompt tokens, all layers
and heads, positions counted over the whole conversation, S = end of the system prompt):
  none            unmodified model
  log:s           + s * log((q+1)/S)
  const:match:s   + c, with c = the mean bias that log:s actually applied to its generated tokens
                  (log:s must come earlier in --conditions). Same total boost, different shape:
                  if log beats const in late turns, the log t shape matters.

Reported per turn: French share of the reply, instruction echo (leak words per 1000), reply length.
Summary: French share over the last 3 turns, share of conversations with every turn >= 0.5 French,
first turn below 0.5, leak/1k overall, mean reply length, repeated 4-grams.

  python multiturn_test.py --quick
  python multiturn_test.py                       # Qwen2.5-1.5B-Instruct, 4 topics x 8 samples x 8 turns
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_test import SYSTEM, TOPICS, USER, fr_share, leak_count, rep4   # noqa: E402

FOLLOW = [
    "That's great! Please continue with the next chapter.",
    "I love it. What happens next? Write the next chapter.",
    "Nice. Continue the story, and make this chapter even more detailed.",
    "Keep going with the next chapter, please.",
    "Wonderful. Please write the following chapter now.",
    "Great work. Continue the story with a new chapter.",
    "Thanks! Go on with the next part of the story.",
    "Please continue. I want to know how the story develops.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--conditions", default="none,log:0.1,const:match:0.1,log:0.25,const:match:0.25")
    p.add_argument("--n_topics", type=int, default=4)
    p.add_argument("--samples", type=int, default=8, help="conversations per topic (one batch)")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--turn_max", type=int, default=400, help="max new tokens per reply")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="multiturn")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n_topics, a.samples, a.turns, a.turn_max = 1, 2, 3, 200

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else \
        ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="eager").to(dev).eval()
    print("device:", dev, flush=True)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    eos_ids = {i for i in [im_end, tok.eos_token_id] if i is not None}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    head = tok("<|im_start|>system\n", add_special_tokens=False).input_ids

    # ---- boost with left padding: each row has its own offset
    ctx = {"kind": "none", "s": 0.0, "s0": 0, "s1": 1, "pads": None, "cache": None}
    def hook(mod, args, kwargs):
        if ctx["kind"] == "none" or ctx["s"] == 0:
            return None
        m = kwargs.get("attention_mask")
        if m is None or m.dim() != 4 or not m.is_floating_point():
            raise RuntimeError("expected a 4D additive attention mask (eager attention)")
        Bn, _, Q, K = m.shape
        key = (Q, K)
        if ctx["cache"] is None or ctx["cache"][0] != key:            # same for every layer: build once
            pads = ctx["pads"].to(m.device).float()                      # (B,)
            qa = torch.arange(K - Q, K, device=m.device, dtype=torch.float32)
            real = qa[None, :] - pads[:, None]                           # (B, Q) position in the conversation
            s1 = float(ctx["s1"])
            if ctx["kind"] == "log":
                val = ctx["s"] * torch.log((real.clamp(min=0) + 1) / s1)
            else:
                val = torch.full_like(real, ctx["s"])
            bq = torch.where(real >= s1, val, torch.zeros_like(val))      # (B, Q)
            kk = torch.arange(K, device=m.device, dtype=torch.float32)[None, :]
            span = ((kk >= pads[:, None] + ctx["s0"]) & (kk < pads[:, None] + ctx["s1"])).float()  # (B, K)
            ctx["cache"] = (key, (bq[:, :, None] * span[:, None, :])[:, None])                    # (B,1,Q,K)
        kwargs["attention_mask"] = (m.float() + ctx["cache"][1]).to(m.dtype)
        return args, kwargs
    mods = [ly.self_attn if hasattr(ly, "self_attn") else ly.attention for ly in model.base_model.layers]
    handles = [m_.register_forward_pre_hook(hook, with_kwargs=True) for m_ in mods]

    def encode_conv(msgs):
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False).input_ids

    conds = [c.split(":") for c in a.conditions.split(",")]
    topics = TOPICS[:a.n_topics]
    os.makedirs(a.out, exist_ok=True)
    res, mean_bias, saved = {}, {}, []
    for cond in conds:
        kind = cond[0]
        if kind == "none":
            name, s = "none", 0.0
        elif kind == "log":
            s = float(cond[1]); name = f"log:{s:g}"
        else:                                                            # const:match:s
            ref = f"log:{float(cond[2]):g}"
            if ref not in mean_bias:
                sys.exit(f"{':'.join(cond)} needs {ref} earlier in --conditions")
            s = mean_bias[ref]; name = f"const:{s:.2f}"
            kind = "const"
        fr = np.full((len(topics) * a.samples, a.turns), np.nan)
        lk = np.zeros_like(fr); nw = np.zeros_like(fr); ln = np.zeros_like(fr); rp = np.zeros_like(fr)
        bias_sum, bias_n = 0.0, 0
        for ti, topic in enumerate(topics):
            convs = [[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER.format(topic=topic)}] for _ in range(a.samples)]
            torch.manual_seed(a.seed + 1000 * ti)                         # same seeds across conditions
            for turn in range(a.turns):
                ids_list = [encode_conv(c) for c in convs]
                s1 = ids_list[0].index(im_end)
                s0 = len(head) if ids_list[0][:len(head)] == head else 0
                Lmax = max(len(x) for x in ids_list)
                pads = [Lmax - len(x) for x in ids_list]
                x = torch.tensor([[pad_id] * pd + ids for pd, ids in zip(pads, ids_list)], device=dev)
                att = torch.tensor([[0] * pd + [1] * len(ids) for pd, ids in zip(pads, ids_list)], device=dev)
                ctx.update(kind=kind, s=s, s0=s0, s1=s1, pads=torch.tensor(pads), cache=None)
                with torch.no_grad():
                    y = model.generate(x, attention_mask=att, do_sample=True, temperature=a.temp, top_k=0, top_p=1.0,
                                       max_new_tokens=a.turn_max, pad_token_id=pad_id,
                                       bad_words_ids=[[im_start]] if im_start is not None else None)
                ctx["cache"] = None
                for b in range(a.samples):
                    gen = y[b, Lmax:].tolist()
                    cut = next((i for i, t_ in enumerate(gen) if t_ in eos_ids), len(gen))
                    gen = gen[:cut]
                    reply = tok.decode(gen, skip_special_tokens=True)
                    convs[b].append({"role": "assistant", "content": reply})
                    convs[b].append({"role": "user", "content": FOLLOW[turn % len(FOLLOW)]})
                    r = ti * a.samples + b
                    fr[r, turn] = fr_share(reply); lk[r, turn] = leak_count(reply)
                    nw[r, turn] = max(1, len(re.findall(r"\w+", reply))); ln[r, turn] = len(gen)
                    rp[r, turn] = rep4(gen)
                    if kind == "log":                                     # bias this condition actually applied
                        pos = np.arange(len(ids_list[b]), len(ids_list[b]) + len(gen))
                        pos = pos[pos >= s1]
                        bias_sum += float(np.sum(s * np.log((pos + 1) / s1))); bias_n += len(pos)
                if dev == "mps": torch.mps.empty_cache()
                print(f"[{name}] topic {ti} turn {turn} done", flush=True)
            for b in range(min(2, a.samples)):
                saved.append(dict(condition=name, topic=topic,
                                  turns=[m_["content"] for m_ in convs[b] if m_["role"] == "assistant"]))
        if kind == "log":
            mean_bias[name] = bias_sum / max(1, bias_n)
        below = np.nan_to_num(fr, nan=1.0) < 0.5
        first = np.where(below.any(1), below.argmax(1) + 1, a.turns + 1)
        res[name] = dict(fr_turn=np.nanmean(fr, 0).tolist(), leak_turn=(1000 * lk.sum(0) / nw.sum(0)).tolist(),
                         len_turn=ln.mean(0).tolist(), last3=float(np.nanmean(fr[:, -3:])),
                         stay=float((~below).all(1).mean()), first_drop=float(first.mean()),
                         leak_1k=float(1000 * lk.sum() / nw.sum()), mean_len=float(ln.mean()),
                         rep4=float(rp.mean()), mean_bias=mean_bias.get(name, s if kind == "const" else 0.0))
    for h in handles:
        h.remove()

    names = list(res)
    T = a.turns
    print(f"\nmodel={a.model}  topics={len(topics)} x samples={a.samples}  turns={T}  temp={a.temp}")
    for title, key, fmt in [("French share of the reply, per turn", "fr_turn", "{:>7.2f}"),
                            ("instruction echo (leak words per 1000), per turn", "leak_turn", "{:>7.1f}"),
                            ("reply length (tokens), per turn", "len_turn", "{:>7.0f}")]:
        print(f"\n{title}")
        hdr = f"{'condition':>12s}" + "".join(f"{'t' + str(i + 1):>7s}" for i in range(T))
        print(hdr); print("-" * len(hdr))
        for n in names:
            print(f"{n:>12s}" + "".join(fmt.format(v) for v in res[n][key]))
    print(f"\n{'condition':>12s}{'bias':>7s}{'last3':>8s}{'stay':>7s}{'1st drop':>10s}{'leak/1k':>9s}{'len':>7s}{'rep4':>7s}")
    for n in names:
        r = res[n]
        print(f"{n:>12s}{r['mean_bias']:>7.2f}{r['last3']:>8.2f}{r['stay']:>7.2f}{r['first_drop']:>10.2f}"
              f"{r['leak_1k']:>9.2f}{r['mean_len']:>7.0f}{r['rep4']:>7.3f}")
    print("\nbias = mean bias applied to generated tokens; last3 = French share over the last 3 turns;")
    print(f"stay = every turn >= 0.5 French; 1st drop = first turn below 0.5 ({T + 1} if never).")
    json.dump(res, open(os.path.join(a.out, "multiturn.json"), "w"), indent=1)
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for r in saved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", a.out + "/multiturn.json and samples.jsonl")


if __name__ == "__main__":
    main()
