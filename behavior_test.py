#!/usr/bin/env python3
"""
behavior_test.py — does compensating attention dilution keep an instruction alive in long generations?

Task: the system prompt says "answer only in French"; the user asks, in English, for a very long story.
We generate N tokens and track, window by window, the share of French in the output.
Conditions (bias on every attention logit from a later position q to the system-prompt tokens,
S = end of the system prompt, all layers and heads, during generation incl. KV-cache steps):
  none        unmodified model
  log:s       + s * log((q+1)/S)          the dilution compensation
  const:c     + c for every q >= S        a position-independent boost (split-softmax-like control)
  const:auto  + c chosen as the mean over the generation of the bias of log:<--const_match, default 0.5>,
              so both spend the same total boost; if log beats const, the log t shape is what matters

Reported per condition:
  French share per window (function-word based, as in nohair_gen_v2.py langswitch)
  stay   : fraction of samples whose every window is >= 0.5 French
  first drop : mean generation step of the first window below 0.5 (samples that never drop count as N)
  rep4   : share of repeated 4-grams in the second half of the output (degeneration check)
  nll    : mean NLL of the output under the UNMODIFIED model (fluency check; lower is not always better,
           loops have low NLL, so read it together with rep4)
  leak   : instruction echo. Share of samples whose 2nd half mentions the instruction itself (French/English/
           language/respond..., also 法语/英语), and such mentions per 1000 words

  python behavior_test.py                      # Qwen2.5-1.5B-Instruct, 4 topics x 8 samples x 2048 tokens
  python behavior_test.py --model Qwen/Qwen2.5-0.5B-Instruct
  python behavior_test.py --quick              # 2 topics x 2 samples x 1024 tokens, to check it runs
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import FR_WORDS, EN_WORDS   # noqa: E402

SYSTEM = "You must respond only in French, whatever language the user writes in. Never switch to English."
TOPICS = [
    "a lighthouse keeper who finds a message in a bottle",
    "two rival bakers in a small mountain village",
    "a young astronomer who discovers a new comet",
    "a stray dog that travels across an entire country",
    "an old violin passed down through four generations",
    "a librarian who can enter the books she reads",
]
USER = ("Write a very long and detailed story about {topic}. Make it at least 3000 words, "
        "with many chapters, rich dialogue and detailed descriptions.")


def fr_share(text):
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœæ]+", text.lower())
    fr = sum(w in FR_WORDS for w in words); en = sum(w in EN_WORDS for w in words)
    return fr / (fr + en) if fr + en else np.nan


LEAK = re.compile(r"\b(fran[cç]ais[e]?|french|anglais[e]?|english|langue|language|r[ée]pond\w*|respond\w*|reply)\b"
                  r"|法语|英语|语言", re.I)


def leak_count(text):
    return len(LEAK.findall(text))


def rep4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(g)) / max(1, len(g))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--conditions", default="none,log:0.25,log:0.5,log:0.75,const:auto")
    p.add_argument("--n_topics", type=int, default=4)
    p.add_argument("--const_match", type=float, default=0.5,
                   help="const:auto spends the same mean bias as log:<this value>")
    p.add_argument("--samples", type=int, default=8, help="samples per topic (generated as one batch)")
    p.add_argument("--N", type=int, default=2048, help="generated tokens")
    p.add_argument("--win", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="behavior")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.n_topics, a.samples, a.N = 2, 2, 1024

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else \
        ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="eager").to(dev).eval()
    H = model.config.num_attention_heads
    print("device:", dev, flush=True)

    # ---- the boost, applied through pre-hooks so it also acts on cached generation steps
    ctx = {"kind": "none", "s": 0.0, "s0": 0, "s1": 0}
    def hook(mod, args, kwargs):
        if ctx["kind"] == "none" or ctx["s"] == 0:
            return None
        m = kwargs.get("attention_mask")
        if m is None or m.dim() != 4 or not m.is_floating_point():
            raise RuntimeError("expected a 4D additive attention mask (eager attention)")
        Q, K = m.shape[-2], m.shape[-1]
        q = torch.arange(K - Q, K, dtype=torch.float32, device=m.device)       # absolute query positions
        s0, s1 = ctx["s0"], ctx["s1"]
        val = ctx["s"] * torch.log((q + 1) / s1) if ctx["kind"] == "log" else torch.full_like(q, ctx["s"])
        bias = torch.where(q >= s1, val, torch.zeros_like(q))
        B = torch.zeros((Q, K), dtype=torch.float32, device=m.device)
        B[:, s0:s1] = bias[:, None]
        kwargs["attention_mask"] = (m.float() + B).to(m.dtype)
        return args, kwargs
    mods = [ly.self_attn if hasattr(ly, "self_attn") else ly.attention for ly in model.base_model.layers]
    handles = [m_.register_forward_pre_hook(hook, with_kwargs=True) for m_ in mods]

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    bad = [[im_start]] if im_start is not None and im_start != tok.unk_token_id else None

    def build(topic):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(topic=topic)}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
        s1 = ids.index(im_end)                                   # end of the system message
        head = tok("<|im_start|>system\n", add_special_tokens=False).input_ids
        s0 = len(head) if ids[:len(head)] == head else 0
        return ids, s0, s1

    topics = TOPICS[:a.n_topics]
    # parse conditions; const:auto = mean log-bias of the strongest log condition over the generated positions
    conds = []
    for c in a.conditions.split(","):
        kind, _, v = c.partition(":")
        conds.append([kind, v])
    s_max = a.const_match
    ids0, _, s1_0 = build(topics[0])
    qs = np.arange(len(ids0), len(ids0) + a.N)
    auto_c = float(np.mean(s_max * np.log((qs + 1) / s1_0)))
    conds = [(k, auto_c if v == "auto" else (float(v) if v else 0.0)) for k, v in conds]
    names = [k if k == "none" else f"{k}:{v:.2f}" for k, v in conds]
    print("conditions:", ", ".join(names), flush=True)
    nw = a.N // a.win
    os.makedirs(a.out, exist_ok=True)
    res, samples_txt = {}, []
    for (kind, s), name in zip(conds, names):
        shares, outs, texts = [], [], []
        for ti, topic in enumerate(topics):
            ids, s0, s1 = build(topic)
            ctx.update(kind=kind, s=s, s0=s0, s1=s1)
            torch.manual_seed(a.seed + 1000 * ti)                 # same seeds across strengths
            x = torch.tensor([ids] * a.samples, device=dev)
            with torch.no_grad():
                y = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=a.temp,
                                   top_k=0, top_p=1.0, max_new_tokens=a.N, min_new_tokens=a.N,
                                   bad_words_ids=bad, pad_token_id=tok.eos_token_id)
            for row in y[:, len(ids):].cpu().numpy():
                row = row[:a.N]
                outs.append(row)
                texts.append(tok.decode(row.tolist()))
                shares.append([fr_share(tok.decode(row[w * a.win:(w + 1) * a.win].tolist())) for w in range(nw)])
            for j in range(min(2, a.samples)):
                samples_txt.append(dict(condition=name, topic=topic, text=texts[-a.samples + j]))
            if dev == "mps": torch.mps.empty_cache()
            print(f"[{name}] topic {ti} done", flush=True)
        # fluency check under the UNMODIFIED model
        ctx["kind"] = "none"
        nlls = []
        for ti, topic in enumerate(topics):
            ids, _, _ = build(topic)
            for row in outs[ti * a.samples:(ti + 1) * a.samples]:
                x = torch.tensor([ids + row.tolist()], device=dev)
                tgt = x[0, len(ids):]
                with torch.no_grad():   # LM head in chunks: full-vocab logits for 2k tokens would be > 1 GB
                    h = model.base_model(x).last_hidden_state[0, len(ids) - 1:-1]
                    tot = 0.0
                    for c0 in range(0, len(tgt), 256):
                        lp = torch.log_softmax(model.lm_head(h[c0:c0 + 256]).float(), -1)
                        tot += -lp.gather(-1, tgt[c0:c0 + 256][:, None]).sum().item()
                nlls.append(tot / len(tgt))
                del h, lp
        sh = np.array(shares, float)                               # (n_samples, n_windows)
        below = np.nan_to_num(sh, nan=1.0) < 0.5
        first = np.where(below.any(1), below.argmax(1) * a.win, a.N)
        halves = [t_[len(t_) // 2:] for t_ in texts]
        lc = [leak_count(h_) for h_ in halves]
        nwords = [max(1, len(re.findall(r"\w+", h_))) for h_ in halves]
        res[name] = dict(curve=np.nanmean(sh, 0).tolist(), stay=float((~below).all(1).mean()),
                         first_drop=float(first.mean()), rep4=float(np.mean([rep4(o[a.N // 2:]) for o in outs])),
                         nll=float(np.mean(nlls)), leak=float(np.mean([c_ > 0 for c_ in lc])),
                         leak_per_1k=float(1000 * np.sum(lc) / np.sum(nwords)), n=len(outs))
    for h in handles:
        h.remove()

    print(f"\nmodel={a.model}  topics={len(topics)} x samples={a.samples}  N={a.N}  temp={a.temp}")
    print("French share per window (window end, tokens):")
    hdr = f"{'condition':>12s}" + "".join(f"{(w + 1) * a.win:>7d}" for w in range(nw))
    print(hdr); print("-" * len(hdr))
    for name in names:
        print(f"{name:>12s}" + "".join(f"{v:>7.2f}" for v in res[name]["curve"]))
    print(f"\n{'condition':>12s}{'stay':>8s}{'first drop':>12s}{'rep4':>8s}{'nll':>8s}{'leak':>8s}{'leak/1k':>9s}")
    for name in names:
        r = res[name]
        print(f"{name:>12s}{r['stay']:>8.2f}{r['first_drop']:>12.0f}{r['rep4']:>8.3f}{r['nll']:>8.3f}"
              f"{r['leak']:>8.2f}{r['leak_per_1k']:>9.2f}")
    print("\nstay = every window >= 0.5 French; first drop = mean step of the first window below 0.5 (N if never);")
    print("rep4 = repeated 4-gram share in the 2nd half; nll = mean NLL under the unmodified model;")
    print("leak = share of samples whose 2nd half talks about the instruction/language; leak/1k = such words per 1000.")
    print(f"const:auto used c = {auto_c:.2f} (mean bias of log:{s_max} over the generation)")
    json.dump(res, open(os.path.join(a.out, "behavior.json"), "w"), indent=1)
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for r in samples_txt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved", a.out + "/behavior.json and samples.jsonl")


if __name__ == "__main__":
    main()