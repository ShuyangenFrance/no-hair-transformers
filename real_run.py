#!/usr/bin/env python3
"""
real_run.py — teacher-forced alpha on prompt pairs drawn from real corpora (no hand-written scenes).

Axes (each with its own pairs, n_pairs per axis):
  detail    : a WikiText-103 *test* sentence vs. the same sentence with one content word swapped
  para      : MRPC sentence pairs labelled as paraphrases (GLUE)
  lang      : WMT14 English sentence vs. its French translation (validation split)
  unrelated : two random WikiText test sentences (reference: completely different prompts)
Shared continuations: WikiText-103 *validation* chunks (disjoint from the prompt source).
Uncertainty: bootstrap over prompt pairs.

  python real_run.py --model EleutherAI/pythia-160m --revision step0      --out real/p160_step0
  python real_run.py --model EleutherAI/pythia-160m --revision step143000 --out real/p160_final
  python real_run.py ... --n_pairs 10 --m 4                                 (quick check)
  python real_run.py --out real/p160_final --analyze_only                    (re-analyse saved curves)
Needs: pip install datasets
"""
import argparse, json, os, re, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nohair_gen_v2 import load_model, EN_WORDS   # noqa: E402
from tf_controls import human_chunks             # noqa: E402

AXES = ["detail", "para", "lang", "unrelated"]
STOP = EN_WORDS | set("which there their about would could should these those other after before into "
                      "over under while where when what also than then them were been being have this "
                      "that with from they will your more most some such only very just near upon until within without "
                      "between during against through around among along across behind below above since "
                      "though although because unless whether either neither each every many much both "
                      "first second third later early also again still even ever often".split())


def sentences_wikitext_test():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    out = []
    for line in ds["text"]:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            if 12 <= len(s.split()) <= 35 and "@" not in s:
                out.append(s)
    return out


def swap_one_word(s, vocab, rng):
    """replace one content word by another of similar length and the same capitalisation."""
    words = s.split(" ")
    cand = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 4 and w.lower() not in STOP]
    if not cand:
        return None
    i = cand[rng.integers(len(cand))]
    w = words[i]
    pool = [v for v in vocab if abs(len(v) - len(w)) <= 2 and v.lower() != w.lower()]
    if not pool:
        return None
    v = pool[rng.integers(len(pool))]
    v = v.capitalize() if w[0].isupper() else v.lower()
    words[i] = v
    return " ".join(words)


def shuffled_chunks(tok, n, L, seed):
    """n continuations of L tokens made of WikiText-103 validation sentences in random order:
    no headings, no document boundaries, no long-range structure beyond a sentence."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    sents = []
    for line in ds["text"]:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        sents += [x for x in re.split(r"(?<=[.!?])\s+", line) if len(x.split()) >= 5]
    rng = np.random.default_rng(seed + 12345)
    order = rng.permutation(len(sents))
    ids, out, k = [], [], 0
    while len(out) < n:
        ids += tok(" " + sents[order[k % len(order)]], add_special_tokens=False).input_ids
        k += 1
        if len(ids) >= L:
            out.append(ids[:L]); ids = []
    print(f"shuffled-sentence continuations: {n} x {L} tokens from {len(sents)} WikiText sentences", flush=True)
    return np.array(out)


def build_pairs(n, rng):
    from datasets import load_dataset
    wiki = sentences_wikitext_test()
    vocab = sorted({w for s in wiki for w in s.split(" ") if w.isalpha() and len(w) >= 4
                    and w.lower() not in STOP and w.islower()})
    pairs = {}
    idx = rng.permutation(len(wiki))
    det = []
    for i in idx:
        t = swap_one_word(wiki[i], vocab, rng)
        if t:
            det.append((wiki[i], t))
        if len(det) == n:
            break
    pairs["detail"] = det
    mrpc = load_dataset("nyu-mll/glue", "mrpc", split="train")
    mp = [(r["sentence1"], r["sentence2"]) for r in mrpc if r["label"] == 1]
    pairs["para"] = [mp[i] for i in rng.permutation(len(mp))[:n]]
    wmt = load_dataset("wmt/wmt14", "fr-en", split="validation")
    wp = [(r["translation"]["en"], r["translation"]["fr"]) for r in wmt
          if 12 <= len(r["translation"]["en"].split()) <= 35]
    pairs["lang"] = [wp[i] for i in rng.permutation(len(wp))[:n]]
    j = rng.permutation(len(wiki))
    pairs["unrelated"] = [(wiki[j[2 * k]], wiki[j[2 * k + 1]]) for k in range(n)]
    return pairs


def fit_alpha(curve, lt):
    return -np.polyfit(lt, np.log10(np.maximum(curve, 1e-30)), 1)[0] / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--revision", default=None)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n_pairs", type=int, default=100, help="prompt pairs per axis")
    p.add_argument("--m", type=int, default=8, help="shared continuations per pair")
    p.add_argument("--windows", default="16,32,64,128,256,512,1024")
    p.add_argument("--win", type=int, default=16)
    p.add_argument("--layers", default="", help="default: 1,2,3,~L/4,~L/2,~3L/4,L-1")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--estimator", choices=["unbiased", "plugin"], default="unbiased",
                   help="signal estimator: plug-in (mean offset)^2, or unbiased (subtract Var/(n_c) with ddof=1)")
    p.add_argument("--label", choices=["center", "end"], default="center",
                   help="time assigned to each window: its centre t-(win-1)/2 (default) or its end t")
    p.add_argument("--cont", choices=["wiki", "shuffled"], default="wiki",
                   help="continuations: contiguous WikiText chunks (default) or shuffled WikiText sentences")
    p.add_argument("--prefix", default="", help="text prepended to every prompt, e.g. '<|endoftext|>' to give "
                   "both prompts of a pair the same first token (shared attention sink)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = [int(x) for x in a.windows.split(",")]
    npz = os.path.join(a.out, "real.npz")

    if not a.analyze_only:
        rng = np.random.default_rng(a.seed)
        pairs = build_pairs(a.n_pairs, rng)
        json.dump(pairs, open(os.path.join(a.out, "pairs.json"), "w"), ensure_ascii=False, indent=0)
        la = types.SimpleNamespace(smoke=False, model=a.model, random_init=a.random_init,
                                   revision=a.revision, seed=a.seed)
        torch, model, tok, encode, _, _, dev, empty_cache = load_model(la)
        L = model.config.num_hidden_layers
        layers = [int(x) for x in a.layers.split(",")] if a.layers else \
            sorted(set([1, 2, 3] + [max(1, round(L * f)) for f in (0.25, 0.5, 0.75)] + [L - 1]))
        Tmax = max(W)
        pool = human_chunks(tok, 64, Tmax, "", model.config.vocab_size, a.seed) if a.cont == "wiki" \
            else shuffled_chunks(tok, 64, Tmax, a.seed)
        maxpos = getattr(model.config, "max_position_embeddings", 10 ** 9)

        def hid(ids, c):
            P = len(ids)
            x = torch.tensor([ids + c[:Tmax].tolist()], device=dev)
            with torch.no_grad():
                hs = model.base_model(x, output_hidden_states=True).hidden_states
            h = torch.stack([torch.stack([hs[l][0, P + t - a.win:P + t].float().mean(0) for l in layers])
                             for t in W]).cpu().numpy().astype(np.float64)
            del hs
            empty_cache()
            return h

        sig = np.zeros((len(AXES), a.n_pairs, len(layers), len(W)))
        noi = np.zeros_like(sig); flo = np.zeros_like(sig); plen = np.zeros((len(AXES), a.n_pairs, 2))
        for ai, ax in enumerate(AXES):
            for pi, (s1, s2) in enumerate(pairs[ax]):
                i1, i2 = encode(a.prefix + s1), encode(a.prefix + s2)
                if max(len(i1), len(i2)) + Tmax > maxpos:
                    sys.exit("prompt + window exceeds the context; use shorter --windows")
                plen[ai, pi] = len(i1), len(i2)
                C = pool[rng.choice(len(pool), a.m, replace=False)]
                HA = np.array([hid(i1, c) for c in C]); HB = np.array([hid(i2, c) for c in C])
                for li in range(len(layers)):
                    A_, B_ = HA[:, :, li], HB[:, :, li]
                    var = 0.5 * (A_.var(0) + B_.var(0))
                    s0 = np.sqrt(var[0]) + 1e-8
                    sig[ai, pi, li] = (((B_ - A_).mean(0) / s0) ** 2).mean(-1)
                    noi[ai, pi, li] = (var / s0 ** 2).mean(-1)
                    flo[ai, pi, li] = ((B_ - A_).var(0) / len(C) / s0 ** 2).mean(-1)
                if (pi + 1) % 10 == 0:
                    print(f"[{ax}] {pi + 1}/{a.n_pairs} pairs", flush=True)
        name = a.model + (f"@{a.revision}" if a.revision else "") + (" [random init]" if a.random_init else "")
        np.savez_compressed(npz, sig=sig, noi=noi, flo=flo, plen=plen, windows=np.array(W),
                            layers=np.array(layers), model=name, m=a.m)

    D = np.load(npz)
    W = [int(x) for x in D["windows"]]
    tlab = np.array(W, float) - ((a.win - 1) / 2 if a.label == "center" else 0.0)
    lt = np.log10(tlab)
    sig, noi, flo = D["sig"], D["noi"], D["flo"]
    mC = int(D["m"])
    msq = sig + flo * mC            # E_c ||Delta||^2 (normalised): (mean)^2 + var, does not cancel varying directions
    if a.estimator == "unbiased":           # (mean)^2 - s^2/n with s^2 the ddof=1 variance = plug-in - flo*n/(n-1)
        sig = sig - flo * mC / (mC - 1)
        neg = (sig <= 0).mean()
        if neg > 0:
            print(f"note: {100 * neg:.1f}% of per-pair signal values are <= 0 after bias removal (kept; curves are averaged first)")
    layers, name, plen = [int(x) for x in D["layers"]], str(D["model"]), D["plen"]
    n = sig.shape[1]
    rng = np.random.default_rng(a.seed)
    print(f"\nmodel={name}  pairs per axis={n}  continuations per pair={int(D['m'])}  windows={W}  "
          f"(window width {a.win}, labelled by its {a.label}, {a.estimator} signal)")
    print("mean prompt length (tokens): " + ", ".join(f"{ax} {plen[i].mean():.0f}" for i, ax in enumerate(AXES)))

    res = {}
    print("\nalpha per axis and layer, 95% bootstrap interval over prompt pairs; beta in the last column (pooled)")
    print(f"{'layer':>6s}" + "".join(f"{ax:>20s}" for ax in AXES) + f"{'beta':>7s}")
    for li, l in enumerate(layers):
        cells = []
        for ai, ax in enumerate(AXES):
            a0 = fit_alpha(sig[ai, :, li].mean(0), lt)
            bs = [fit_alpha(sig[ai, rng.integers(0, n, n), li].mean(0), lt) for _ in range(a.boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[f"{ax}|{l}"] = (a0, lo, hi)
            cells.append(f"{a0:.2f} [{lo:.2f},{hi:.2f}]")
        b0 = fit_alpha(noi[:, :, li].mean((0, 1)), lt)
        print(f"{'L' + str(l):>6s}" + "".join(f"{c:>20s}" for c in cells) + f"{b0:>7.2f}")

    print("\nexponent of the mean squared offset E_c|Delta|^2 (no cancellation across continuations), axes pooled")
    print(f"{'layer':>6s}{'alpha_S':>9s}{'alpha_msq':>11s}   local alpha_msq per octave")
    for li, l in enumerate(layers):
        cm = np.log10(msq[:, :, li].mean((0, 1)))
        locm = [-(cm[i + 1] - cm[i]) / (lt[i + 1] - lt[i]) / 2 for i in range(len(W) - 1)]
        print(f"{'L' + str(l):>6s}{fit_alpha(sig[:, :, li].mean((0, 1)), lt):>9.2f}{fit_alpha(msq[:, :, li].mean((0, 1)), lt):>11.2f}   "
              + " ".join(f"{v:.2f}" for v in locm))

    print("\nlocal alpha per octave, pooled over the detail/para/lang axes; '*' = signal < 3x sampling floor")
    hdr = f"{'layer':>6s}" + "".join(f"{f'{W[i]}-{W[i + 1]}':>11s}" for i in range(len(W) - 1))
    print(hdr); print("-" * len(hdr))
    local = {}
    for li, l in enumerate(layers):
        cs = sig[:3, :, li].mean((0, 1)); fs = flo[:3, :, li].mean((0, 1))
        c = np.log10(cs)
        loc = [-(c[i + 1] - c[i]) / (lt[i + 1] - lt[i]) / 2 for i in range(len(W) - 1)]
        local[l] = loc
        print(f"{'L' + str(l):>6s}" + "".join(f"{v:>10.2f}{'*' if cs[i + 1] < 3 * fs[i + 1] else ' '}"
                                             for i, v in enumerate(loc)))
    # fits of the local exponents (untrained regime): pure log power and small-gain (Bessel) form
    P = float(plen[:3].mean())
    tm = np.sqrt(tlab[:-1] * tlab[1:])                         # geometric midpoint of each octave
    print(f"\nfits of the local exponents (P = {P:.0f}, octave midpoints)")
    print(f"{'layer':>6s}{'k (log power)':>15s}{'R2':>7s}{'G (Bessel)':>12s}{'R2':>7s}")
    fits = {}
    for l in layers:
        oma = 1 - np.array(local[l])
        if np.any(oma <= 0):
            print(f"{'L' + str(l):>6s}   (local exponent >= 1 somewhere: not in the untrained regime)"); continue
        y1, x1 = 1 / oma, np.log(tm); b1, a1 = np.polyfit(x1, y1, 1)
        r1 = 1 - np.var(y1 - (a1 + b1 * x1)) / np.var(y1)
        y2, x2 = 1 / oma ** 2, np.log((P + tm) / P); b2, a2 = np.polyfit(x2, y2, 1)
        r2 = 1 - np.var(y2 - (a2 + b2 * x2)) / np.var(y2)
        fits[l] = dict(k=1 / b1, r2_k=r1, G=1 / b2, r2_G=r2)
        print(f"{'L' + str(l):>6s}{1 / b1:>15.2f}{r1:>7.3f}{1 / b2:>12.2f}{r2:>7.3f}")
    json.dump(dict(model=name, layers=layers, windows=W, label=a.label, win=a.win,
                   alpha={k: list(v) for k, v in res.items()},
                   local={str(k): v for k, v in local.items()}, fits={str(k): v for k, v in fits.items()}),
              open(os.path.join(a.out, "real.json"), "w"), indent=1, default=float)
    print("saved", npz, "real.json and pairs.json")


if __name__ == "__main__":
    main()
