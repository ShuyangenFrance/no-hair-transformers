#!/usr/bin/env python3
"""
nohair_gen_v1.py — Transformer "no-hair" experiment, small scale.

Time      = generation step t (free autoregressive evolution, no external input after the prompt).
Exterior  = statistics of the generated text in a window ending at t (definition 1).
Interior  = hidden states averaged over the same window (definition 2).
Metric    = decodability: can a linear probe tell which side of a prompt pair a trajectory came from?

Two stages, so the analysis can be rerun without regenerating:
  python nohair_gen_v1.py gen     --out run1            # needs GPU ideally
  python nohair_gen_v1.py analyze --out run1            # CPU, numpy + sklearn only

Smoke test of the whole pipeline with a tiny random Qwen2 (no download):
  python nohair_gen_v1.py gen --smoke --out smoke && python nohair_gen_v1.py analyze --out smoke
"""
import argparse, json, os, zlib
import numpy as np

# ---------------------------------------------------------------- prompts
# Each scene: a base prompt and one variant per axis. A pair for axis a is (base, scene[a]).
SCENES = [
    dict(base="巴黎今天下雨了，我站在地铁口，",
         detail="巴黎今天下雪了，我站在地铁口，",
         para="今天巴黎落着雨，我在地铁站门口停下来，",
         lang="It was raining in Paris today. I stood at the metro entrance,",
         genre="【本报讯】巴黎今日降雨，记者在地铁站口看到，"),
    dict(base="外婆家的厨房里，雨水顺着窗户往下流，她",
         detail="外婆家的厨房里，雪花落在窗台上，她",
         para="雨沿着外婆厨房的玻璃窗一道道淌下来，她",
         lang="In my grandmother's kitchen, rain ran down the window as she",
         genre="【本报讯】受连日降雨影响，本市多户居民厨房窗户渗水，一位老人"),
    dict(base="那天下着雨，他在火车站等了很久，终于",
         detail="那天下着雪，他在火车站等了很久，终于",
         para="雨一直没停，他在车站里等了很久，最后",
         lang="It was raining that day. He waited at the train station for a long time, and finally",
         genre="【本报讯】昨日降雨导致多趟列车晚点，一名旅客在车站等候多时后"),
]
AXES = ["detail", "para", "lang", "genre"]


# ================================================================ stage 1: generation
def stage_gen(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else \
        ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print("device:", dev, flush=True)

    def empty_cache():
        if dev == "cuda": torch.cuda.empty_cache()
        elif dev == "mps": torch.mps.empty_cache()

    if a.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
        torch.manual_seed(0)
        model, tok = Qwen2ForCausalLM(cfg), None
        encode = lambda s: [ord(c) % 512 for c in s]
        pad_id, eos_id = 0, None
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        tok = AutoTokenizer.from_pretrained(a.model)
        if a.random_init:   # same architecture and tokenizer, untrained weights: the "no learning" baseline
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(a.model)).to(dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)
        encode = lambda s: tok(s, add_special_tokens=False).input_ids
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id
    model = model.to(dev).eval()

    L = model.config.num_hidden_layers
    # hidden_states[0] = embeddings, hidden_states[L] = after the final norm -> use L-1 as "last block"
    layers = [int(x) for x in a.layers.split(",")] if a.layers else \
        sorted(set([max(1, round(L * f)) for f in (0.25, 0.5, 0.75)] + [L - 1]))
    windows = [int(x) for x in a.windows.split(",")]
    assert max(windows) <= a.T and min(windows) >= a.win, "windows must lie inside [win, T]"

    def generate(ids, n, seed):
        out = []
        for s in range(0, n, a.batch):
            b = min(a.batch, n - s)
            torch.manual_seed(seed + s)
            x = torch.tensor([ids] * b, device=dev)
            kw = dict(do_sample=True, temperature=a.temp, top_k=0, top_p=1.0, repetition_penalty=1.0,
                      max_new_tokens=a.T, pad_token_id=pad_id)
            if eos_id is not None:
                kw["min_new_tokens"] = a.T  # suppress EOS so every trajectory has length T
            with torch.no_grad():
                y = model.generate(x, attention_mask=torch.ones_like(x), **kw)
            out.append(y[:, len(ids):].cpu())
            del y
            empty_cache()
        return torch.cat(out).numpy().astype(np.int64)

    def hidden(ids, conts):
        """Window-averaged hidden states, shape (n, n_win, n_layers, d). Positions are counted
        from the start of the continuation, so prompts of different length share the same clock."""
        P, res = len(ids), []
        for s in range(0, len(conts), a.hbatch):
            c = torch.tensor(conts[s:s + a.hbatch], device=dev)
            x = torch.cat([torch.tensor([ids] * len(c), device=dev), c], 1)
            with torch.no_grad():
                # base_model = transformer without the LM head: avoids materialising the
                # (batch x length x vocab) logits, which is ~10 GB for Qwen at batch 32
                hs = model.base_model(x, output_hidden_states=True).hidden_states
            f = torch.stack([torch.stack([hs[l][:, P + t - a.win:P + t].float().mean(1) for l in layers], 1)
                             for t in windows], 1)
            res.append(f.cpu().numpy().astype(np.float32))
            del hs, f
            empty_cache()
        return np.concatenate(res)

    enc = None
    if a.encoder and not a.smoke:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(a.encoder, device=dev)

    os.makedirs(a.out, exist_ok=True)
    data, samples = {}, []
    for si, sc in enumerate(SCENES[:a.n_scenes]):
        ids = {k: encode(v) for k, v in sc.items()}
        conts = {}
        for k in ["base"] + AXES:
            n = 2 * a.n if k == "base" else a.n          # base gets 2n for the noise-floor split
            seed = a.seed + 100003 * si + zlib.crc32(k.encode()) % 10007 * 1000
            conts[k] = generate(ids[k], n, seed)
            data[f"conts|{si}|{k}"] = conts[k]
            data[f"hid|{si}|{k}"] = hidden(ids[k], conts[k])
            if enc is not None:
                texts = [tok.decode(c[t - a.win:t].tolist()) for c in conts[k] for t in windows]
                E = enc.encode(texts, batch_size=64, normalize_embeddings=True)
                data[f"enc|{si}|{k}"] = E.reshape(n, len(windows), -1).astype(np.float32)
            if tok is not None:
                for c in conts[k][:2]:
                    samples.append(dict(scene=si, prompt=k, text=sc[k] + tok.decode(c.tolist())))
            print(f"scene {si} {k:6s} done", flush=True)
        # teacher forcing: the SAME continuations fed after both prompts of each pair
        for ax in AXES:
            m = a.n // 2
            C = np.concatenate([conts["base"][:m], conts[ax][:m]])
            data[f"tfA|{si}|{ax}"] = hidden(ids["base"], C)
            data[f"tfB|{si}|{ax}"] = hidden(ids[ax], C)

    np.savez_compressed(os.path.join(a.out, "data.npz"), **data)
    meta = dict(model="smoke" if a.smoke else a.model + (" [random init]" if a.random_init else ""), temp=a.temp, T=a.T, n=a.n, win=a.win,
                windows=windows, layers=layers, n_scenes=min(a.n_scenes, len(SCENES)), axes=AXES,
                vocab=int(model.config.vocab_size), encoder=a.encoder if enc is not None else None)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("saved to", a.out)


# ================================================================ stage 2: analysis
def probe_acc(X, y, groups=None, k=5, seed=0, scale=True, ncomp=16):
    """Cross-validated accuracy of [standardize] -> PCA -> whiten -> L2 logistic regression.
    PCA keeps the probe from overfitting when d >> n."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    if groups is None:
        splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y)
    else:
        splits = GroupKFold(k).split(X, y, groups)
    accs = []
    for tr, te in splits:
        nc = max(1, min(ncomp, len(tr) - 1, X.shape[1]))
        steps = ([StandardScaler()] if scale else []) + \
                [PCA(nc, random_state=seed), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)]
        clf = make_pipeline(*steps).fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def unigram(conts, windows, win, D=4096):
    """Hashed token histogram of each window, Hellinger-transformed. Model-independent exterior view."""
    out = np.zeros((len(conts), len(windows), D), np.float32)
    for j, t in enumerate(windows):
        seg = conts[:, t - win:t] % D
        for i in range(len(conts)):
            out[i, j] = np.bincount(seg[i], minlength=D)
    return np.sqrt(out / win)


def stage_analyze(a):
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    W, win, layers, S, n = meta["windows"], meta["win"], meta["layers"], meta["n_scenes"], meta["n"]
    rng = np.random.default_rng(a.seed)

    # feature getters: kind -> (scene, prompt, window index) -> (n_samples, dim)
    cache = {}
    def feat(kind, si, k, j):
        if kind == "uni":
            key = ("uni", si, k)
            if key not in cache:
                cache[key] = unigram(Z[f"conts|{si}|{k}"], W, win)
            return cache[key][:, j]
        if kind == "enc":
            return Z[f"enc|{si}|{k}"][:, j]
        li = layers.index(int(kind[1:]))  # "L12"
        return Z[f"hid|{si}|{k}"][:, j, li]

    kinds = ["uni"] + (["enc"] if meta.get("encoder") else []) + [f"L{l}" for l in layers]
    scale = lambda kind: kind not in ("uni", "enc")
    res = {}

    # ---- free generation: exterior + interior
    for kind in kinds:
        for j, t in enumerate(W):
            # noise floor: split the 2n base trajectories into two random halves, n vs n
            nulls = []
            for si in range(S):
                Xb = feat(kind, si, "base", j)
                for r in range(a.n_null):
                    perm = rng.permutation(len(Xb))
                    y = np.zeros(len(Xb), int); y[perm[: len(Xb) // 2]] = 1
                    nulls.append(probe_acc(Xb, y, seed=r, scale=scale(kind)))
            res[("null", kind, t)] = (float(np.mean(nulls)), float(np.percentile(nulls, 95)))
            for ax in meta["axes"]:
                accs = []
                for si in range(S):
                    X = np.concatenate([feat(kind, si, "base", j)[:n], feat(kind, si, ax, j)])
                    y = np.r_[np.zeros(n, int), np.ones(n, int)]
                    accs.append(probe_acc(X, y, seed=0, scale=scale(kind)))
                res[(ax, kind, t)] = float(np.mean(accs))

    # ---- teacher forcing: interior only; same continuation after both prompts
    for l in layers:
        li = layers.index(l)
        for j, t in enumerate(W):
            nulls = []
            for ax in meta["axes"]:
                accs = []
                for si in range(S):
                    A = Z[f"tfA|{si}|{ax}"][:, j, li]; B = Z[f"tfB|{si}|{ax}"][:, j, li]
                    m = len(A)
                    X = np.concatenate([A, B]); g = np.r_[np.arange(m), np.arange(m)]
                    y = np.r_[np.zeros(m, int), np.ones(m, int)]
                    accs.append(probe_acc(X, y, groups=g))
                    # null: swap the two labels within each continuation pair at random
                    for r in range(max(1, a.n_null // len(meta["axes"]))):
                        flip = rng.random(m) < 0.5
                        yn = y.copy(); yn[:m][flip] = 1; yn[m:][flip] = 0
                        nulls.append(probe_acc(X, yn, groups=g))
                res[(ax, f"tf{l}", t)] = float(np.mean(accs))
            res[("null", f"tf{l}", t)] = (float(np.mean(nulls)), float(np.percentile(nulls, 95)))

    # ---- report
    tkinds = kinds + [f"tf{l}" for l in layers]
    label = {"uni": "ext:unigram", "enc": "ext:encoder"}
    print(f"\nmodel={meta['model']}  temp={meta['temp']}  scenes={S}  n/side={n}  win={win}")
    print("accuracy of telling the two prompts apart (chance 0.50); '*' = above the 95th pct of the noise floor\n")
    hdr = f"{'axis':8s}{'view':14s}" + "".join(f"t={t:<9d}" for t in W)
    print(hdr); print("-" * len(hdr))
    for ax in ["null"] + meta["axes"]:
        for kind in tkinds:
            name = label.get(kind, ("int:" if kind.startswith("L") else "tf:") + kind.lstrip("Ltf"))
            cells = []
            for t in W:
                if ax == "null":
                    mu, p95 = res[("null", kind, t)]; cells.append(f"{mu:.2f}/{p95:.2f} ")
                else:
                    v = res[(ax, kind, t)]; star = "*" if v > res[("null", kind, t)][1] else " "
                    cells.append(f"{v:.2f}{star}     ")
            print(f"{ax:8s}{name:14s}" + "".join(cells))
        print()
    print("null rows: mean/95th percentile.  ext = generated text, int = hidden states of free generation,")
    print("tf = hidden states with a shared (teacher-forced) continuation, i.e. pure memory of the prompt.")

    # ---- paired amplitude: a continuous, non-saturating measure of prompt memory under teacher forcing
    # For each shared continuation c: delta = h(x'+c) - h(x+c). The prompt's "hair amplitude" is the
    # consistent part of delta, measured per dimension against the spread over continuations:
    #     SNR = mean_j ( mean_c delta_j / sd_j )^2 ,  sd_j = std over continuations of h_j
    # Null: random sign flips of delta (no consistent offset). Unlike probe accuracy, this does not pin
    # at 1.0 or 0.5, so its time dependence can be fitted (exponential vs power law).
    print("\nteacher-forced hair amplitude SNR (sign-flip null 95th pct in the 'null' rows; '*' = above it)\n")
    hdr = f"{'axis':8s}{'layer':8s}" + "".join(f"t={t:<11d}" for t in W)
    print(hdr); print("-" * len(hdr))
    def snr(D, sd):
        return float(np.mean((D.mean(0) / sd) ** 2))
    amp = {}
    for ax in meta["axes"]:
        for l in layers:
            li = layers.index(l); cells = []
            for j, t in enumerate(W):
                vals, nulls = [], []
                for si in range(S):
                    A = Z[f"tfA|{si}|{ax}"][:, j, li].astype(np.float64)
                    B = Z[f"tfB|{si}|{ax}"][:, j, li].astype(np.float64)
                    D = B - A
                    sd = np.sqrt(0.5 * (A.var(0) + B.var(0))) + 1e-8
                    vals.append(snr(D, sd))
                    for r in range(a.n_null):
                        nulls.append(snr(D * rng.choice([-1.0, 1.0], size=(len(D), 1)), sd))
                v, p95 = float(np.mean(vals)), float(np.percentile(nulls, 95))
                amp[(ax, l, t)] = (v, p95)
                cells.append(f"{v:.2e}{'*' if v > p95 else ' '}   ")
            print(f"{ax:8s}{'L'+str(l):8s}" + "".join(cells))
        print()
    print("null 95th pct (avg over axes/layers):",
          "  ".join(f"t={t}: {np.mean([amp[(ax, l, t)][1] for ax in meta['axes'] for l in layers]):.1e}" for t in W))
    for k, v in amp.items():
        res[("amp",) + k] = v

    json.dump({"|".join(map(str, k)): v for k, v in res.items()},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)


# ================================================================ cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["gen", "analyze"])
    p.add_argument("--out", default="nohair_run")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--smoke", action="store_true", help="tiny random model, no download")
    p.add_argument("--random_init", action="store_true", help="real tokenizer and architecture, untrained weights")
    p.add_argument("--encoder", default="", help="optional sentence encoder for the exterior view, "
                   "e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--n_scenes", type=int, default=3)
    p.add_argument("--n", type=int, default=32, help="trajectories per prompt (base gets 2n)")
    p.add_argument("--T", type=int, default=512, help="generation length")
    p.add_argument("--win", type=int, default=32)
    p.add_argument("--windows", default="64,128,256,512", help="window end times t")
    p.add_argument("--layers", default="", help="comma list; default ~L/4, L/2, 3L/4, L-1")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16, help="trajectories generated in parallel")
    p.add_argument("--hbatch", type=int, default=4, help="sequences per forward pass when reading hidden states")
    p.add_argument("--n_null", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    stage_gen(a) if a.stage == "gen" else stage_analyze(a)