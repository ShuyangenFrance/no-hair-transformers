#!/usr/bin/env python3
"""
nohair_gen_v2.py — Transformer "no-hair" experiment.

Time      = generation step t (free autoregressive evolution, no external input after the prompt).
Exterior  = statistics of the generated text in a window ending at t (definition 1).
Interior  = hidden states averaged over the same window (definition 2).
Metric    = decodability: can a linear probe tell which side of a prompt pair a trajectory came from?

Stages (the later ones only read what gen saved; old run directories from v1 work too):
  python nohair_gen_v2.py gen        --out run1   # generate trajectories, store hidden states
  python nohair_gen_v2.py analyze    --out run1   # probes, hair-amplitude SNR, decay slopes
  python nohair_gen_v2.py attn       --out run1   # where late tokens attend: sink / rest of prompt / local
  python nohair_gen_v2.py langswitch --out run1   # does a trajectory stay in its language?

attn and langswitch reload the same model as the run (name, revision, random init, seed are read
from meta.json), so no model flags are needed for them.

Smoke test with a tiny random Qwen2 (no download):
  python nohair_gen_v2.py gen --smoke --out smoke && python nohair_gen_v2.py analyze --out smoke \
      && python nohair_gen_v2.py attn --out smoke
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
SCENES_EN = [  # for English-centric models (Pythia); the language axis is French
    dict(base="It was raining in Paris today, and I stood at the metro entrance,",
         detail="It was snowing in Paris today, and I stood at the metro entrance,",
         para="Rain was falling over Paris today; I stopped at the door of the metro station,",
         lang="Il pleuvait à Paris aujourd'hui, et je me tenais à l'entrée du métro,",
         genre="PARIS (Reuters) - Heavy rain hit the French capital on Tuesday, and at one metro entrance"),
    dict(base="In my grandmother's kitchen, rain ran down the window as she",
         detail="In my grandmother's kitchen, snow piled up on the windowsill as she",
         para="Rain streaked the glass of my grandmother's kitchen window while she",
         lang="Dans la cuisine de ma grand-mère, la pluie coulait sur la vitre pendant qu'elle",
         genre="LONDON (AP) - Days of heavy rain left several homes with leaking kitchen windows, and one elderly resident"),
    dict(base="It was raining that day. He waited at the train station for a long time, and finally",
         detail="It was snowing that day. He waited at the train station for a long time, and finally",
         para="The rain never stopped. He spent hours waiting at the station until, at last,",
         lang="Il pleuvait ce jour-là. Il a attendu longtemps à la gare, et finalement",
         genre="BERLIN (Reuters) - Rain delayed dozens of trains on Monday, and one passenger who had waited for hours"),
]
SCENE_SETS = {"zh": SCENES, "en": SCENES_EN}
AXES = ["detail", "para", "lang", "genre"]


# ================================================================ model loading (shared)
def load_model(a, eager=False):
    """Returns torch, model, tok, encode, pad_id, eos_id, dev, empty_cache.
    eager=True is needed to get attention weights. Random init is reproduced exactly from a.seed."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else \
        ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print("device:", dev, flush=True)

    def empty_cache():
        if dev == "cuda": torch.cuda.empty_cache()
        elif dev == "mps": torch.mps.empty_cache()

    kw = {"attn_implementation": "eager"} if eager else {}
    if a.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, **kw)
        torch.manual_seed(0)
        model, tok = Qwen2ForCausalLM(cfg), None
        encode = lambda s: [ord(c) % 512 for c in s]
        pad_id, eos_id = 0, None
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        rev = {"revision": a.revision} if a.revision else {}
        tok = AutoTokenizer.from_pretrained(a.model, **rev)
        if a.random_init:   # same architecture and tokenizer, untrained weights: the "no learning" baseline
            torch.manual_seed(a.seed)
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(a.model, **rev), **kw).to(dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, **rev, **kw)
        encode = lambda s: tok(s, add_special_tokens=False).input_ids
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        eos_id = tok.eos_token_id
    model = model.to(dev).eval()
    return torch, model, tok, encode, pad_id, eos_id, dev, empty_cache


def restore_run_args(a, meta):
    """Make a model-reloading stage use exactly the model of the run in a.out (also for v1 runs)."""
    a.smoke = meta["model"] == "smoke"
    a.model = meta.get("model_name", meta["model"].replace(" [random init]", "").split("@")[0])
    a.random_init = meta.get("random_init", "[random init]" in meta["model"])
    a.revision = meta.get("revision", None)
    a.seed = meta.get("seed", a.seed)
    return SCENE_SETS[meta.get("scenes", "zh")][: meta["n_scenes"]]


# ================================================================ stage 1: generation
def stage_gen(a):
    torch, model, tok, encode, pad_id, eos_id, dev, empty_cache = load_model(a)
    scene_set = SCENE_SETS[a.scenes]

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
    for si, sc in enumerate(scene_set[:a.n_scenes]):
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
    meta = dict(model="smoke" if a.smoke else a.model + (f"@{a.revision}" if a.revision else "")
                + (" [random init]" if a.random_init else ""), temp=a.temp, T=a.T, n=a.n, win=a.win,
                windows=windows, layers=layers, n_scenes=min(a.n_scenes, len(scene_set)), axes=AXES,
                scenes=a.scenes, model_name=a.model, random_init=bool(a.random_init), revision=a.revision,
                seed=a.seed,
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

    # ---- decay law: slope of log10 SNR against log10 t, with a bootstrap over continuations
    # (rows of tfA/tfB are resampled in pairs, so each resample keeps the shared-continuation design)
    lt = np.log10(np.array(W, float))
    def curve(A, B):                       # A, B: (m, n_win, d) -> SNR per window
        D = B - A
        sd = np.sqrt(0.5 * (A.var(0) + B.var(0))) + 1e-8
        return np.mean((D.mean(0) / sd) ** 2, axis=-1)
    def slope(y):
        return float(np.polyfit(lt, np.log10(np.maximum(y, 1e-30)), 1)[0])
    print(f"\ndecay slope d log SNR / d log t  (point estimate [95% bootstrap CI], {a.boot} resamples)")
    print("SNR ~ t^slope; the amplitude of the prompt's offset decays as t^(slope/2)\n")
    print(f"{'axis':8s}" + "".join(f"{'L'+str(l):26s}" for l in layers))
    for ax in meta["axes"]:
        cells = []
        for l in layers:
            li = layers.index(l)
            AB = [(Z[f"tfA|{si}|{ax}"][:, :, li].astype(np.float64), Z[f"tfB|{si}|{ax}"][:, :, li].astype(np.float64))
                  for si in range(S)]
            s0 = slope(np.mean([curve(A, B) for A, B in AB], 0))
            bs = []
            for b in range(a.boot):
                cs = []
                for A, B in AB:
                    idx = rng.integers(0, len(A), len(A))
                    cs.append(curve(A[idx], B[idx]))
                bs.append(slope(np.mean(cs, 0)))
            lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
            res[("slope", ax, l)] = (s0, float(lo), float(hi))
            cells.append(f"{s0:+.2f} [{lo:+.2f},{hi:+.2f}]".ljust(26))
        print(f"{ax:8s}" + "".join(cells))

    json.dump({"|".join(map(str, k)): v for k, v in res.items()},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)


# ================================================================ stage 3: attention
def attach_value_norms(model, torch, full=False):
    """Forward hooks that record, per layer, n[h, k] = || W_O^h v_k^h ||: the size of what key k would
    write into the residual stream through head h if it got all the attention. alpha_qk * n[h, k] is then
    the size of key k's actual contribution (value-weighted attention). Attention sinks get large alpha
    but tiny n, so this measure discounts them. Supports Llama/Qwen-style and GPT-NeoX (Pythia) blocks."""
    cfg = model.config
    H = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    store, handles = {}, []
    for l, layer in enumerate(model.base_model.layers):
        if hasattr(layer, "self_attn"):                       # Qwen2 / Llama: separate v_proj, GQA
            at = layer.self_attn
            kvh = getattr(cfg, "num_key_value_heads", H)
            def hook(mod, inp, out, l=l, Wo=at.o_proj.weight, kvh=kvh):
                v = out[0].float().view(out.shape[1], kvh, hd).repeat_interleave(H // kvh, dim=1)
                Woh = Wo.float().view(Wo.shape[0], H, hd)
                fv = torch.einsum("dhe,nhe->hnd", Woh, v)                    # H x N x d
                store[l] = fv if full else fv.norm(dim=-1)                     # H x N
            handles.append(at.v_proj.register_forward_hook(hook))
        elif hasattr(layer, "attention"):                     # GPT-NeoX: fused qkv laid out per head
            at = layer.attention
            def hook(mod, inp, out, l=l, Wo=at.dense.weight):
                v = out[0].float().view(out.shape[1], H, 3 * hd)[..., 2 * hd:]
                Woh = Wo.float().view(Wo.shape[0], H, hd)
                fv = torch.einsum("dhe,nhe->hnd", Woh, v)
                store[l] = fv if full else fv.norm(dim=-1)
            handles.append(at.query_key_value.register_forward_hook(hook))
        else:
            raise NotImplementedError("value norms: unknown attention block in layer %d" % l)
    return store, handles


def stage_attn(a):
    """Where do late tokens look? For every query position in each window, per layer and head:
      sink   = weight on prompt tokens 0..sink_k-1 (with no BOS, token 0 usually acts as attention sink)
      rest   = weight on prompt tokens sink_k..P-1  (the prompt proper)
      local  = weight on the a.local most recent tokens (including the query itself)
    Each is reported twice: raw attention weight, and value-weighted share
    (alpha_qk * ||W_O v_k||, normalised over keys), which is closer to actual information flow.
    Also prints the per-token profile of the prompt at the last window, to show where sinks sit.
    Uses the base prompt's free-generation continuations already stored by gen."""
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    scenes = restore_run_args(a, meta)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(a, eager=True)
    W, win, S = meta["windows"], meta["win"], meta["n_scenes"]
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    nA = min(a.n_attn, len(Z["conts|0|base"]))
    K = a.sink_k
    sink, rest, local, vsink, vrest, vlocal = (np.zeros((S, nA, len(W), L, H), np.float32) for _ in range(6))
    unif = np.zeros((S, len(W)))
    vstore, handles = attach_value_norms(model, torch)
    prof = []          # per scene: (2, P) mean share of each prompt token at the last window, raw / weighted
    for si, sc in enumerate(scenes):
        ids = encode(sc["base"]); P = len(ids)
        for j, t in enumerate(W):   # uniform attention over q+1 keys puts (P-K)/(q+1) on the prompt proper
            unif[si, j] = np.mean([(P - K) / (q + 1) for q in range(P + t - win, P + t)])
        pr = np.zeros((2, P))
        for i, c in enumerate(Z[f"conts|{si}|base"][:nA]):
            x = torch.tensor([ids + c.tolist()], device=dev)
            with torch.no_grad():
                att = model.base_model(x, output_attentions=True).attentions
            for l, A in enumerate(att):
                A = A[0].float()                                    # H x N x N
                vn = vstore[l].to(A.device)                         # H x N
                ar = torch.arange(win, device=A.device)
                for j, t in enumerate(W):
                    pos = torch.arange(P + t - win, P + t, device=A.device)
                    rows_raw = A[:, pos, :]                         # H x win x N
                    rows_w = rows_raw * vn[:, None, :]
                    rows_w = rows_w / rows_w.sum(-1, keepdim=True).clamp(min=1e-12)
                    for rows, (o_s, o_r, o_l) in ((rows_raw, (sink, rest, local)), (rows_w, (vsink, vrest, vlocal))):
                        o_s[si, i, j, l] = rows[:, :, :K].sum(-1).mean(1).cpu().numpy()
                        o_r[si, i, j, l] = rows[:, :, K:P].sum(-1).mean(1).cpu().numpy()
                        cs = rows.cumsum(-1)
                        hi = cs[:, ar, pos]
                        lo_idx = pos - a.local
                        lo = torch.where(lo_idx >= 0, cs[:, ar, lo_idx.clamp(min=0)], torch.zeros_like(hi))
                        o_l[si, i, j, l] = (hi - lo).mean(1).cpu().numpy()
                    if j == len(W) - 1:
                        pr[0] += rows_raw[:, :, :P].mean((0, 1)).cpu().numpy() / (L * nA)
                        pr[1] += rows_w[:, :, :P].mean((0, 1)).cpu().numpy() / (L * nA)
            del att
            empty_cache()
        prof.append(pr)
        print(f"scene {si} done ({nA} trajectories)", flush=True)
    for h in handles:
        h.remove()

    np.savez_compressed(os.path.join(a.out, "attn.npz"), sink=sink, rest=rest, local=local,
                        vsink=vsink, vrest=vrest, vlocal=vlocal, uniform=unif, windows=np.array(W),
                        local_span=a.local, sink_k=K,
                        **{f"profile{si}": pr for si, pr in enumerate(prof)})
    lt = np.log10(np.array(W, float))
    def slope(y):
        y = np.asarray(y, float)
        return float(np.polyfit(lt, np.log10(y), 1)[0]) if np.all(y > 0) else float("nan")

    print(f"\nmodel={meta['model']}  scenes={S}  trajectories/scene={nA}  win={win}  sink_k={K}  (mean over heads)")
    tables = [(f"prompt_rest  (prompt tokens {K}..P-1)", rest), (f"sink  (prompt tokens 0..{K-1})", sink),
              (f"local  (last {a.local} tokens)", local),
              ("VALUE-WEIGHTED prompt_rest", vrest), ("VALUE-WEIGHTED sink", vsink),
              (f"VALUE-WEIGHTED local", vlocal)]
    for name, arr in tables:
        m = arr.mean(axis=(0, 1, 4))                                # n_win x L
        is_rest = arr is rest or arr is vrest
        print(f"\n{name}")
        hdr = f"{'layer':8s}" + "".join(f"t={t:<9d}" for t in W) + ("slope" if is_rest else "")
        print(hdr); print("-" * len(hdr))
        if is_rest:
            print(f"{'uniform':8s}" + "".join(f"{v:<11.2e}" for v in unif.mean(0)) + f"{slope(unif.mean(0)):+.2f}")
        for l in range(L):
            print(f"{l:<8d}" + "".join(f"{v:<11.2e}" for v in m[:, l]) + (f"{slope(m[:, l]):+.2f}" if is_rest else ""))
        mm = m.mean(1)
        print(f"{'mean':8s}" + "".join(f"{v:<11.2e}" for v in mm) + (f"{slope(mm):+.2f}" if is_rest else ""))
    print(f"\nprompt profile at t={W[-1]}: share of attention on each prompt token, x1000, "
          "mean over layers and heads (raw / value-weighted)")
    for si, pr in enumerate(prof):
        print(f"scene {si} raw: " + " ".join(f"{v*1000:5.1f}" for v in pr[0]))
        print(f"scene {si} vw : " + " ".join(f"{v*1000:5.1f}" for v in pr[1]))
    print("\nsaved", os.path.join(a.out, "attn.npz"), "(per scene, trajectory, window, layer, head)")


# ================================================================ stage 3b: direct vs relay paths
def stage_paths(a):
    """Exact split of each attention block's contribution to the prompt-induced offset.
    For a query q in a late window, block k writes  sum_h sum_key alpha_qkey W_O^h v_key  into the
    residual stream. Split the keys into
        direct = prompt tokens            (the query reads the prompt itself)
        relay  = continuation tokens      (the query reads tokens that earlier layers already
                                           filled with prompt information)
    and take the difference between the two prompts of a pair under the SAME continuation.
    Sinks drop out automatically when they are the same token in both prompts.
    The amplitude of each path is tracked against t exactly like alpha_beta.py (fixed per-dim scale
    from the first window), so its decay exponent can be compared with the measured alpha."""
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    scenes = restore_run_args(a, meta)
    torch, model, tok, encode, _, _, dev, empty_cache = load_model(a, eager=True)
    W, win, S, axes = meta["windows"], meta["win"], meta["n_scenes"], meta["axes"]
    L, d = model.config.num_hidden_layers, model.config.hidden_size
    store, handles = attach_value_norms(model, torch, full=True)
    m = a.n_paths // 2

    def paths(ids, c):
        """(L, n_win, 2, d): window-mean of direct / relay writes of every block."""
        P = len(ids)
        x = torch.tensor([ids + c.tolist()], device=dev)
        with torch.no_grad():
            att = model.base_model(x, output_attentions=True).attentions
        out = np.zeros((L, len(W), 2, d), np.float32)
        for k, A in enumerate(att):
            A = A[0].float(); fv = store[k].to(A.device)            # H x N x N ; H x N x d
            for j, t in enumerate(W):
                rows = A[:, P + t - win:P + t, :]                   # H x win x N
                out[k, j, 0] = torch.einsum("hqn,hnd->d", rows[:, :, :P], fv[:, :P]).cpu().numpy() / win
                out[k, j, 1] = torch.einsum("hqn,hnd->d", rows[:, :, P:], fv[:, P:]).cpu().numpy() / win
        del att
        empty_cache()
        return out

    D, T = [], []          # per (scene, axis): differences (m2, L, nW, 2, d) and base writes for the scale
    for si, sc in enumerate(scenes):
        ib = encode(sc["base"])
        for ax in axes:
            C = np.concatenate([Z[f"conts|{si}|base"][:m], Z[f"conts|{si}|{ax}"][:m]])
            ia = encode(sc[ax])
            pb = np.stack([paths(ib, c) for c in C]); pa = np.stack([paths(ia, c) for c in C])
            D.append(pa - pb); T.append(pb.sum(3))                  # total attention write, base prompt
            print(f"scene {si} {ax} done", flush=True)
    for h in handles:
        h.remove()

    lt = np.log10(np.array(W, float))
    def amp(Dg, Tt):
        """squared consistent offset per window, per-dim scale fixed at the first window"""
        s0 = Tt[:, 0].std(0) + 1e-8                                 # (d,)
        return ((Dg.mean(0) / s0) ** 2).mean(-1)                    # (n_win,)
    def slope(y):
        return float(np.polyfit(lt, np.log10(np.maximum(y, 1e-30)), 1)[0])

    # per block
    blk = np.zeros((L, 2, len(W)))
    for k in range(L):
        for g in range(2):
            blk[k, g] = np.mean([amp(Dx[:, k, :, g], Tx[:, k]) for Dx, Tx in zip(D, T)], 0)
    # cumulative over blocks 0..l-1  (what the residual stream at hidden_states[l] has received)
    lays = sorted(set(meta["layers"] + [L]))
    cum = {}
    for l in lays:
        for g, name in ((0, "direct"), (1, "relay"), (None, "both")):
            vals = []
            for Dx, Tx in zip(D, T):
                Dg = Dx[:, :l].sum(1)                                # (m2, nW, 2, d)
                Dg = Dg.sum(2) if g is None else Dg[:, :, g]
                vals.append(amp(Dg, Tx[:, :l].sum(1)))
            cum[(l, name)] = np.mean(vals, 0)

    np.savez_compressed(os.path.join(a.out, "paths.npz"), blk=blk, windows=np.array(W),
                        **{f"cum_{l}_{n}": v for (l, n), v in cum.items()})
    print(f"\nmodel={meta['model']}  continuations per pair={2*m}  (alpha = -slope/2 of the squared offset)")
    print("\nper block: decay exponent alpha of what the block writes, and relay share at the last window")
    print(f"{'block':7s}{'alpha_direct':>14s}{'alpha_relay':>13s}{'relay/(dir+rel) @t=' + str(W[-1]):>26s}")
    for k in range(L):
        ad, ar_ = -slope(blk[k, 0]) / 2, -slope(blk[k, 1]) / 2
        share = blk[k, 1, -1] / (blk[k, 0, -1] + blk[k, 1, -1])
        print(f"{k:<7d}{ad:>14.2f}{ar_:>13.2f}{share:>26.2f}")
    print("\ncumulative over blocks 0..l-1 (compare with alpha from alpha_beta.py at the same layer)")
    print(f"{'layer':7s}{'alpha_direct':>14s}{'alpha_relay':>13s}{'alpha_both':>12s}"
          f"{'relay share @t=' + str(W[-1]):>22s}")
    for l in lays:
        dd, rr, bb = cum[(l, "direct")], cum[(l, "relay")], cum[(l, "both")]
        print(f"{l:<7d}{-slope(dd)/2:>14.2f}{-slope(rr)/2:>13.2f}{-slope(bb)/2:>12.2f}"
              f"{rr[-1] / (dd[-1] + rr[-1]):>22.2f}")
    print("\nnote: cumulative sums cover attention writes only; MLPs act position-wise on what attention brought in.")


# ================================================================ stage 4: language switching
FR_WORDS = set("le la les de des du et est que qui une un dans pour pas il elle ce cette sur au aux "
               "je nous vous ils elles mais avec son sa ses était été plus très".split())
EN_WORDS = set("the and of to a in is that it was he she for on with as his her at but not they "
               "had have this from be were been very".split())

def stage_langswitch(a):
    """Per trajectory and window: share of the 'lang' language (CJK characters for the zh set,
    French function words for the en set). Tells whether the language axis is held by every
    trajectory or eroded by some trajectories switching."""
    import re
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    Z = np.load(os.path.join(a.out, "data.npz"))
    restore_run_args(a, meta)
    if a.smoke:
        print("langswitch needs a real tokenizer; skipped for smoke runs"); return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, **({"revision": a.revision} if a.revision else {}))
    W, win, S = meta["windows"], meta["win"], meta["n_scenes"]
    zh = meta.get("scenes", "zh") == "zh"

    def score(s):
        if zh:
            chars = [ch for ch in s if not ch.isspace()]
            return sum("\u4e00" <= ch <= "\u9fff" for ch in chars) / len(chars) if chars else np.nan
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœæ]+", s.lower())
        fr = sum(w in FR_WORDS for w in words); en = sum(w in EN_WORDS for w in words)
        return fr / (fr + en) if fr + en else np.nan

    what = "CJK character fraction" if zh else "French share of function words"
    print(f"\nmodel={meta['model']}   score = {what}   (base prompt vs lang prompt)\n")
    print(f"{'scene':6s}{'prompt':8s}" + "".join(f"t={t:<8d}" for t in W) + "switched  undetermined")
    for si in range(S):
        for k in ["base", "lang"]:
            C = Z[f"conts|{si}|{k}"][: meta["n"]]
            texts = [[tok.decode(c[t - win:t].tolist()) for t in W] for c in C]
            sc = np.array([[score(x) for x in row] for row in texts])
            first, last = sc[:, 0], sc[:, -1]
            ok = ~np.isnan(first) & ~np.isnan(last)
            sw = ok & ((first > 0.5) != (last > 0.5))
            print(f"{si:<6d}{k:8s}" + "".join(f"{v:<10.2f}" for v in np.nanmean(sc, 0))
                  + f"{sw.sum()}/{ok.sum():<7d} {(~ok).sum()}")
            for i in np.where(sw)[0][:2]:
                print(f"      switched trajectory {i}, last window: {texts[i][-1]!r}")
    print("\nswitched = first and last window fall on different sides of 0.5; "
          "undetermined = a window with no scorable text")


# ================================================================ cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["gen", "analyze", "attn", "paths", "langswitch"])
    p.add_argument("--out", default="nohair_run")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--smoke", action="store_true", help="tiny random model, no download")
    p.add_argument("--random_init", action="store_true", help="real tokenizer and architecture, untrained weights")
    p.add_argument("--revision", default=None, help="model revision, e.g. step1000 for Pythia checkpoints")
    p.add_argument("--scenes", choices=["zh", "en"], default="zh", help="prompt set (en: language axis is French)")
    p.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the decay slope")
    p.add_argument("--n_attn", type=int, default=8, help="attn: trajectories per scene")
    p.add_argument("--local", type=int, default=32, help="attn: span of the 'local' attention measure")
    p.add_argument("--sink_k", type=int, default=1, help="attn: number of leading prompt tokens counted as sink")
    p.add_argument("--n_paths", type=int, default=8, help="paths: shared continuations per prompt pair")
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
    dict(gen=stage_gen, analyze=stage_analyze, attn=stage_attn, paths=stage_paths,
         langswitch=stage_langswitch)[a.stage](a)
