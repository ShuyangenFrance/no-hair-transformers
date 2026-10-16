#!/usr/bin/env python3
"""
make_figures.py — all figures of the paper from the saved results (no model is loaded).

Run from the repo root:   python src/make_figures.py
Writes figures/fig*.pdf (and .png previews). Missing inputs are reported and their panels skipped.

  fig1_decay        prompt offset S(t), trained vs untrained, Pythia-160m, real-corpus pairs
  fig2_local        local exponent vs t, every layer of untrained Pythia-1b with the small-gain prediction,
                    and the trained model for comparison
  fig3_depth        alpha vs layer index: untrained Pythia-160m / 1b / Qwen-0.5B, trained models
  fig4_training     alpha across training per layer, attention entropy and sink mass
  fig5_paths        relay share and path exponents vs depth at steps 0, 512, 143000
  fig6_causal       alpha under compensation; retention vs perplexity, prompt vs placebo
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN = 16
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C_INIT, C_TRAIN = "#d95f02", "#1b9e77"
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def log_ticks(ax, ticks, axis="x"):
    """explicit, readable ticks on a log axis (no minor labels)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    a.set_minor_locator(NullLocator())
missing = []


def have(path):
    if os.path.exists(path):
        return True
    missing.append(path)
    return False


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.pdf")


def load_real(d):
    """unbiased signal curves (axes, pairs, layers, windows), centre-labelled times, layers."""
    Z = np.load(f"{d}/real.npz")
    m = int(Z["m"])
    sig = Z["sig"] - Z["flo"] * m / (m - 1)
    t = np.array(Z["windows"], float) - (WIN - 1) / 2
    return sig, t, [int(x) for x in Z["layers"]], Z


def pooled_alpha(d):
    """layer -> mean alpha over the four axes (from real.json, written by the latest analysis)."""
    r = json.load(open(f"{d}/real.json"))
    out = {}
    for l in r["layers"]:
        v = [r["alpha"][f"{ax}|{l}"][0] for ax in ("detail", "para", "lang", "unrelated")]
        out[l] = float(np.mean(v))
    return out


def local_from_curve(c, t):
    lc, lt = np.log(c), np.log(t)
    return -(lc[1:] - lc[:-1]) / (lt[1:] - lt[:-1]) / 2, np.sqrt(t[1:] * t[:-1])


# ---------------------------------------------------------------- fig 1
def fig1():
    a, b = "real/p160_step0", "real/p160_final"
    if not (have(a) and have(b)):
        return
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.7))
    rng = np.random.default_rng(0)
    for d, col, lab in ((a, C_INIT, "untrained"), (b, C_TRAIN, "trained")):
        sig, t, layers, _ = load_real(d)
        for l, ls in ((1, ":"), (layers[-1], "-")):
            li = layers.index(l)
            cur = sig[:, :, li].mean((0, 1))
            n = sig.shape[1]
            bs = np.array([sig[:, rng.integers(0, n, n), li].mean((0, 1)) for _ in range(300)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            ax.fill_between(t, lo / cur[0], hi / cur[0], color=col, alpha=0.2, lw=0)
            ax.plot(t, cur / cur[0], ls, color=col, label=f"{lab}, layer {l}")
    ref = t ** -2.0 * t[0] ** 2
    ax.plot(t, ref, color="grey", lw=0.8, ls="--", label=r"$t^{-2}$ ($\alpha=1$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("continuation token $t$"); ax.set_ylabel(r"$\hat{S}(t)\,/\,\hat{S}(t_1)$")
    ax.set_title("Pythia-160m, real-corpus pairs")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig1_decay")


# ---------------------------------------------------------------- fig 2
def fig2():
    a, b = "real/p1b_step0_all", "real/p1b_final"
    if not have(a):
        return
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    sig, t, layers, Z = load_real(a)
    P = float(Z["plen"][:3].mean())
    cmap = plt.get_cmap("viridis")
    for li, l in enumerate(layers):
        loc, tm = local_from_curve(sig[:3, :, li].mean((0, 1)), t)
        col = cmap(li / max(1, len(layers) - 1))
        axs[0].plot(tm, loc, "o-", ms=2.5, color=col, lw=1, label=f"L{l}" if l in (1, 2, 4, 8, 15) else None)
        if l >= 3 and np.all(loc < 1):                     # small-gain prediction with the fitted G_l
            u = np.log((P + tm) / P)
            slope, icpt = np.polyfit(u, 1 / (1 - loc) ** 2, 1)      # (1-a)^-2 = u/G + c
            if slope <= 0:
                continue
            uu = np.linspace(u.min(), u.max(), 50)
            axs[0].plot(P * (np.exp(uu) - 1), 1 - 1 / np.sqrt(slope * uu + icpt), color=col, lw=0.7, ls="--")
    axs[0].set_xscale("log"); axs[0].set_xlabel("$t$"); axs[0].set_ylabel(r"local exponent $\alpha_{\rm loc}$")
    axs[0].set_title("untrained Pythia-1b (dashed: small-gain fit)")
    axs[0].axhline(1, color="grey", lw=0.6); axs[0].legend(frameon=False, ncol=2)
    if have(b):
        sig, t, layers, _ = load_real(b)
        for li, l in enumerate(layers):
            loc, tm = local_from_curve(sig[:3, :, li].mean((0, 1)), t)
            axs[1].plot(tm, loc, "o-", ms=2.5, lw=1, color=cmap(li / max(1, len(layers) - 1)), label=f"L{l}")
        axs[1].set_xscale("log"); axs[1].set_xlabel("$t$"); axs[1].axhline(1, color="grey", lw=0.6)
        axs[1].set_title("trained Pythia-1b"); axs[1].legend(frameon=False, ncol=2)
    axs[0].set_ylim(0, 1.6)
    for a_ in axs:
        log_ticks(a_, [16, 64, 256, 1024])
    save(fig, "fig2_local")


# ---------------------------------------------------------------- fig 3
def fig3():
    runs = [("real/p160_step0", "Pythia-160m", C_INIT, "o-"), ("real/p1b_step0_all", "Pythia-1b", C_INIT, "s--"),
            ("real/qwen05_random", "Qwen2.5-0.5B", C_INIT, "^:"),
            ("real/p160_final", "Pythia-160m", C_TRAIN, "o-"), ("real/p1b_final", "Pythia-1b", C_TRAIN, "s--"),
            ("real/qwen05_final", "Qwen2.5-0.5B", C_TRAIN, "^:"), ("real/qwen15_final", "Qwen2.5-1.5B", C_TRAIN, "v-.")]
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))
    for d, lab, col, sty in runs:
        if not have(d):
            continue
        al = pooled_alpha(d)
        ls = sorted(al)
        state = "untrained" if col == C_INIT else "trained"
        ax.plot(ls, [al[l] for l in ls], sty, color=col, ms=3, lw=1, label=f"{lab}, {state}")
    ax.set_xscale("log"); ax.set_xlabel("layer $l$"); ax.set_ylabel(r"$\alpha$")
    log_ticks(ax, [1, 2, 3, 5, 10, 20])
    ax.set_title("decay exponent vs depth (real-corpus pairs)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.set_size_inches(5.0, 2.8)
    save(fig, "fig3_depth")


# ---------------------------------------------------------------- fig 4
def fig4():
    steps = [0, 128, 256, 512, 1000, 4000, 16000, 143000]
    dirs = {s: ("real/p160_final" if s == 143000 else f"real/p160_step{s}") for s in steps}
    ok = [s for s in steps if have(dirs[s])]
    if not ok:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    x = lambda s: max(s, 64)                                # step 0 drawn at 64
    al = {s: pooled_alpha(dirs[s]) for s in ok}
    layers = sorted(al[ok[0]])
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        axs[0].plot([x(s) for s in ok], [al[s][l] for s in ok], "o-", ms=2.5, lw=1,
                    color=cmap(i / max(1, len(layers) - 1)), label=f"L{l}")
    for k, ttl, ylab in ((1, "attention entropy", "normalised entropy"), (2, "attention on token 0", "sink mass")):
        key = "entropy" if k == 1 else "sink"
        for i, l in enumerate((1, 3, 7, 12)):
            vals = []
            for s in ok:
                f = f"attn_stats/pythia-160m_step{s}.json"
                vals.append(json.load(open(f))[str(l - 1)][key] if os.path.exists(f) else np.nan)
            axs[k].plot([x(s) for s in ok], vals, "o-", ms=2.5, lw=1, color=cmap(i / 3), label=f"L{l}")
        axs[k].set_title(ttl); axs[k].set_ylabel(ylab)
    for a_ in axs:
        a_.set_xscale("log"); a_.set_xlabel("training step")
        a_.axvspan(128, 512, color="grey", alpha=0.12, lw=0)
        a_.set_xticks([64, 512, 4000, 143000]); a_.set_xticklabels(["0", "512", "4k", "143k"])
    axs[0].set_ylabel(r"$\alpha$"); axs[0].set_title("decay exponent")
    axs[0].legend(frameon=False, ncol=2, fontsize=6.5); axs[1].legend(frameon=False, fontsize=6.5)
    axs[2].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "fig4_training")


# ---------------------------------------------------------------- fig 5
def fig5():
    ck = [("mech2/p160_step0", "step 0", C_INIT), ("mech2/p160_step512", "step 512", "#7570b3"),
          ("mech2/p160_final", "step 143k", C_TRAIN)]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for d, lab, col in ck:
        if not have(f"{d}/mech.json"):
            continue
        r = json.load(open(f"{d}/mech.json"))["result"]
        ls = sorted(int(k) for k in r)
        axs[0].plot(ls, [r[str(l)]["relay_share"] for l in ls], "o-", ms=3, color=col, label=lab)
        axs[1].plot(ls, [r[str(l)]["direct"] for l in ls], "o-", ms=3, color=col, label=f"direct, {lab}")
        axs[1].plot(ls, [r[str(l)]["relay"] for l in ls], "s--", ms=3, color=col, label=f"relay, {lab}")
    axs[0].set_xlabel("layer $l$"); axs[0].set_ylabel("relay share (last window)"); axs[0].set_ylim(0, 1)
    axs[0].legend(frameon=False); axs[0].set_title("where the late offset comes from")
    axs[1].set_xlabel("layer $l$"); axs[1].set_ylabel(r"$\alpha$ of the path"); axs[1].set_title("decay of each path")
    axs[1].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.set_size_inches(6.8, 3.0)
    save(fig, "fig5_paths")


# ---------------------------------------------------------------- fig 6
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    ck = [("mech2/p160_step0", "step 0", C_INIT), ("mech2/p160_step512", "step 512", "#7570b3"),
          ("mech2/p160_final", "step 143k", C_TRAIN)]
    for d, lab, col in ck:
        if not have(f"{d}/mech.json"):
            continue
        r = json.load(open(f"{d}/mech.json"))["result"]
        ls = sorted(int(k) for k in r)
        axs[0].plot(ls, [r[str(l)]["measured"][0] for l in ls], "o-", ms=3, color=col, label=f"{lab}")
        axs[0].plot(ls, [r[str(l)]["perhead"][0] for l in ls], "o:", ms=3, color=col)
    axs[0].axhline(0, color="grey", lw=0.6)
    axs[0].set_xlabel("layer $l$"); axs[0].set_ylabel(r"$\alpha$")
    axs[0].set_title("original (solid) vs per-head compensation (dotted)")
    axs[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    f = "tradeoff2/p160_final/tradeoff.json"
    if have(f):
        T = json.load(open(f))
        base = np.array(T["none|0.0"]["nll"])
        for tg, col, mk in (("prompt", C_TRAIN, "o-"), ("placebo", "#e7298a", "s--")):
            keys = sorted([k for k in T if k.startswith(tg + "|")], key=lambda k: float(k.split("|")[1]))
            keys = ["none|0.0"] + keys
            a11 = [T[k]["alpha"][-1][0] for k in keys]
            dppl = [(np.exp(np.mean(T[k]["nll"]) - base.mean()) - 1) * 100 for k in keys]
            axs[1].plot(dppl, a11, mk, ms=3.5, color=col, label=tg)
            for k, xv, yv in zip(keys, dppl, a11):
                if tg == "prompt" and k != "none|0.0":
                    axs[1].annotate(f"s={float(k.split('|')[1]):g}", (xv, yv), fontsize=6, xytext=(3, 2),
                                    textcoords="offset points")
        axs[1].set_xlabel(r"perplexity change (%)"); axs[1].set_ylabel(r"$\alpha$, last layer")
        axs[1].set_title("retention vs cost (step 143k)"); axs[1].legend(frameon=False)
    fig.set_size_inches(6.8, 3.0)
    save(fig, "fig6_causal")


for f in (fig1, fig2, fig3, fig4, fig5, fig6):
    try:
        f()
    except Exception as e:                                   # keep going; report at the end
        print(f"{f.__name__} failed: {e!r}")
if missing:
    print("\nmissing inputs (their panels were skipped):")
    for m in sorted(set(missing)):
        print("  ", m)
