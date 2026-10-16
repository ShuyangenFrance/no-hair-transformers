#!/usr/bin/env python3
"""
pythia_sweep.py — run the no-hair experiment across Pythia training checkpoints, then summarise.

  python pythia_sweep.py run                 # gen + analyze + attn for every checkpoint (resumable)
  python pythia_sweep.py summarize           # csv + two plots

Checkpoints already finished are skipped, so an interrupted sweep can simply be restarted.
step0 is Pythia's own initialisation, i.e. the natural "untrained" baseline.
Test the plumbing first:  python pythia_sweep.py run --smoke --steps 0 1000 --root pythia_smoke
"""
import argparse, csv, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "nohair_gen_v2.py")
STEPS = [0, 128, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 143000]


def call(args, log=None):
    print(">>", " ".join(args[1:]), flush=True)
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout[-3000:], flush=True)
    if log:
        open(log, "w").write(r.stdout)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(args)}")


def run(a):
    for s in a.steps:
        out = os.path.join(a.root, f"step{s}")
        gen = [sys.executable, MAIN, "gen", "--out", out, "--model", a.model, "--revision", f"step{s}",
               "--scenes", "en", "--win", str(a.win), "--windows", a.windows, "--n", str(a.n), "--T", str(a.T),
               "--batch", str(a.batch), "--hbatch", str(a.hbatch), "--temp", str(a.temp)]
        if a.smoke:
            gen.append("--smoke")
        if not os.path.exists(os.path.join(out, "data.npz")):
            call(gen)
        if not os.path.exists(os.path.join(out, "results.json")):
            call([sys.executable, MAIN, "analyze", "--out", out], log=os.path.join(out, "analyze.txt"))
        if a.redo_attn or not os.path.exists(os.path.join(out, "attn.npz")):
            call([sys.executable, MAIN, "attn", "--out", out, "--n_attn", str(a.n_attn)],
                 log=os.path.join(out, "attn.txt"))
        if a.paths and (a.redo_paths or not os.path.exists(os.path.join(out, "paths.npz"))):
            call([sys.executable, MAIN, "paths", "--out", out, "--n_paths", str(a.n_paths)],
                 log=os.path.join(out, "paths.txt"))
        print(f"=== step{s} done ===", flush=True)


def summarize(a):
    rows = []
    for s in a.steps:
        out = os.path.join(a.root, f"step{s}")
        if not os.path.exists(os.path.join(out, "results.json")):
            print(f"step{s}: no results, skipped"); continue
        meta = json.load(open(os.path.join(out, "meta.json")))
        res = json.load(open(os.path.join(out, "results.json")))
        layers, W = meta["layers"], meta["windows"]
        lm = layers[len(layers) // 2]                    # a middle layer for the headline numbers
        row = dict(step=s, layer=lm)
        for ax in meta["axes"]:
            sl = res.get(f"slope|{ax}|{lm}", [np.nan] * 3)
            row[f"{ax}_slope"], row[f"{ax}_lo"], row[f"{ax}_hi"] = sl
            row[f"{ax}_ext_last"] = res.get(f"{ax}|uni|{W[-1]}", np.nan)
            row[f"{ax}_tf_last"] = res.get(f"{ax}|tf{lm}|{W[-1]}", np.nan)
        ap = os.path.join(out, "attn.npz")
        if os.path.exists(ap):
            A = np.load(ap)
            rest = A["rest"].mean(axis=(0, 1, 4)).mean(1)   # mean over scenes, trajectories, heads, layers
            sink = A["sink"].mean(axis=(0, 1, 4)).mean(1)
            lt = np.log10(np.array(W, float))
            row["attn_rest_slope"] = float(np.polyfit(lt, np.log10(rest), 1)[0]) if np.all(rest > 0) else np.nan
            row["attn_rest_last"], row["attn_sink_last"] = float(rest[-1]), float(sink[-1])
            row["attn_uniform_last"] = float(A["uniform"].mean(0)[-1])
            if "vrest" in A.files:                           # value-weighted share (newer attn runs)
                vrest = A["vrest"].mean(axis=(0, 1, 4)).mean(1)
                row["vw_rest_slope"] = float(np.polyfit(lt, np.log10(vrest), 1)[0]) if np.all(vrest > 0) else np.nan
                row["vw_rest_last"] = float(vrest[-1])
        pp = os.path.join(out, "paths.npz")
        if os.path.exists(pp):
            Pz = np.load(pp); lt = np.log10(np.array(W, float))
            for name in ("direct", "relay", "both"):
                key = f"cum_{lm}_{name}"
                if key in Pz.files:
                    row[f"path_alpha_{name}"] = -float(np.polyfit(lt, np.log10(np.maximum(Pz[key], 1e-30)), 1)[0]) / 2
            kd, kr = f"cum_{lm}_direct", f"cum_{lm}_relay"
            if kd in Pz.files:
                row["path_relay_share_last"] = float(Pz[kr][-1] / (Pz[kd][-1] + Pz[kr][-1]))
        rows.append(row)
    if not rows:
        sys.exit("nothing to summarise")

    path = os.path.join(a.root, "summary.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", path)
    for r in rows:
        print(f"step{r['step']:<7d}" + "  ".join(f"{ax}:{r[f'{ax}_slope']:+.2f}" for ax in meta["axes"])
              + (f"   attn_rest_slope:{r.get('attn_rest_slope', np.nan):+.2f}" if "attn_rest_slope" in r else "")
              + (f"   vw_rest_slope:{r.get('vw_rest_slope', np.nan):+.2f}" if "vw_rest_slope" in r else "")
              + (f"   alpha direct/relay/both: {r['path_alpha_direct']:.2f}/{r['path_alpha_relay']:.2f}/"
                 f"{r['path_alpha_both']:.2f}  relay share {r['path_relay_share_last']:.2f}"
                 if "path_alpha_both" in r else ""))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: csv only"); return
    xs = [max(r["step"], 32) for r in rows]            # step0 drawn at x=32, labelled 0
    ticks = sorted(set(xs)); labels = ["0" if t == 32 and 0 in [r["step"] for r in rows] else str(t) for t in ticks]

    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4.5))
    for axn in meta["axes"]:
        y = np.array([r[f"{axn}_slope"] for r in rows])
        lo = np.array([r[f"{axn}_lo"] for r in rows]); hi = np.array([r[f"{axn}_hi"] for r in rows])
        ax1[0].errorbar(xs, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=axn)
    if "attn_rest_slope" in rows[0]:
        ax1[0].plot(xs, [2 * r.get("attn_rest_slope", np.nan) for r in rows], "k--", label="2 x attn-to-prompt slope")
    if any("vw_rest_slope" in r for r in rows):
        ax1[0].plot(xs, [2 * r.get("vw_rest_slope", np.nan) for r in rows], "k:", lw=2,
                    label="2 x value-weighted slope")
    ax1[0].set_xscale("log"); ax1[0].set_xticks(ticks); ax1[0].set_xticklabels(labels, rotation=45)
    ax1[0].set_xlabel("training step"); ax1[0].set_ylabel("d log SNR / d log t  (teacher forced)")
    ax1[0].set_title(f"decay of prompt memory, layer {rows[0]['layer']}"); ax1[0].legend(fontsize=8)
    for axn in meta["axes"]:
        ax1[1].plot(xs, [r[f"{axn}_ext_last"] for r in rows], marker="o", label=f"{axn} (text)")
    ax1[1].axhline(0.5, color="gray", lw=0.8)
    ax1[1].set_xscale("log"); ax1[1].set_xticks(ticks); ax1[1].set_xticklabels(labels, rotation=45)
    ax1[1].set_xlabel("training step"); ax1[1].set_ylabel(f"probe accuracy at t={W[-1]}")
    ax1[1].set_title("what the exterior still shows at late time"); ax1[1].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(a.root, "summary.png"); fig.savefig(png, dpi=150); print("wrote", png)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summarize"])
    p.add_argument("--root", default="pythia")
    p.add_argument("--model", default="EleutherAI/pythia-160m")
    p.add_argument("--steps", type=int, nargs="+", default=STEPS)
    p.add_argument("--win", type=int, default=8)
    p.add_argument("--windows", default="16,32,64,128,256,512")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--hbatch", type=int, default=4)
    p.add_argument("--n_attn", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--redo_attn", action="store_true", help="recompute attn even if attn.npz exists")
    p.add_argument("--paths", action="store_true", help="also run the direct/relay path decomposition")
    p.add_argument("--redo_paths", action="store_true")
    p.add_argument("--n_paths", type=int, default=8)
    a = p.parse_args()
    run(a) if a.cmd == "run" else summarize(a)
