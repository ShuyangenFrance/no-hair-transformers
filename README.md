# A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt

This repository contains the code and paper for the work investigating how autoregressive transformers forget their prompts over long generations.

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Measure prompt offset

Measure the decay exponent $\alpha$ for a model on real-corpus prompt pairs:

```bash
python src/real_run.py --model EleutherAI/pythia-160m --out results/p160_step0
```

Analyze the saved results:

```bash
python src/real_run.py --out results/p160_step0 --analyze_only
```

### 3. Measure on hand-written scenes

```bash
python src/scale_run.py --model EleutherAI/pythia-160m --out results/p160_scenes
```

### 4. Compute attention statistics

```bash
python src/attn_stats.py --model EleutherAI/pythia-160m --pairs_from results/p160_step0
```

### 5. Decompose and compensate dilution

```bash
python src/real_mech.py --model EleutherAI/pythia-160m --pairs_from results/p160_step0
```

### 6. Fit gains from exact solution

```bash
python src/fit_gains.py results/p160_step0
```

### 7. Reproduce figures

```bash
python src/make_figures.py
```

## 🗂️ Repository Structure

```
no-hair-transformers/
├── article/           # Paper (PDF, LaTeX, figures)
├── prompts/           # Prompt sets used in experiments
├── results/           # Output results from runs
├── src/               # Toolkit scripts
└── requirements.txt
```

## 🧰 Toolkit Reference

| Script | Description |
|---|---|
| `real_run.py` | Core measurement: teacher-forced offset, unbiased signal estimator, octave windows, global/local $\alpha$, bootstrap intervals |
| `scale_run.py` | Same measurement for 20 hand-written scenes with 5 axes each |
| `attn_stats.py` | Attention entropy, logit spread, attention to sinks, mid-block norms (for sequential blocks) |
| `real_mech.py` | Direct/relay path decomposition; uniform and per-head dilution compensation |
| `fit_gains.py` | Fit layer gains $g_1,\ldots,g_L$ to curves using exact linear relay recursion |
| `tradeoff_real.py` | Retention vs perplexity trade-off with placebo control |
| `behavior_test.py` | Single-turn instruction adherence under dilution boosts |
| `multiturn_test.py` | Multi-turn conversation adherence tests |
| `make_figures.py` | Generate figures from computed results |
| `nohair_gen_v2.py`, `tf_controls.py` | Model loading, data chunking, and utilities |
| `nohair_gen_v1.py`, `pythia_sweep.py` | Auxiliary/legacy experiments |

## ⚙️ Key Arguments

- `--model` — Hugging Face model ID (e.g., `EleutherAI/pythia-160m`, `Qwen/Qwen2.5-0.5B`)
- `--revision step0|step143000` — Use specific Pythia checkpoint
- `--random_init` — Use random initialization instead of pretrained weights
- `--windows` — Measurement times in tokens (default: `16,32,64,128,256,512,1024`)
- `--label center|end` — Window time labeling (default: `center`)
- `--estimator unbiased|plugin` — Signal estimator (default: `unbiased`)
- `--cont wiki|shuffled` — Continuation text type (default: `wiki`)
- `--prefix <text>` — Prepend common token to make sink shared

See `-h` for each script for full details.

## 📚 Citation

```bibtex
@article{xiang2026conditional,
  title={A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt},
  author={Xiang, Shuyang},
  year={2026}
}
```

## 📜 License

MIT License. See [`LICENSE`](LICENSE) for details.
