# A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt

This repository contains the code and paper for the work investigating how autoregressive transformers forget their prompts over long generations.

## 📄 Paper

- **PDF**: [`article/main.pdf`](article/main.pdf)
- **LaTeX source**: [`article/main.tex`](article/main.tex)
- **Figures**: [`article/figures/`](article/figures/)

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

## 🗂️ Repository Structure

```
no-hair-transformers/
├── article/           # Paper (PDF, LaTeX, figures)
├── results/           # Output results from runs
├── src/               # Toolkit scripts
└── requirements.txt
```

## 🧰 Toolkit Reference

| Script | Description |
|---|---|
| `real_run.py` | Core measurement: teacher-forced offset, unbiased signal estimator, octave windows, global/local $\alpha$, bootstrap intervals |
| `scale_run.py` | Same measurement for hand-written scenes |
| `attn_stats.py` | Attention statistics (entropy, spread, sinks, mid-block norms) |
| `real_mech.py` | Direct/relay path decomposition; dilution compensation |
| `fit_gains.py` | Fit per-layer gains $g_1,\ldots,g_L$ from exact linear relay solution |
| `tradeoff_real.py` | Retention vs perplexity trade-off with placebo control |
| `behavior_test.py` | Single-turn instruction adherence tests |
| `multiturn_test.py` | Multi-turn conversation adherence tests |
| `make_figures.py` | Reproduce paper figures |

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
