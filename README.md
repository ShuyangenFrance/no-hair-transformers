# A Conditional No-Hair Theorem for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and paper for *A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt*.

## Paper

- PDF: [`article/main.pdf`](article/main.pdf)
- LaTeX: [`article/main.tex`](article/main.tex)
- Figures: [`article/figures/`](article/figures/)

## Getting started

```bash
pip install -r requirements.txt
```

## Core toolkit

| Script | Purpose |
|---|---|
| [`src/real_run.py`](src/real_run.py) | Measure prompt offset under teacher forcing with unbiased estimator, octave windows, global/local exponents, bootstrap |
| [`src/scale_run.py`](src/scale_run.py) | Same as real_run.py for hand-written scenes |
| [`src/attn_stats.py`](src/attn_stats.py) | Attention statistics (entropy, spread, sinks, mid-block norms) |
| [`src/real_mech.py`](src/real_mech.py) | Direct/relay decomposition and dilution compensation |
| [`src/fit_gains.py`](src/fit_gains.py) | Fit per-layer gains from exact linear relay solution |
| [`src/tradeoff_real.py`](src/tradeoff_real.py) | Retention/perplexity trade-off with placebo control |
| [`src/behavior_test.py`](src/behavior_test.py) | Single-turn instruction adherence tests |
| [`src/multiturn_test.py`](src/multiturn_test.py) | Multi-turn instruction adherence tests |
| [`src/make_figures.py`](src/make_figures.py) | Reproduce paper figures |

## Utilities

- [`src/nohair_gen_v2.py`](src/nohair_gen_v2.py), [`src/tf_controls.py`](src/tf_controls.py) — model loading, chunking, helpers
- [`src/nohair_gen_v1.py`](src/nohair_gen_v1.py), [`src/pythia_sweep.py`](src/pythia_sweep.py) — auxiliary experiments

## Usage example

```bash
python src/real_run.py --model EleutherAI/pythia-160m --out results/p160_step0
python src/real_run.py --out results/p160_step0 --analyze_only
```

## Citation

```bibtex
@article{xiang2026conditional,
  title={A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt},
  author={Xiang, Shuyang},
  year={2026}
}
```

## License

MIT - see [`LICENSE`](LICENSE).
