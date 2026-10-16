# A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt

[![arXiv](https://img.shields.io/badge/arXiv-26XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/26XX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code, data, and paper for:

> Shuyang Xiang. *A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt*. 2026.

## Overview

We measure how much an autoregressive transformer retains of its prompt after generating $t$ tokens by feeding two different prompts the same continuation text and tracking the distance between their hidden states. We find the prompt offset decays approximately as a power law $t^{-\alpha}$ and prove a conditional no-hair theorem: under mild assumptions, the offset is $O(P(\log t)^{L-1}/t)$. We provide a toolkit to measure $\alpha$ for any Hugging Face causal language model.

- **Paper**: [`article/main.pdf`](article/main.pdf) | [`article/main.tex`](article/main.tex)
- **Figures**: [`article/figures/`](article/figures/)

## Repository layout

- `src/` or top-level scripts? See below. Core measurement scripts:
  - `real_run.py` — teacher-forced prompt offset measurement with unbiased estimator, octave windows, global/local exponents, bootstrap
  - `scale_run.py` — same as `real_run.py` but for the hand-written scenes
  - `attn_stats.py` — attention statistics (entropy, spread, sinks, mid-block norms)
  - `real_mech.py` — direct/relay decomposition and dilution compensation
  - `fit_gains.py` — fit gains $g_1,\ldots,g_L$ to the measured offset curves from the exact linear relay solution
  - `tradeoff_real.py` — retention/perplexity trade-off with placebo control
  - `behavior_test.py`, `multiturn_test.py` — behavioural tests (single-turn and multi-turn)
  - `make_figures.py` — reproduce figures from results
  - `nohair_gen_v2.py`, `tf_controls.py` — utilities (model loading, chunking, etc.)
- `article/` — LaTeX source and figures
- `prompts/` — prompt sets used in the paper (to be added)
- `results/` — example result files (to be added)

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- NumPy, SciPy, matplotlib, pandas
- tqdm

Install with:

```bash
pip install torch transformers numpy scipy matplotlib pandas tqdm
```

## Quick start

1. Measure prompt offset on real-corpus pairs:
```bash
python real_run.py --model EleutherAI/pythia-160m --out real/p160_step0
```

2. Analyze and save summary:
```bash
python real_run.py --out real/p160_step0 --analyze_only
```

3. Reproduce figures:
```bash
python make_figures.py
```

See individual script help (`-h`) for full options.

## Prompts and results

- Hand-written scenes and real-corpus pair specifications: see [`article/main.tex`](article/main.tex) Appendix C and the code in `real_run.py`/`scale_run.py`.
- Example outputs can be placed under `results/`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or find the results useful, please cite:

```bibtex
@article{xiang2026conditional,
  title={A Conditional No-Hair Theorem for Transformers: How Language Models Forget Their Prompt},
  author={Xiang, Shuyang},
  year={2026}
}
```

