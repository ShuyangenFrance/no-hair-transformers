#!/usr/bin/env python3
"""init_std.py — standard deviation of the step-0 weights of block 0 in Pythia-160m/410m/1b.
Answers: does the initial scale of the output projections depend on the total depth (wang_init)?"""
from transformers import GPTNeoXForCausalLM
for m in ["pythia-160m", "pythia-410m", "pythia-1b"]:
    net = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{m}", revision="step0")
    L, d = net.config.num_hidden_layers, net.config.hidden_size
    print(f"\n{m}: layers={L} width={d}  small_init={(2 / (5 * d)) ** 0.5:.5f}  wang_init={2 / (L * d ** 0.5):.5f}")
    for n, p in net.named_parameters():
        if n.startswith("gpt_neox.layers.0.") and n.endswith("weight") and p.dim() == 2:
            print(f"  {n:55s} std={p.std().item():.5f}")
