#!/bin/bash
# Everything still needed for the paper's TODOs. Each step logs to todo_logs/<name>.txt; failures do not stop the rest.
mkdir -p todo_logs
run () { name=$1; shift; echo "=== $name  $(date)"; python "$@" > todo_logs/$name.txt 2>&1 && echo "    done" || echo "    FAILED (see todo_logs/$name.txt)"; }

# 1. Pythia initialisation scale (minutes)
run init_std        init_std.py

# 2. attention statistics across training: sink mass, logit spread Delta (about 1 h in total)
for s in 0 128 256 512 1000 4000 16000 143000; do
  run attn_p160_step$s attn_stats.py --model EleutherAI/pythia-160m --revision step$s --pairs_from real/p160_step0 --n_prompts 400
done

# 3. untrained / trained Qwen: attention entropy, max weight, mid-block residual (minutes each)
run attn_qwen05_random attn_stats.py --model Qwen/Qwen2.5-0.5B --random_init --pairs_from real/p160_step0
run attn_qwen05_final  attn_stats.py --model Qwen/Qwen2.5-0.5B --pairs_from real/p160_step0

# 4. every layer of untrained Pythia-1b, for the layer-by-layer gain fit (a few hours, run last)
run p1b_step0_all   real_run.py --model EleutherAI/pythia-1b --revision step0 --out real/p1b_step0_all --layers 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

echo "=== all finished  $(date)"
for f in todo_logs/*.txt; do echo "===== $f"; tail -45 "$f"; echo; done > todo_results.txt
