#!/bin/bash
# Overnight batch. Each step logs to logs/<name>.txt; a failed step does not stop the next ones.
mkdir -p logs
run () { name=$1; shift; echo "=== $name  $(date)"; python "$@" > logs/$name.txt 2>&1 && echo "    done" || echo "    FAILED (see logs/$name.txt)"; }

# 1-2. Qwen2.5-0.5B on the real-corpus pairs, trained and randomly initialised, windows up to 2048
run qwen05_trained real_run.py --model Qwen/Qwen2.5-0.5B --out real/qwen05_final --windows 16,32,64,128,256,512,1024,2048
run qwen05_random  real_run.py --model Qwen/Qwen2.5-0.5B --random_init --out real/qwen05_random --windows 16,32,64,128,256,512,1024,2048

# 3. non-degeneracy check for Qwen (RMSNorm)
run qwen05_prenorm prenorm_check.py --model Qwen/Qwen2.5-0.5B --pairs_from real/qwen05_final --rms

# 4. trade-off + placebo on real pairs, Pythia-160m final
run tradeoff_p160 tradeoff_real.py --model EleutherAI/pythia-160m --revision step143000 --pairs_from real/p160_step0 --out tradeoff/p160_final

# 5. if time remains: Qwen2.5-1.5B base, trained only
run qwen15_trained real_run.py --model Qwen/Qwen2.5-1.5B --out real/qwen15_final --windows 16,32,64,128,256,512,1024,2048

echo "=== all finished  $(date)"
