#!/bin/bash
# Re-run the path decomposition and the trade-off with centre labels and the unbiased signal.
mkdir -p logs
run () { name=$1; shift; echo "=== $name  $(date)"; python "$@" > logs/$name.txt 2>&1 && echo "    done" || echo "    FAILED (see logs/$name.txt)"; }
run mech_step0   real_mech.py --model EleutherAI/pythia-160m --revision step0      --pairs_from real/p160_step0 --out mech2/p160_step0
run mech_step512 real_mech.py --model EleutherAI/pythia-160m --revision step512    --pairs_from real/p160_step0 --out mech2/p160_step512
run mech_final   real_mech.py --model EleutherAI/pythia-160m --revision step143000 --pairs_from real/p160_step0 --out mech2/p160_final
run tradeoff2    tradeoff_real.py --model EleutherAI/pythia-160m --revision step143000 --pairs_from real/p160_step0 --out tradeoff2/p160_final
echo "=== all finished  $(date)"
