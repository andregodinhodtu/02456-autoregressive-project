#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval_mlp
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/eval_mlp_%J.out
#BSUB -e hpc/logs/eval_mlp_%J.err

# Score sample sets on a GPU, structures included:   bsub < hpc/eval.sh
#
# Run `bash hpc/setup_env.sh` once before this, to install the environment and download the
# ESMFold weights. Add your own sample CSVs to the list below; a name with no CSV is skipped with
# a message rather than failing the job, so this can be submitted before every model has been
# sampled. Each set lands in results/eval/<name>/.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd /dtu/blackhole/08/223976/02456-autoregressive-project
source hpc/env.sh   # uv on the PATH, caches off the home quota

# MLP baseline: 256 generated sequences
uv run --group eval python eval/eval.py --samples results/sequences/mlp/simple_mlp_baseline_256seq.csv --fold

# Controls: 256 real test proteins and the same proteins shuffled
uv run --group eval python eval/eval.py --samples results/sequences/control_real.csv --fold
uv run --group eval python eval/eval.py --samples results/sequences/control_shuffled.csv --fold
