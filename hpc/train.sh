#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train_%J.out
#BSUB -e hpc/logs/train_%J.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd /dtu/blackhole/08/223976/02456-autoregressive-project
source hpc/env.sh   # uv on the PATH, caches off the home quota

# uv run python scripts/generative/train.py -m model=mlp optimizer=adam optimizer.lr=0.0003,0.001,0.003 train.epochs=20
# uv run python scripts/generative/train.py -m model=mlp optimizer=sgd  optimizer.lr=0.01,0.03,0.1      train.epochs=20


uv run python scripts/generative/train.py model=transformer optimizer=sgd train.epochs=100
