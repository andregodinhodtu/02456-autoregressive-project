#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/logs/train_%J.out
#BSUB -e hpc/logs/train_%J.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd /dtu/blackhole/12/223826/02456-autoregressive-project
source hpc/env.sh   # uv on the PATH, caches off the home quota

# First sweep with adam
uv run python scripts/generative/train.py -m model=transformer \
    model.num_layers=1,2,4 model.embedding_dim=128,256,512 model.num_heads=2,4,8 \
    optimizer=adam optimizer.lr=0.0001,0.0003,0.001 \
    train.epochs=50 train.val_every_step=200


# Second sweep with SGD
uv run python scripts/generative/train.py -m model=transformer \
    model.num_layers=1,2,4 model.embedding_dim=128,256,512 model.num_heads=2,4,8 \
    optimizer=sgd optimizer.lr=0.01,0.05,0.1 optimizer.momentum=0.9 \
    train.epochs=70 train.val_every_step=200

