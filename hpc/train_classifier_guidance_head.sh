#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train_cg
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train_cg_%J.out
#BSUB -e hpc/logs/train_cg_%J.err

# Classifier-guidance head sweep:   bsub < hpc/train_classifier_guidance_head.sh

set -eu
cd /dtu/blackhole/08/223976/02456-autoregressive-project
source hpc/env.sh

BASE=transformer/best_transformer

# num_layers=0 is a single Linear: the linear probe baseline. hidden_units is unused
# there, so it gets its own line to avoid running identical configs three times.
uv run python scripts/classifier/guidance/train_classifier_guidance.py -m \
    trained_model.name=$BASE \
    head.num_layers=0 \
    optimizer=adam optimizer.lr=0.003,0.001,0.0003 \
    train.epochs=20 seed=1,2,3

uv run python scripts/classifier/guidance/train_classifier_guidance.py -m \
    trained_model.name=$BASE \
    head.num_layers=1,2 head.hidden_units=128,256 \
    optimizer=adam optimizer.lr=0.003,0.001,0.0003 \
    train.epochs=20 seed=1,2,3
