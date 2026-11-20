#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
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

# MLP (model.d_emb): 16, 32, 64
# With only about 23 tokens and a 3-token context, a few dimensions are enough to describe each amino acid. These values test whether going smaller or bigger than your current 32 changes anything.

# MLP (model.d_hidden): 128, 256, 512

# d_hidden is the MLP's capacity: how much it can learn about which next token follows each 3-token context. These values go half and double around your current 256. If the loss stops improving from 256 to 512, the 3-token context is the limit and not the model size.

uv run python scripts/generative/train.py -m model=mlp seed=1,2,3 optimizer=adam optimizer.lr=0.0003,0.001,0.003 train.epochs=100 model.d_emb=16,32,64 model.d_hidden=128,256,512 model.dropout=0,0.1,0.3

uv run python scripts/generative/train.py -m model=mlp seed=1,2,3 optimizer=sgd  optimizer.lr=0.01,0.03,0.1 train.epochs=100 model.d_emb=16,32,64 model.d_hidden=128,256,512 model.dropout=0,0.1,0.3



# uv run python scripts/generative/train.py model=transformer optimizer=sgd train.epochs=100
# uv run python scripts/generative/train.py -m model=mlp optimizer=adam optimizer.lr=0.0003,0.001,0.003 train.epochs=50,100
# uv run python scripts/generative/train.py -m model=mlp optimizer=sgd optimizer.lr=0.01,0.03,0.1 optimizer.momentum=0.0,0.9 train.epochs=50,100
