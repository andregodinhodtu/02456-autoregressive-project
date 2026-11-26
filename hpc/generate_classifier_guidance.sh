#!/bin/sh
#BSUB -q gpuv100
#!/bin/sh
#BSUB -q gpuv100
#BSUB -J gen_cg
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/logs/gen_cg_%J.out
#BSUB -e hpc/logs/gen_cg_%J.err

# Classifier-guided generation:   bsub < hpc/generate_classifier_guidance.sh
#
# Memory note: the guided path scores every candidate continuation, so each step runs a
# forward pass on n_seq * vocab_size = 256 * 23 = 5888 sequences. That is ~23x the unguided
# cost and the attention scores grow with T^2, so the last steps are the expensive ones.
# If this OOMs on the V100, halve generate.n_seq rather than the walltime.
#
# Walltime: 15 guided runs, sequential (Hydra's default launcher). 8:00 is a guess with
# margin - check the Duration line in the log and tighten it on the next submission.

set -eu
cd /dtu/blackhole/08/223976/02456-autoregressive-project
source hpc/env.sh

TRANSFORMER=best_transformer
HEAD=best_head_for_best_transformer

# The unguided baseline, and the lambda = 0 point of every curve. Every guided run is only
# interpretable against this, and it costs 1/23 of one guided run.
uv run python scripts/classifier/guidance/generate_classifier_guidance.py \
    trained_models.transformer=$TRANSFORMER \
    trained_models.head=$HEAD \
    generate.family=null

# The lambda / family sweep for the report: does guidance trade sequence quality for family
# accuracy, and where does it break? 3 families x 5 lambdas = 15 runs, into sweeps/.
# lambda=0 is not in the grid because it is identical to the baseline above.
uv run python scripts/classifier/guidance/generate_classifier_guidance.py -m \
    trained_models.transformer=$TRANSFORMER \
    trained_models.head=$HEAD \
    generate.family=AcyP,HPr,HU \
    generate.guidance_scale=0.5,1.0,2.0,5.0,10.0
