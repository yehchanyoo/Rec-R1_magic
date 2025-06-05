#!/bin/bash

#SBATCH --job-name=amazon-review-prep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yehchany@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-l40,gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --time=20:00:00  # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/scrubbed/yehchany/Rec-R1_magic
#SBATCH --export=all
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

module load apptainer

# Run from root of the directory
srun apptainer exec --nv --pwd /home/rapids/Rec-R1_magic \
                    --bind /dev/shm:/dev/shm \
                    --bind /gscratch/scrubbed/yehchany/Rec-R1_magic:/home/rapids/Rec-R1_magic \
                    --env-file secrets.env /gscratch/scrubbed/yehchany/Rec-R1_magic/docker/ngc-vllm.sif \
                    bash -c "bash scripts/repro/amz_review_data_prep_part_1.sh && \
                              bash scripts/repro/amz_review_data_prep_part_2.sh && \
                              bash scripts/repro/amz_review_train_eval.sh" 
