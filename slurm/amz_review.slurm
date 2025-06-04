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

# Before running the data‐prep scripts inside the container, we need to use the overlay
# and bind-mounts exactly as in:
#   apptainer shell \
#     --overlay /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/overlay.img \
#     --nv \
#     --bind /dev/shm:/dev/shm \
#     --env-file /gscratch/scrubbed/yehchany/Rec-R1_Magic/secrets.env \
#     /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/ngc-vllm.sif
#
# In this batch job, we’ll run each script via `apptainer exec` with the same overlay + binds.

# Run Part 1 of the Amazon Review data‐prep pipeline:
srun apptainer exec \
     --nv \
     --overlay /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/overlay.img \
     --bind /dev/shm:/dev/shm \
     --bind /gscratch/scrubbed/yehchany/Rec-R1_magic:/home/rapids/Rec-R1_magic \
     --env-file /gscratch/scrubbed/yehchany/Rec-R1_Magic/secrets.env \
     /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/ngc-vllm.sif \
     bash /home/rapids/Rec-R1_magic/scripts/repro/amz_review_data_prep_part_1.sh

# Run Part 2 of the Amazon Review data‐prep pipeline, which also initiates training:
srun apptainer exec \
     --nv \
     --overlay /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/overlay.img \
     --bind /dev/shm:/dev/shm \
     --bind /gscratch/scrubbed/yehchany/Rec-R1_magic:/home/rapids/Rec-R1_magic \
     --env-file /gscratch/scrubbed/yehchany/Rec-R1_Magic/secrets.env \
     /gscratch/scrubbed/yehchany/Rec-R1_Magic/docker/ngc-vllm.sif \
     bash /home/rapids/Rec-R1_magic/scripts/repro/amz_review_data_prep_part_2.sh
