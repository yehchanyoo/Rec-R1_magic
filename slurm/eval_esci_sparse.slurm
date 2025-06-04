#!/bin/bash

#SBATCH --job-name=esci_sparse
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jchae3@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-l40,gpu-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --time=15:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/scrubbed/jchae3/Rec-R1_magic
#SBATCH --export=all
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Modules to use
module load apptainer

# Your programs to run.
srun apptainer exec --nv --pwd /home/rapids/Rec-R1_magic --bind /dev/shm:/dev/shm --bind /gscratch/scrubbed/jchae3/Rec-R1_magic:/home/rapids/Rec-R1_magic --env-file secrets.env /gscratch/scrubbed/jchae3/images/r1.sif bash scripts/repro/eval_esci_sparse.sh
