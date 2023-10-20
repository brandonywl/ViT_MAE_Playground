#!/bin/sh
#SBATCH --time=500
#SBATCH --job-name=vitScratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=branywl@comp.nus.edu.sg
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --output=/home/b/branywl/CS5242/slurm_output/slurm-%j-wbc1scratch.out

srun python3 ./ViTClassifier_scratch_WBC.py wbc1
