#!/bin/sh
#SBATCH --time=500
#SBATCH --job-name=maeScratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=branywl@comp.nus.edu.sg
#SBATCH --gpus=1
#SBATCH --partition=long
#SBATCH --output=/home/b/branywl/CS5242/slurm_output/slurm-%j-maewbc1.out

srun python3 ./MAE_scratch_WBC.py wbc1
