#!/bin/bash
#SBATCH --partition=interactive          # For how long can it run?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --ntasks=1              # How many cpus per mpi proc
#SBATCH --cpus-per-task=24              # How many cpus per mpi proc
#SBATCH --job-name=predict-ai2
#SBATCH --output=/home/cld9301/WORK/repos/ai2/slurm/output/predict/%j/output.%j        # Where to write results
#SBATCH --error=/home/cld9301/WORK/repos/ai2/slurm/output/predict/%j/error.%j

export CUDA_VISIBLE_DEVICES=0    # Very important to make the GPUs visible

source $HOME/.bashrc
micromamba activate ai2

output_dir=slurm/output/predict/$SLURM_JOBID
# mkdir $output_dir

time python -u ai2/predict.py >> $output_dir/predict.log