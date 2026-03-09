#!/bin/bash
#SBATCH --partition=gpu          # For how long can it run?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --nodelist=g64-01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1              # How many cpus per mpi proc
#SBATCH --cpus-per-task=64              # How many cpus per mpi proc
#SBATCH --job-name=tune-ai2
#SBATCH --output=/home/cld9301/WORK/repos/ai2/slurm/output/train/%j/output.%j        # Where to write results
#SBATCH --error=/home/cld9301/WORK/repos/ai2/slurm/output/train/%j/error.%j

export CUDA_VISIBLE_DEVICES=0    # Very important to make the GPUs visible

nvidia-smi

source $HOME/.bashrc
micromamba activate ai2

output_dir=slurm/output/train/$SLURM_JOBID
# mkdir $output_dir

time python -u ai2/tune.py >> $output_dir/tune.log