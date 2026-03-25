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

work_dir="/mnt/local_scratch/work/ai2"
source_dir="/home/cld9301/WORK/repos/ai2"

rsync -avu --delete $source_dir/ai2 $work_dir
rsync -avu --delete $source_dir/configs $work_dir

output_dir=$source_dir/slurm/output/train/$SLURM_JOBID

time python -u ai2/tune.py >> $output_dir/tune.log

rsync -avu "$work_dir/checkpoints/" "$source_dir/checkpoints"
rsync -avu "$work_dir/lightning_logs/" "$source_dir/lightning_logs"
rsync -avu "$work_dir/wandb/" "$source_dir/wandb"