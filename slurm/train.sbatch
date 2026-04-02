#!/bin/bash
#SBATCH --partition=gpu          # For how long can it run?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --nodelist=g64-01
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1              # How many cpus per mpi proc
#SBATCH --cpus-per-task=64              # How many cpus per mpi proc
#SBATCH --job-name=train-ai2
#SBATCH --output=/home/cld9301/WORK/repos/ai2/slurm/output/train/%j/output.%j        # Where to write results
#SBATCH --error=/home/cld9301/WORK/repos/ai2/slurm/output/train/%j/error.%j

datetime=$1
task=$2

echo "Running task: $task in folder: $datetime"

# Make the GPUs visible
export CUDA_VISIBLE_DEVICES=0

nvidia-smi

source $HOME/.bashrc
micromamba activate ai2

work_dir="/mnt/local_scratch/work/ai2"
source_dir="/home/cld9301/WORK/repos/ai2"

rsync -avu --delete $source_dir/transfer/$datetime $work_dir

output_dir=$source_dir/slurm/output/train/$SLURM_JOBID

cd $source_dir/transfer/$datetime

time python -u ai2/$task.py >> $output_dir/$task.log

rsync -avu "$work_dir/checkpoints/" "$source_dir/checkpoints"
rsync -avu "$work_dir/lightning_logs/" "$source_dir/lightning_logs"
rsync -avu "$work_dir/wandb/" "$source_dir/wandb"