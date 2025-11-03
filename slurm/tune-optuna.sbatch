#!/bin/bash
#SBATCH --partition=normal          # For how long can it run?
#SBATCH --time=08:00:00          # For how long can it run?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --ntasks=1              # How many cpus per mpi proc
#SBATCH --cpus-per-task=38              # How many cpus per mpi proc
#SBATCH --mail-type=END
#SBATCH --mail-user=lars.dammann@tuhh.de
#SBATCH --job-name=tune-ai2
#SBATCH --output=/home/hgf_desy/hgf_yqq0513/repos/ai2/slurm/output/%j/output.%j        # Where to write results
#SBATCH --error=/home/hgf_desy/hgf_yqq0513/repos/ai2/slurm/output/%j/error.%j

export CUDA_VISIBLE_DEVICES=0    # Very important to make the GPUs visible
# export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

nvidia-smi

source $HOME/.bashrc
micromamba activate ai2

cd $HOME/repos/ai2/

output_dir=slurm/output/$SLURM_JOBID
# mkdir $output_dir

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > $output_dir/results-file.csv &

time python -u ai2/tune-optuna.py >> $output_dir/tune.log