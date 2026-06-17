#!/bin/bash
#SBATCH --job-name=maddpg_01
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=slurm-out-%x.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="$PWD"
PYTHON="$PROJECT_DIR/.pixi/envs/default/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo -e "–" * 81
echo
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Python Version:"
"$PYTHON" --version

echo "==============================="
echo -e "Training started: $(date)\n"

srun --cpu-bind=cores "$PYTHON" -u -m src.train --algorithm maddpg

echo -e "\nTraining finished: $(date)"
echo "==============================="
echo -e "Evaluation started: $(date)\n"

srun --cpu-bind=cores xvfb-run -a "$PYTHON" -u -m src.evaluate --algorithm maddpg

echo -e "\nEvaluation finished: $(date)"
echo
echo -e "–" * 81
