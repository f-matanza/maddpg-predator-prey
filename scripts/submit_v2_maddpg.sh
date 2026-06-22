#!/bin/bash
#SBATCH --job-name=maddpg_v2
#SBATCH --gpus=1
#SBATCH --output=logs/slurm-out-%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="$PWD"
PYTHON="$PROJECT_DIR/.pixi/envs/default/bin/python"
export PYTHONPATH="$PROJECT_DIR"

JOB_TOKEN="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-src_v2_maddpg_slow_${JOB_TOKEN}}"
EVAL_TAG="${EVAL_TAG:-${JOB_TOKEN}}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_FPS="${EVAL_FPS:-10}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
echo
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Run name: ${RUN_NAME}"
echo "Eval tag: ${EVAL_TAG}"
echo "Python Version:"
"$PYTHON" --version

echo "==============================="
echo "Training started: $(date)"

srun --cpu-bind=cores "$PYTHON" -u -B -m src_v2.train --algorithm maddpg --run-name "$RUN_NAME"

echo "Training finished: $(date)"
echo "==============================="
echo "Evaluation started: $(date)"

srun --cpu-bind=cores "$PYTHON" -u -B -m src_v2.evaluate --algorithm maddpg --run-name "$RUN_NAME" --tag "$EVAL_TAG" --num-episodes "$EVAL_EPISODES" --fps "$EVAL_FPS"

echo "Evaluation finished: $(date)"
echo
echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
