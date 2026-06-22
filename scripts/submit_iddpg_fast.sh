#!/bin/bash
#SBATCH --job-name=iddpg_fast
#SBATCH --gpus=1
#SBATCH --output=logs/slurm-out-%x.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="$PWD"
PYTHON="$PROJECT_DIR/.pixi/envs/default/bin/python"
export PYTHONPATH="$PROJECT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"
echo
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Python Version:"
"$PYTHON" --version

echo "==============================="
echo -e "Training started: $(date)\n"

srun --cpu-bind=cores "$PYTHON" -u -B -m src.train_fast --algorithm iddpg --run-name fast_run

echo -e "\nTraining finished: $(date)"
echo "==============================="
echo -e "Evaluation started: $(date)\n"

srun --cpu-bind=cores "$PYTHON" -u -B -m src.evaluate --algorithm iddpg --run-name fast_run --num-episodes 1 --fps 10

echo -e "\nEvaluation finished: $(date)"
echo "==============================="
echo -e "Plotting reward comparison: $(date)\n"

if [[ -f results/maddpg_fast_run_rewards.csv && -f results/iddpg_fast_run_rewards.csv ]]; then
    PYTHONPATH=. "$PYTHON" -u -B -m src.plot_results \
      --maddpg-csv results/maddpg_fast_run_rewards.csv \
      --iddpg-csv results/iddpg_fast_run_rewards.csv \
      --output results/reward_comparison_fast_run.png
else
    echo "Skipping plot: one comparison CSV is still missing."
fi
echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"

