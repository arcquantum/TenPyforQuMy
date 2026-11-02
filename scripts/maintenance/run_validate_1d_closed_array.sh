#!/bin/bash
#SBATCH --job-name=val_1d_closed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/val_1d_closed_%A_%a.log
#SBATCH --error=logs/val_1d_closed_%A_%a.err
#
# Validation array. Each array task runs the full grid=20
# chi-scan for one (alpha, N) tuple.
#
# Task id -> (alpha, N):
#    0-4:  alpha in {1, 2, 3, 4, 6},  N = 8
#    5-9:  alpha in {1, 2, 3, 4, 6},  N = 12
#   10-14: alpha in {1, 2, 3, 4, 6},  N = 16
#
# Each task runs 4 chi values (4, 8, 16, 32) at grid=20.
# Wall time per task: ~8 h (N=8), ~12 h (N=12), ~16 h (N=16).
# The 24 h ceiling is comfortable.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs data/validation_reports

ALPHAS=(1 2 3 4 6)
NS=(8 12 16)
CHIS="4 8 16 32"
GRID_SIZE=20

IDX=${SLURM_ARRAY_TASK_ID}
ALPHA_IDX=$(( IDX % 5 ))
N_IDX=$(( IDX / 5 ))
ALPHA=${ALPHAS[$ALPHA_IDX]}
N=${NS[$N_IDX]}

TAG="val_N${N}_a${ALPHA}_"
echo "[task ${IDX}] alpha=${ALPHA}, N=${N}, chi=${CHIS}, grid=${GRID_SIZE}"

python scripts/checks/validate_1d_closed.py \
    --N ${N} \
    --alpha ${ALPHA} \
    --chi ${CHIS} \
    --grid-size ${GRID_SIZE} \
    --skip-independence \
    --skip-algebra \
    --output-tag "${TAG}"