#!/bin/bash
#SBATCH --job-name=validate_1d_closed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/validate_1d_closed_%j.log
#SBATCH --error=logs/validate_1d_closed_%j.err
#
# SLURM wrapper for scripts/checks/validate_1d_closed.py.
#
# Parameterised via environment variables so the same script can
# submit the N=8, N=12 and N=16 sweeps separately:
#
#   N_VALUES="8"  OUTPUT_TAG="N8_"  sbatch --export=ALL scripts/maintenance/run_validate_1d_closed.sh
#   N_VALUES="12" OUTPUT_TAG="N12_" sbatch --export=ALL scripts/maintenance/run_validate_1d_closed.sh
#   N_VALUES="16" OUTPUT_TAG="N16_" sbatch --export=ALL scripts/maintenance/run_validate_1d_closed.sh
#
# Or the full sweep in one job:
#   sbatch scripts/maintenance/run_validate_1d_closed.sh

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs data/validation_reports

N_VALUES=${N_VALUES:-"8 12 16"}
ALPHA_VALUES=${ALPHA_VALUES:-"1 2 3 4 6"}
CHI_VALUES=${CHI_VALUES:-"4 8 16 32 64"}
GRID_SIZE=${GRID_SIZE:-20}
OUTPUT_TAG=${OUTPUT_TAG:-""}

TAG_ARG=""
if [ -n "$OUTPUT_TAG" ]; then
    TAG_ARG="--output-tag $OUTPUT_TAG"
fi

python scripts/checks/validate_1d_closed.py \
    --N $N_VALUES \
    --alpha $ALPHA_VALUES \
    --chi $CHI_VALUES \
    --grid-size $GRID_SIZE \
    --verbose \
    $TAG_ARG