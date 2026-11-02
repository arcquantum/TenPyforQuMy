#!/bin/bash
#SBATCH --job-name=echo_array_2
#SBATCH --output=logs/echo2_%A_%a.log
#SBATCH --array=0-799%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00

# Prevent thread-thrashing by locking C-libraries to allocated CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

eval "$(conda shell.bash hook)"
conda activate metrology
cd ~/metrology_thesis

mkdir -p data/echo_pts

# Calculate the shifted task ID for the second half of the 1600-point grid
REAL_TASK_ID=$(($SLURM_ARRAY_TASK_ID + 800))
EXPECTED_FILE="data/echo_pts/pt_${REAL_TASK_ID}.npy"

# Smart Resume: Only run python if this specific grid point is missing
if [ ! -f "$EXPECTED_FILE" ]; then
    python scripts/echo_worker.py --task_id $REAL_TASK_ID --N 32 --chi_max 256 --grid_size 40
else
    echo "Task ${REAL_TASK_ID} already complete. Skipping."
fi