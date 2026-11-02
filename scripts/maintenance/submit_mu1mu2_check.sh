#!/bin/bash
#SBATCH --job-name=mu1mu2_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/mu1mu2_%A_%a.log
#SBATCH --error=logs/mu1mu2_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis
mkdir -p logs plots data/exact_1d_closed data/powerlaw_echo_pts

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

case $SLURM_ARRAY_TASK_ID in
    0) N=4; A=1.0; CHI=32 ;;
    1) N=4; A=3.0; CHI=32 ;;
    2) N=8; A=1.0; CHI=64 ;;
    3) N=8; A=3.0; CHI=64 ;;
esac

python3 scripts/checks/compare_mu1mu2_pipeline.py \
    --N $N --alpha $A --chi $CHI --grid_res 20 --n_cores 16