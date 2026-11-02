#!/bin/bash
#SBATCH --job-name=val_indep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/val_indep_%j.log
#SBATCH --error=logs/val_indep_%j.err
#
# Step 1.1 independence and algebra checks. Single job, one task.
# Runs dt and svd_min independence plus the [S_x, S_y] = i S_z
# algebra check at N=16, alpha=1, chi=32 (chi=32 is 2x the
# chi_conv=8 that we already measured at N=8; scaling predicts
# chi_conv=16-32 at N=16, so chi=32 is the appropriate test point).

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p logs data/validation_reports

# The independence checks are hard-wired inside validate_1d_closed.py
# at INDEP_N=16, INDEP_ALPHA=1.0, INDEP_CHI=64. Override to chi=32
# via a small sed on a copy of the constants — or, simpler, run at
# chi=32 by calling the independence functions directly.
python - <<'EOF'
import sys
sys.path.insert(0, '.')
import scripts.checks.validate_1d_closed as v

# Override the module constants to the appropriate Step 1.1 test point.
v.INDEP_N = 16
v.INDEP_ALPHA = 1.0
v.INDEP_CHI = 32
v.INDEP_GRID = 5

import os
out_dir = os.path.join('data', 'validation_reports')
os.makedirs(out_dir, exist_ok=True)
tag = 'indep_'

print("=== dt independence ===")
v.dt_independence_check([v.INDEP_ALPHA], v.INDEP_N, v.INDEP_CHI,
                        v.INDEP_GRID, out_dir, tag)
print("=== svd_min independence ===")
v.svdmin_independence_check([v.INDEP_ALPHA], v.INDEP_N, v.INDEP_CHI,
                             v.INDEP_GRID, out_dir, tag)
print("=== algebra check ===")
v.algebra_check(N=v.INDEP_N, alpha=v.INDEP_ALPHA, chi=v.INDEP_CHI,
                dt=0.02, output_dir=out_dir, tag=tag)
print("=== all independence checks done ===")
EOF