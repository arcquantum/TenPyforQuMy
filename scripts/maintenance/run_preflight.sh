#!/bin/bash
#SBATCH --job-name=preflight_wrapper
# Wrapper script for scripts/checks/preflight.py.
#
# Usage:
#   sbatch scripts/maintenance/run_preflight.sh 1d_open
#   sbatch scripts/maintenance/run_preflight.sh 2d_closed
#   sbatch scripts/maintenance/run_preflight.sh 2d_open
#
# The first positional argument is passed to --classes.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <class> [additional preflight args...]"
    echo "  <class> in {1d_closed, 1d_open, 2d_closed, 2d_open}"
    exit 2
fi

CLASS="$1"
shift

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis

python scripts/checks/preflight.py --classes "$CLASS" "$@"