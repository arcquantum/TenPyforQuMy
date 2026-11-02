#!/bin/bash
#SBATCH --job-name=check_tjm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/check_tjm_%j.log
#SBATCH --error=logs/check_tjm_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd ~/metrology_thesis

mkdir -p logs
python scripts/checks/check_tjm.py
