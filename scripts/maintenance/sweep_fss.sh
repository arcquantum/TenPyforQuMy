#!/bin/bash

N_VALUES=(10 32 64 128 200)
CHI_VALUES=(16 64 128 256)

mkdir -p logs

echo "Starting Unified OAT Parameter Sweep on Regulus (14 Days)..."
echo "--------------------------------------------------------"

for N in "${N_VALUES[@]}"; do
    for CHI in "${CHI_VALUES[@]}"; do
        
        EXPECTED_FILE="data/out_N${N}_chi${CHI}_tmax2.0.npy"
        if [ -f "$EXPECTED_FILE" ]; then
            echo "Skipping: N=${N} | chi_max=${CHI} (Data exists)"
            continue
        fi
        
        JOB_NAME="oat_N${N}_c${CHI}"
        LOG_FILE="logs/${JOB_NAME}_%j.log"
        
        echo "Submitting: N=${N} | chi_max=${CHI}"
        
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

eval "\$(conda shell.bash hook)"
conda activate metrology
cd ~/metrology_thesis

python scripts/main_fss.py --N ${N} --chi_max ${CHI} --t_max 2.0 --dt 0.01
EOT
    done
done

echo "--------------------------------------------------------"
echo "Unified OAT sweep deployed successfully!"