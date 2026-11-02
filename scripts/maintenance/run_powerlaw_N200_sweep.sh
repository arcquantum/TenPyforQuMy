#!/bin/bash

# Additional Power-Law sweep for N=200
ALPHA_VALUES=(0.0 1.0 2.0 3.0 6.0)
N=200
CHI_VALUES=(16 64 128 256)

mkdir -p logs
mkdir -p data

echo "Starting Power-Law N=200 Sweep on Regulus (14 Days)..."
echo "--------------------------------------------------------"

for ALPHA in "${ALPHA_VALUES[@]}"; do
    for CHI in "${CHI_VALUES[@]}"; do

        EXPECTED_FILE="data/powerlaw_N${N}_chi${CHI}_alpha${ALPHA}.npy"

        if [ -f "$EXPECTED_FILE" ]; then
            echo "Skipping: N=${N} | chi=${CHI} | alpha=${ALPHA} (Data exists)"
            continue
        fi

        JOB_NAME="plaw_a${ALPHA}_N${N}_c${CHI}"
        LOG_FILE="logs/${JOB_NAME}_%j.log"

        echo "Submitting: N=${N} | chi=${CHI} | alpha=${ALPHA}"

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

python scripts/main_powerlaw.py \
    --N ${N} \
    --chi_max ${CHI} \
    --alpha ${ALPHA} \
    --t_max 2.0 \
    --dt 0.01 \
    --J 1.0
EOT

    done
done

echo "--------------------------------------------------------"
echo "Power-Law N=200 sweep deployed successfully!"