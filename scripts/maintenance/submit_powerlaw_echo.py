import os
import subprocess

PYTHON = "/home/c705144/miniconda3/envs/metrology/bin/python"

N_values = [32, 64, 128, 200]
chi_values = [16, 64, 128, 256]
alpha_values = [0.0, 1.0, 2.0, 3.0, 6.0]

protocols = {
    "OUT":       (1.57079632679, -1.57079632679),
    "ExactEcho": (1.57079632679,  0.0),
    "Squeezing": (1.57079632679,  1.57079632679),
}

os.makedirs("logs", exist_ok=True)

for alpha in alpha_values:
    for N in N_values:
        mem = "128G" if N == 200 else "64G"
        for chi in chi_values:
            for protocol, (mu, nu) in protocols.items():
                job_name = f"ple_a{alpha}_N{N}_c{chi}_{protocol}"
                log_file = f"logs/{job_name}_%j.log"
                out_file = (
                    f"data/powerlaw_echo_pts/"
                    f"plaw_echo_N{N}_chi{chi}_alpha{alpha}_mu{mu:.4f}_nu{nu:.4f}.npy"
                )

                if os.path.exists(out_file):
                    print(f"Skipping {job_name}, already exists")
                    continue

                script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time=14-00:00:00

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ~/metrology_thesis
{PYTHON} scripts/workers/closed_1d/powerlaw_echo_worker.py \
    --N {N} \
    --chi_max {chi} \
    --alpha {alpha} \
    --mu {mu} \
    --nu {nu} \
    --J 1.0 \
    --dt_max 0.05
"""
                script_file = f"submit_{job_name}.sh"
                with open(script_file, "w") as f:
                    f.write(script)

                subprocess.run(["sbatch", script_file], cwd="/home/c705144/metrology_thesis")
                print(f"Submitted {job_name}")