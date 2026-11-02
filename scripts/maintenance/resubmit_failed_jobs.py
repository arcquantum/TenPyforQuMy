import subprocess

PYTHON = "/home/c705144/miniconda3/envs/metrology/bin/python"

jobs = [
    {
        "name": "plaw_a0_N200_c64_mem128",
        "script": "scripts/main_powerlaw.py",
        "args": "--N 200 --chi_max 64 --alpha 0.0 --t_max 2.0 --dt 0.01 --J 1.0",
        "mem": "128G",
    },
    {
        "name": "ple_a0_N128_c128_OUT_mem128",
        "script": "scripts/workers/closed_1d/powerlaw_echo_worker.py",
        "args": "--N 128 --chi_max 128 --alpha 0.0 --mu 1.57079632679 --nu -1.57079632679 --J 1.0 --dt_max 0.05",
        "mem": "128G",
    },
    {
        "name": "ple_a0_N128_c256_OUT_mem128",
        "script": "scripts/workers/closed_1d/powerlaw_echo_worker.py",
        "args": "--N 128 --chi_max 256 --alpha 0.0 --mu 1.57079632679 --nu -1.57079632679 --J 1.0 --dt_max 0.05",
        "mem": "128G",
    },
    {
        "name": "ple_a0_N128_c128_ExactEcho_mem128",
        "script": "scripts/workers/closed_1d/powerlaw_echo_worker.py",
        "args": "--N 128 --chi_max 128 --alpha 0.0 --mu 1.57079632679 --nu 0.0 --J 1.0 --dt_max 0.05",
        "mem": "128G",
    },
    {
        "name": "ple_a0_N128_c256_ExactEcho_mem128",
        "script": "scripts/workers/closed_1d/powerlaw_echo_worker.py",
        "args": "--N 128 --chi_max 256 --alpha 0.0 --mu 1.57079632679 --nu 0.0 --J 1.0 --dt_max 0.05",
        "mem": "128G",
    },
]

for job in jobs:
    name = job["name"]
    log_file = f"logs/{name}_%j.log"
    cmd = [
        "sbatch",
        f"--job-name={name}",
        f"--output={log_file}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        f"--mem={job['mem']}",
        "--time=14-00:00:00",
        f"--wrap={PYTHON} {job['script']} {job['args']}"
    ]
    subprocess.run(cmd, cwd="/home/c705144/metrology_thesis")
    print("submitted", name)