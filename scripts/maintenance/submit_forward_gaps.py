#!/usr/bin/env python3
"""Fill the 8 missing forward-sweep files (a).

The 8 were pre-Step-0.1 memory-wall casualties (see code_documentation
Sec.15). With current exponential MPO compression they should complete
in 1-2 days each. Each job skips if output already exists.
"""
import os
import subprocess
import sys

REPO = os.path.expanduser('~/metrology_thesis')
JOB_DIR = os.path.join(REPO, 'scripts/maintenance/_forward_gap_jobs')
LOG_DIR = os.path.join(REPO, 'logs/forward_gaps')

# (N, alpha, chi, t_max, mem_gb)
GAPS = [
    (128, 1.0, 128, 2.0,  96),
    (128, 1.0, 256, 2.0, 128),
    (200, 0.0,  64, 2.0, 128),
    (200, 0.0, 256, 2.0, 192),
    (200, 1.0,  64, 1.5, 192),
    (200, 1.0, 128, 1.5, 192),
    (200, 1.0, 256, 1.5, 224),
    (200, 2.0, 256, 2.0, 192),
]
DT = 0.05
WALLTIME = '48:00:00'

def write_script(idx, N, alpha, chi, t_max, mem_gb):
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    tag = f'fwd_gap_{idx:02d}_N{N}_a{alpha}_chi{chi}'
    path = os.path.join(JOB_DIR, f'job_{tag}.sh')
    body = f"""#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem_gb}G
#SBATCH --time={WALLTIME}
#SBATCH --output={LOG_DIR}/{tag}_%j.log
#SBATCH --error={LOG_DIR}/{tag}_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metrology
cd {REPO}
OUT=data/powerlaw_N{N}_chi{chi}_alpha{alpha}.npy
if [ -f "$OUT" ]; then
    echo "[skip] $OUT exists"
    exit 0
fi
python scripts/main_powerlaw.py \\
    --N {N} \\
    --chi_max {chi} \\
    --alpha {alpha} \\
    --t_max {t_max} \\
    --dt {DT}
"""
    with open(path, 'w') as f:
        f.write(body)
    os.chmod(path, 0o755)
    return path

def main():
    dry = '--dry-run' in sys.argv
    ok, fail = [], []
    for idx, (N, alpha, chi, t_max, mem_gb) in enumerate(GAPS):
        script = write_script(idx, N, alpha, chi, t_max, mem_gb)
        if dry:
            print(f'[dry-run] {script}')
            continue
        r = subprocess.run(['sbatch', '--parsable', script],
                           capture_output=True, text=True)
        if r.returncode == 0:
            jid = r.stdout.strip()
            print(f'[submit] N={N} a={alpha} chi={chi} '
                  f't_max={t_max} mem={mem_gb}G -> {jid}')
            ok.append(jid)
        else:
            print(f'[fail]   N={N} a={alpha} chi={chi}: '
                  f'{r.stderr.strip().splitlines()[0]}')
            fail.append((N, alpha, chi))
    if not dry:
        print(f'\n[done] {len(ok)}/{len(GAPS)} submitted, '
              f'{len(fail)} failed')
        if fail:
            print('[retry] run again later when SLURM has capacity:')
            for t in fail:
                print(f'  N={t[0]} alpha={t[1]} chi={t[2]}')

if __name__ == '__main__':
    main()
