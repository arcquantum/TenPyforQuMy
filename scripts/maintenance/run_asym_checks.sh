#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd ~/metrology_thesis
python scripts/checks/check_asymmetric_echo.py --N 8 --open
python scripts/checks/check_asymmetric_echo.py --N 8 --open --gamma1 1e-3 --gamma2 1e-3
python scripts/checks/check_asymmetric_echo.py --Lx 2 --Ly 4 --open
python scripts/checks/check_asymmetric_echo.py --Lx 2 --Ly 4 --open --gamma1 1e-3 --gamma2 1e-3
