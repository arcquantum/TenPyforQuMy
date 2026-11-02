#!/usr/bin/env python
"""
Sanity check for compute_qfi_mixed_bounds.

Three physical tests:
  1. Single pure state |+z>: qv = qa = 1.
  2. Two identical pure trajectories: qv = qa = 1.
  3. Maximally mixed single qubit via |0>, |1>: qv = 1 (diagnostic),
     qa = 0 (correct upper bound, true QFI = 0).
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.observables import compute_qfi_mixed_bounds


def main():
    cov = np.diag([0.25, 0.25, 0.0])
    S_vec = np.array([0.0, 0.0, 0.5])
    qv, qa = compute_qfi_mixed_bounds([cov], [S_vec])
    assert abs(qv - 1.0) < 1e-12, f"test1 qv={qv}"
    assert abs(qa - 1.0) < 1e-12, f"test1 qa={qa}"
    print(f"[test1] single pure state: qv={qv:.4f} qa={qa:.4f}  PASS")

    qv, qa = compute_qfi_mixed_bounds([cov, cov], [S_vec, S_vec])
    assert abs(qv - 1.0) < 1e-12
    assert abs(qa - 1.0) < 1e-12
    print(f"[test2] two identical trajectories: qv={qv:.4f} qa={qa:.4f}  PASS")

    cov0 = np.zeros((3, 3))
    S0 = np.array([0.0, 0.0, +0.5])
    S1 = np.array([0.0, 0.0, -0.5])
    qv, qa = compute_qfi_mixed_bounds([cov0, cov0], [S0, S1])
    assert abs(qv - 1.0) < 1e-12, f"test3 qv={qv}"
    assert abs(qa - 0.0) < 1e-12, f"test3 qa={qa}"
    print(f"[test3] maximally mixed ensemble: qv={qv:.4f} (diag), "
          f"qa={qa:.4f} (bound, true QFI=0)  PASS")

    print("\n[PASS] compute_qfi_mixed_bounds is correct.")


if __name__ == '__main__':
    main()
