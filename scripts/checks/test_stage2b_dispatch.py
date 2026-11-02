#!/usr/bin/env python3
"""Stage 2b dispatch regression test.

Verifies run_task routes to the correct worker CLI arguments for both
conventions:
  mu_nu   -> --mu VAL --nu VAL
  mu1mu2  -> --mu1 VAL --mu2 VAL
Monkeypatches subprocess.run so no SLURM job actually spawns.
"""
import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts', 'maintenance'))

from submit_production import run_task  # noqa: E402


def _capture(fn):
    captured = []
    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        class _R:
            returncode = 0
        return _R()
    orig = subprocess.run
    subprocess.run = fake_run
    try:
        fn()
    finally:
        subprocess.run = orig
    return captured[0]


def _check(cmd, required, forbidden):
    for flag, value in required:
        if flag not in cmd:
            raise AssertionError(f"missing {flag!r} in cmd: {cmd}")
        idx = cmd.index(flag)
        got = cmd[idx + 1]
        if got != value:
            raise AssertionError(
                f"{flag} expected {value!r}, got {got!r}"
            )
    for flag in forbidden:
        if flag in cmd:
            raise AssertionError(f"unexpected {flag!r} in cmd: {cmd}")


def main():
    print("=" * 60)
    print("Stage 2b dispatch regression test")
    print("=" * 60)

    c1 = _capture(lambda: run_task(1, 0, 64, 40, 0,
                                   convention='mu_nu'))
    _check(c1,
           required=[('--mu', '0.000000'),
                     ('--nu', '-3.141593')],
           forbidden=['--mu1', '--mu2'])
    print("PASS: mu_nu convention")

    c2 = _capture(lambda: run_task(1, 0, 64, 40, 0,
                                   convention='mu1mu2'))
    _check(c2,
           required=[('--mu1', '0.000000'),
                     ('--mu2', '0.000000')],
           forbidden=['--mu', '--nu'])
    print("PASS: mu1mu2 convention")

    print()
    print("All dispatch checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())