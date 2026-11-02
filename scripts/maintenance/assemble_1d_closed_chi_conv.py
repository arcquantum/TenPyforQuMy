#!/usr/bin/env python
"""
Assemble the per-(alpha, N) chi_conv CSVs from the 15 SLURM array
tasks into a single 1d_closed.csv, and append the independence
CSVs. This is the deliverable the the documentation review asks for.
"""
import csv
import glob
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REPORT_DIR = os.path.join(ROOT_DIR, 'data', 'validation_reports')

def main():
    files = sorted(glob.glob(os.path.join(REPORT_DIR, 'val_*_1d_closed_chi_conv.csv')))
    if not files:
        print(f'no val_*_1d_closed_chi_conv.csv files found in {REPORT_DIR}')
        return 1

    rows, header = [], None
    for f in files:
        with open(f, newline='') as fh:
            r = csv.DictReader(fh)
            if r.fieldnames is None:
                print(f'[skip] {f}: empty file or missing header')
                continue
            if header is None:
                header = list(r.fieldnames)
            for row in r:
                rows.append(row)
                print(f'{f.split("/")[-1]}: alpha={row["alpha"]}, '
                      f'N={row["N"]}, chi_conv={row["chi_conv"]}')

    if header is None:
        print('ERROR: no valid header found in any input file')
        return 1

    out = os.path.join(REPORT_DIR, '1d_closed.csv')
    assert header is not None, "header must be set before writing"
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nWrote {out} ({len(rows)} rows)')

    # Print the summary in the format the review asks for.
    print('\n=== chi_conv table (the documentation deliverable) ===')
    print(f'{"alpha":>5} {"N":>4} {"chi_conv":>10} {"eps_max":>12} '
          f'{"fid_max":>12} {"M_max":>12} {"Q_max":>12} {"status":>20}')
    for r in rows:
        print(f'{r["alpha"]:>5} {r["N"]:>4} {r["chi_conv"]:>10} '
              f'{r["eps_max"]:>12} {r["fid_max"]:>12} '
              f'{r["M_max"]:>12} {r["Q_max"]:>12} {r["status"]:>20}')

    # Verify the four review criteria.
    print('\n=== Review-criteria check ===')
    all_eps_ok = all(float(r['eps_max']) < 1e-3 for r in rows
                     if r['eps_max'] not in ('nan', ''))
    all_fid_ok = all(float(r['fid_max']) < 1e-6 for r in rows
                     if r['fid_max'] not in ('nan', ''))
    all_M_ok = all(float(r['M_max']) < 1e-3 for r in rows
                   if r['M_max'] not in ('nan', ''))
    all_Q_ok = all(float(r['Q_max']) < 1e-3 for r in rows
                   if r['Q_max'] not in ('nan', ''))
    all_conv = all(r['status'] == 'converged' for r in rows)
    print(f'  eps_max  < 1e-3 : {"PASS" if all_eps_ok else "FAIL"}')
    print(f'  fid_max  < 1e-6 : {"PASS" if all_fid_ok else "FAIL"}')
    print(f'  M_max    < 1e-3 : {"PASS" if all_M_ok else "FAIL"}')
    print(f'  Q_max    < 1e-3 : {"PASS" if all_Q_ok else "FAIL"}')
    print(f'  all converged   : {"PASS" if all_conv else "FAIL"}')

    print('\n=== Independence checks ===')
    for name in ['dt', 'svdmin', 'algebra']:
        path = os.path.join(REPORT_DIR, f'indep_1d_closed_{name}.csv')
        if os.path.exists(path):
            print(f'\n{name}:')
            with open(path) as fh:
                for line in fh:
                    print(f'  {line.rstrip()}')

    return 0 if (all_eps_ok and all_fid_ok and all_conv) else 1


if __name__ == '__main__':
    raise SystemExit(main())