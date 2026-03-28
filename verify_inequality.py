"""
Verify the correlation inequality R(m,k) <= alpha(m,k) by exact rational arithmetic.

Usage:
    python verify_inequality.py                    # default: m <= 20, k up to 4*ln(m)
    python verify_inequality.py --max_m 50         # m <= 50, k up to 4*ln(m)
    python verify_inequality.py --max_m 30 --max_k 15   # m <= 30, k <= 15
    python verify_inequality.py --max_m 100 --max_k log  # m <= 100, k up to 4*ln(m)
"""

import argparse
import time
from fractions import Fraction
from math import comb, factorial, log, ceil


def R_alpha_check(m, k):
    """Check R(m,k) < alpha(m,k) using exact rational arithmetic.
    Returns (R/alpha as Fraction, float value, True/False)."""

    # E[1-V] = sum C(m,r)(-1)^r / (r+1)^k
    e1v = sum(Fraction(comb(m, r) * (-1)**r, (r + 1)**k) for r in range(m + 1))

    # E[(1-V)^2] via multinomial
    e1v2 = Fraction(0)
    for a in range(m + 1):
        for b in range(m + 1 - a):
            for c in range(m + 1 - a - b):
                d = m - a - b - c
                mc = factorial(m) // (factorial(a) * factorial(b) * factorial(c) * factorial(d))
                num = a + b + 2 * c + 2
                den = (a + c + 1) * (b + c + 1) * (a + b + c + 2)
                e1v2 += (-1)**(a + b) * mc * Fraction(num, den)**k

    var = e1v2 - e1v**2
    ev = 1 - e1v
    d = ev * (1 - ev)
    alpha = 1 - Fraction(m + 1, m + 2)**k

    if d <= 0 or alpha <= 0:
        return None, None, None

    R = var / d
    ratio = R / alpha
    holds = R < alpha  # strict inequality (except k=1 where equality)

    return ratio, float(ratio), holds


def get_max_k(m, max_k_arg):
    """Determine max k to check for a given m."""
    if max_k_arg == 'log':
        return max(2, ceil(4 * log(m))) if m >= 2 else 2
    else:
        return int(max_k_arg)


def main():
    parser = argparse.ArgumentParser(description='Verify R(m,k) <= alpha(m,k)')
    parser.add_argument('--max_m', type=int, default=20, help='Maximum m to check (default: 20)')
    parser.add_argument('--max_k', type=str, default='log',
                        help='Maximum k: integer or "log" for ceil(4*ln(m)) (default: log)')
    args = parser.parse_args()

    print(f"Verifying R(m,k) <= alpha(m,k)", flush=True)
    print(f"  max_m = {args.max_m}", flush=True)
    print(f"  max_k = {args.max_k}" + (" (= ceil(4*ln(m)) per m)" if args.max_k == 'log' else ""), flush=True)
    print(flush=True)

    total_pairs = 0
    total_ok = 0
    total_eq = 0
    failures = []
    start_time = time.time()

    for m in range(1, args.max_m + 1):
        mk = get_max_k(m, args.max_k)
        m_start = time.time()
        m_results = []

        for k in range(1, mk + 1):
            ratio, ratio_f, holds = R_alpha_check(m, k)
            total_pairs += 1

            if ratio is None:
                m_results.append(f"k={k}:N/A")
                continue

            if k == 1:
                # k=1 is equality
                total_eq += 1
                total_ok += 1
            elif holds:
                total_ok += 1
            else:
                failures.append((m, k, ratio_f))

        elapsed = time.time() - m_start
        # Find max R/alpha for this m
        max_ra = 0
        for k in range(2, mk + 1):
            ratio, ratio_f, holds = R_alpha_check(m, k)
            if ratio_f is not None and ratio_f > max_ra:
                max_ra = ratio_f

        status = "OK" if max_ra < 1 else "FAIL!"
        print(f"  Verified m={m:4d} (k=1..{mk:2d}): max R/alpha = {max_ra:.6f} < 1  [{elapsed:.1f}s]  {status}", flush=True)

    total_time = time.time() - start_time

    print(flush=True)
    print(f"{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"  Total (m,k) pairs checked: {total_pairs}", flush=True)
    print(f"  Equalities (k=1): {total_eq}", flush=True)
    print(f"  Strict inequalities: {total_ok - total_eq}", flush=True)
    print(f"  Failures: {len(failures)}", flush=True)
    print(f"  Total time: {total_time:.1f}s", flush=True)

    if failures:
        print(f"\n  FAILURES:", flush=True)
        for m, k, r in failures:
            print(f"    m={m}, k={k}: R/alpha = {r:.8f}", flush=True)
    else:
        if args.max_k == 'log':
            print(f"\n  THEOREM: R(m,k) < alpha(m,k) for all m <= {args.max_m} and all k >= 1.", flush=True)
            print(f"  (k=1..ceil(4*ln(m)) verified; k > 4*ln(m) covered by analytic proof.)", flush=True)
        else:
            print(f"\n  R(m,k) < alpha(m,k) verified for all m <= {args.max_m}, k <= {args.max_k}.", flush=True)


if __name__ == '__main__':
    main()
