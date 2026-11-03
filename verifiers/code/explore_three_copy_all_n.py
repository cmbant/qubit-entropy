#!/usr/bin/env python3
"""TA-1: floating-point values of the Schur-Weyl three-copy relaxation for n = 4..N (numerical evidence).

Statement: sec:threecopy-sw (in three_copy_schur_weyl.tex)  The gamma map is
exact (three_copy_schur_weyl.exact_gamma_map, cached in THREECOPY_CACHE if set); only the SDP solve
(multiprecision SDPA, blocks congruence-centred exactly, see search_three_copy_sw_certificate.solve) is in floating point.  Each value is printed
next to the exact Rains LP bound of the summary table (thm:lptable and sec:generalizations), with which
it must agree wherever a state attains the LP.
Usage: explore_three_copy_all_n.py [N]   (default 20)
"""
import os, pickle, sys, time
from fractions import Fraction as Fr
import flint
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import three_copy_schur_weyl as sw
from three_copy_schur_weyl import Q
import search_three_copy_sw_certificate as search
search.SOLVER = 'SDPA'     # multiprecision SDPA-GMP: double precision loses accuracy from n = 15

LP = {4: '1/3', 5: '1/4', 6: '1/8', 7: '1/8', 8: '3/35', 9: '1/14', 10: '1/21', 11: '1/28', 12: '5/224', 13: '25/1408',
      14: '331/27456', 15: '1/104', 16: '1/156', 17: '1/208', 18: '7/2288', 19: '93/38896', 20: '1193/739024'}
if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20; cache = os.environ.get('THREECOPY_CACHE')
    for n in range(4, N+1):
        t0 = time.time(); st = sw.Structure(n); on, oo = st.functionals(n//2)
        cf = os.path.join(cache, 'gamma_n%d.pkl' % n) if cache else None
        if cf and os.path.exists(cf):
            G = {bi: [flint.fmpq_mat([[Q(*e) for e in r] for r in M]) for M in arr] for bi, arr in pickle.load(open(cf, 'rb')).items()}
        else:
            G = sw.exact_gamma_map(st)
            if cf: pickle.dump({bi: [[[(int(M[i, j].p), int(M[i, j].q)) for j in range(M.ncols())] for i in range(M.nrows())] for M in arr] for bi, arr in G.items()}, open(cf, 'wb'))
        val, status = search.solve(st, G, on, oo, rounds=2)[:2]
        lp = Fr(LP[n]) if n in LP else None
        print('TA-1 n=%2d k=%2d: %3d parameters, %3d gamma-blocks; three-copy value %.13f (%s); Rains LP %s = %.13f; difference %+.2e  (%.0fs)'
              % (n, n//2, st.NP, len(st.blocks['ga']), val, status, LP.get(n, '-'), float(lp) if lp else float('nan'),
                 val-float(lp) if lp else float('nan'), time.time()-t0), flush=True)
        if n == 7: print('TA-1 n= 7: three-copy value - 125/978 = %+.2e' % (val-125/978), flush=True)
