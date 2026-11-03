#!/usr/bin/env python3
"""TCN-1--TCN-3: numerical three-copy spin-flip relaxation for n = 4..7, and a
rounded strictly feasible dual at the interior value 639/5000 < L_3.  The sharp
bound L_3 = 125/978 is certified independently, with no data file, by
certify_seven_qubit_three_copy_bound.py.

Statement: sec:threecopy (in replica_hierarchy.tex)
Numerical evidence only (Clarabel); the certificate itself is checked exactly
by the certify_ script.  Writes ../data/three_copy_certificate_n7.json (not used
by any proof).
"""
import json, math, os, sys
from fractions import Fraction as Fr
import numpy as np, cvxpy as cp, sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import three_copy_relaxation as tc

def relaxation(d, k, lam_fixed=None, tight=False):
    NO, rels = d['NO'], d['rels']; o_obj = tc.purity_orbit(d, k); o_n = d['o_norm']
    Zs, sc, lhs = [], [], [0]*NO
    for b in d['blocks']:
        Kf = b['K'].astype(float); s = float(np.abs(Kf).max()); sc.append(s); Kf = Kf/s
        Z = cp.Variable((b['D'], b['D']), symmetric=True); Zs.append(Z)
        for o in range(NO): lhs[o] = lhs[o] + cp.sum(cp.multiply(Z, Kf[o]))
    nu = cp.Variable(len(rels)); t = cp.Variable()
    lam = cp.Variable() if lam_fixed is None else float(lam_fixed)
    cons = [lhs[o] + rels[:, o] @ nu == (1.0 if o == o_obj else 0.0) - (lam if o == o_n else 0.0) for o in range(NO)]
    if lam_fixed is None:
        pr = cp.Problem(cp.Maximize(lam), cons + [Z >> 0 for Z in Zs])
    else:
        pr = cp.Problem(cp.Maximize(t), cons + [Z >> t*np.eye(Z.shape[0]) for Z in Zs] + [t <= 1])
    if tight:
        pr.solve(solver='CLARABEL', tol_gap_abs=1e-14, tol_gap_rel=1e-14, tol_feas=1e-14, max_iter=500)
    else:
        pr.solve(solver='CLARABEL')
    return pr, Zs, nu, sc, t

if __name__ == '__main__':
    for n, k, ref in ((4, 2, Fr(1, 3)), (5, 2, Fr(1, 4)), (6, 3, Fr(1, 8)), (7, 3, Fr(125, 978))):
        d = tc.build(n)
        pr = relaxation(d, k)[0]
        print('TCN-1 n=%d k=%d: three-copy bound on mean %d-body purity = %.10f  (reference %s = %.10f)'
              % (n, k, k, pr.value, ref, float(ref)), flush=True)
    pr = relaxation(d, 3, tight=True)[0]
    print('TCN-1b n=7 tight tolerances: %.13f, minus 125/978 = %.2e' % (pr.value, pr.value - 125/978), flush=True)
    # certificate data for n=7 at lambda = 639/5000
    lam = Fr(639, 5000)
    pr, Zs, nu, sc, t = relaxation(d, 3, lam)
    print('TCN-2 n=7: strictly feasible dual at lambda = %s, min eigenvalue margin %.3e' % (lam, t.value), flush=True)
    Q0 = 10**40 * 5000
    Zt = [[[int(round(Q0*((Z.value[i, j]+Z.value[j, i])/2)/s)) for j in range(Z.shape[0])] for i in range(Z.shape[0])]
          for Z, s in zip(Zs, sc)]
    nut = [int(round(Q0*v)) for v in nu.value]
    NO, rels = d['NO'], d['rels']; o_obj = tc.purity_orbit(d, 3); o_n = d['o_norm']
    def residual(Zt, nut, Q):
        r = []
        for o in range(NO):
            v = sum(int(Zt[b][i][j])*int(B['K'][o][i, j]) for b, B in enumerate(d['blocks'])
                    for i in range(B['D']) for j in range(B['D']))
            v += sum(int(rels[a, o])*nut[a] for a in range(len(nut)))
            target = Fr(Q)*((1 if o == o_obj else 0) - (lam if o == o_n else 0))
            r.append(target - v)
        return r
    r = residual(Zt, nut, Q0)
    print('TCN-3 integer rounding residual max %.3e (relative to Q=%.1e)' % (max(abs(float(x)) for x in r), Q0))
    # columns: symmetric Z entries and nu
    cols = []
    for b, B in enumerate(d['blocks']):
        for i in range(B['D']):
            for j in range(i, B['D']):
                cols.append(('Z', b, i, j, [int(B['K'][o][i, j])*(1 if i == j else 2) for o in range(NO)]))
    for a in range(len(rels)): cols.append(('nu', a, 0, 0, [int(rels[a, o]) for o in range(NO)]))
    A = sp.Matrix([[c[4][o] for c in cols] for o in range(NO)])
    y = (A*A.T).LUsolve(sp.Matrix([sp.Rational(x.numerator, x.denominator) for x in r]))
    den = sp.ilcm(*[sp.fraction(v)[1] for v in y]); yi = [int(v*den) for v in y]
    Q = Q0*int(den)
    Zt = [[[v*int(den) for v in row] for row in Zb] for Zb in Zt]; nut = [v*int(den) for v in nut]
    for c in cols:
        dv = sum(c[4][o]*yi[o] for o in range(NO))
        if c[0] == 'Z':
            _, b, i, j, _ = c; Zt[b][i][j] += dv
            if i != j: Zt[b][j][i] += dv
        else: nut[c[1]] += dv
    assert all(x == 0 for x in residual(Zt, nut, Q))
    out = dict(n=7, k=3, lam=str(lam), Q=str(Q), blocks=[[b['kind'], b['m']] for b in d['blocks']],
               Z=[[[str(v) for v in row] for row in Zb] for Zb in Zt], nu=[str(v) for v in nut])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'three_copy_certificate_n7.json')
    json.dump(out, open(path, 'w'))
    print('TCN-3 exact integer identity data written (Q has %d digits); not checked or used by any proof (L_3 is certified by certify_seven_qubit_three_copy_bound.py)' % len(str(Q)))
