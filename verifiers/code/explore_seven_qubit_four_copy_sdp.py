#!/usr/bin/env python3
"""FS-0--FS-2: numerical four-copy spin-flip relaxation of the minimal mean balanced purity.

Statement: sec:fourcopy (in four_copy_bound.tex)
Numerical evidence only (floating-point construction, Clarabel).  The construction is in
four_copy_relaxation.py; every block is congruence-centred at the Haar-average moment before solving
(without centring the n=7 solves are inaccurate and unstable).  FS-0 checks the per-site algebra and the flip maps against a direct
two-qubit computation; FS-1 gives the relaxation values for n = 4, 5, 6 (true minima 1/3, 1/4, 1/8);
FS-2 gives n = 7 with the omega blocks alone, with each flip family, and with both.
Optional checkpoint directory: environment variable FOURCOPY_CACHE (bases and flip maps are pickled).
"""
import itertools, math, os, sys, time
import numpy as np, cvxpy as cp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import four_copy_relaxation as fc

def fs0():
    fc.check()
    rng = np.random.default_rng(1); n = 2
    z = rng.normal(size=4)+1j*rng.normal(size=4); z /= np.linalg.norm(z)
    zt = np.kron(fc.EPS, fc.EPS) @ z.conj()
    def opword(w):
        M = np.eye(1)
        for k in w: M = np.kron(M, fc.OP[k])
        perm_in = [i*4+c for c in range(4) for i in range(n)]
        return M.reshape((2,)*(8*n)).transpose(perm_in+[4*n+p for p in perm_in]).reshape(16**n, 16**n)
    def fco(vs):
        v = np.ones(1)
        for x in vs: v = np.kron(v, x)
        mu = {w: v.conj() @ opword(w) @ v for w in itertools.product(range(24), repeat=n)}
        X = np.zeros((fc.NC,)*n, dtype=complex)
        for r in itertools.product((0, 1, 2), repeat=n):
            d = [fc.DIM[t] for t in r]; dr = int(np.prod(d)); Dr = np.prod([fc.UDIM[t] for t in r])
            S = sum(val*np.kron(fc.RHO[r[0]][w[0]], fc.RHO[r[1]][w[1]]) for w, val in mu.items())
            F = (dr/(24**n*Dr))*S.T
            X[tuple(fc.SL[t] for t in r)] = F.reshape(d+d).transpose(0, 2, 1, 3).reshape([x*x for x in d])
        return X
    X0, X1, X2 = fco([z, z, z, z]), fco([z, z, z, zt]), fco([z, z, zt, zt])
    def app(T, X):
        for k in range(n): X = np.moveaxis(np.tensordot(T, X, axes=([1], [k])), 0, k)
        return X
    return max(np.abs(app(fc.T1, X0)-X1).max(), np.abs(app(fc.T2, X0)-X2).max())

def solve(params, on, oo, gmaps, use, xh):
    v, x, st, _ = fc.solve_centred(params, on, oo, gmaps, use, xh)
    return v, st

def build(n, cache=None, log=print):
    cf = (lambda s: os.path.join(cache, '%s_n%d.pkl' % (s, n))) if cache else (lambda s: None)
    B = fc.bases(n, cf('bases'))
    params = fc.omega_params(n, B)
    on, oo = fc.obj_norm(n, n//2, params)
    g = {'g1': fc.gamma_maps(n, B, params, 'g1', fc.T1, log, cf('g1')),
         'g2': fc.gamma_maps(n, B, params, 'g2', fc.T2, log, cf('g2'))}
    return params, on, oo, g, fc.haar_point(n, params, on)

if __name__ == '__main__':
    cache = os.environ.get('FOURCOPY_CACHE')
    print('FS-0 per-site algebra checks pass; flip maps T1, T2 agree with a direct n=2 computation to %.1e' % fs0(), flush=True)
    for n, ref in ((4, '1/3'), (5, '1/4'), (6, '1/8')):
        params, on, oo, g, xh = build(n, cache, log=lambda *a: None)
        v, st = solve(params, on, oo, g, ('g1', 'g2'), xh)
        print('FS-1 n=%d: %d parameters; four-copy bound on mean %d-body purity = %.9f [%s] (true minimum %s)'
              % (n, len(params), n//2, v, st, ref), flush=True)
    t0 = time.time()
    params, on, oo, g, xh = build(7, cache, log=lambda s: print(s, flush=True))
    print('FS-2 n=7: %d parameters, maps built in %.0fs' % (len(params), time.time()-t0), flush=True)
    for use in ((), ('g1',), ('g2',), ('g1', 'g2')):
        v, st = solve(params, on, oo, g, use, xh)
        print('FS-2 n=7 omega + %-9s: bound on mean 3-body purity = %.9f [%s]   (three copies 125/978 = %.9f; Psi_ico 121/917 = %.9f)'
              % ('+'.join(use) or 'nothing', v, st, 125/978, 121/917), flush=True)
