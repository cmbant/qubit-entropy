#!/usr/bin/env python3
"""Search for an exact dual certificate of the four-copy relaxation (writes data/four_copy_certificate_n<n>.json).

Statement: sec:fourcopy (in four_copy_bound.tex)  Not a verifier: the
certificate it writes is checked by certify_seven_qubit_four_copy_bound.py.

Recipe.  Solve the floating-point relaxation (four_copy_relaxation.py), every block congruence-centred at
the Haar-average moment, with every centred flip block's trace rewarded by eps:
min <oo, x> - eps sum tr(gamma blocks) s.t. <on, x> = 1, omega(x) >= 0, gamma(x) >= 0.
The duals of the gamma constraints plus eps*I, moved back to uncentred coordinates, are strictly positive
definite multipliers.  They are moved
to the integer bases of four_copy_exact.py (Z_int = B Z B^T with Tr Q_int B = Q_orth), scaled by Qz and
rounded; then the exact residual d_p = oo_p - lam on_p - c_p/(12^n Qz) is computed and each candidate lam
is tested (omega blocks of d positive definite).  The largest passing lam is stored.
Usage: search_four_copy_certificate.py n eps lam1,lam2,...   (FOURCOPY_CACHE: optional pickle directory)
The committed n = 7 data come from `search_four_copy_certificate.py 7 1e-9 2637/20000,6593/50000,13187/100000`
(all passed); the stored residual (FOURCOPY_CACHE/cert_c_n7.pkl) was then re-tested at 211/1600 = 0.131875,
the perturbed dual value, which also passes, and the data file's lambda was set to it.  The certify script
re-checks everything from the data file.
"""
import json, os, pickle, sys, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import four_copy_relaxation as fo
import four_copy_exact as fx

def site_tr(t):
    """Orthonormal irrep basis (columns of V4, V3) in terms of the integer one."""
    return {1: fo.V4.T @ fx.V31, 2: fo.V3.T @ fx.V22}.get(t, np.ones((1, 1)))

def to_int(Bo, Bx, Zs):
    out = []
    for name, (tp, li), Zo in Zs:
        lo, mo, _, Qo = Bo[(tp, name)][li]; lx, mx, _, Qx = Bx[(tp, name)][li]; assert (lo, mo) == (lx, mx)
        Tr = np.ones((1, 1))
        for t in fx.tlist(tp): Tr = np.kron(Tr, site_tr(t))
        A = Tr @ Qx.astype(float); Bm = np.linalg.lstsq(A, Qo, rcond=None)[0]
        assert np.abs(A @ Bm - Qo).max() < 1e-8
        out.append((name, (tp, li), Bm @ Zo @ Bm.T))
    return out

if __name__ == '__main__':
    n, eps = int(sys.argv[1]), float(sys.argv[2]); lams = [Fr(s) for s in sys.argv[3].split(',')]
    Qz = 10**14; cache = os.environ.get('FOURCOPY_CACHE'); t0 = time.time()
    cf = (lambda s: os.path.join(cache, '%s_n%d.pkl' % (s, n))) if cache else (lambda s: None)
    Bo = fo.bases(n, cf('bases')); po = fo.omega_params(n, Bo); on_f, oo_f = fo.obj_norm(n, n//2, po)
    gm = {nm: fo.gamma_maps(n, Bo, po, nm, T, lambda *a: None, cf(nm)) for nm, T in (('g1', fo.T1), ('g2', fo.T2))}
    xh = fo.haar_point(n, po, on_f)
    val, _, st, Zs = fo.solve_centred(po, on_f, oo_f, gm, ('g1', 'g2'), xh, eps=eps); print('perturbed value %.10f %s (%.0fs)' % (val, st, time.time()-t0), flush=True)
    if cache and os.path.exists(cf('xbases')): Bx = pickle.load(open(cf('xbases'), 'rb'))
    else:
        Bx = fx.bases(n)
        if cache: pickle.dump(Bx, open(cf('xbases'), 'wb'))
    blocks, params = fx.omega_data(n, Bx); on, oo = fx.obj_norm(n, n//2, blocks, params)
    Zi = to_int(Bo, Bx, Zs); keys = [(nm, key) for nm, key, _ in Zi]
    Zint = [[[int(round(Qz*(Z[i, j]+Z[j, i])/2)) for j in range(Z.shape[0])] for i in range(Z.shape[0])] for _, _, Z in Zi]
    print('rational data and rounded multipliers (%.0fs)' % (time.time()-t0), flush=True)
    c = fx.c_exact(n, Bx, blocks, params, Zint, keys, log=lambda s: print(s, flush=True))
    if cache: pickle.dump(dict(c=c, Zint=Zint, keys=keys), open(cf('cert_c'), 'wb'))
    best = None
    for lam in lams:
        ok, mn = fx.exact_check(blocks, params, on, oo, lam, c, 12**n*Qz)
        print('lambda %s = %.8f: omega residual positive definite %s (min eig %.3e)' % (lam, float(lam), ok, mn), flush=True)
        if ok and (best is None or lam > best): best = lam
    out = os.path.join(HERE, '..', 'data', 'four_copy_certificate_n%d.json' % n)
    if best is not None and os.path.exists(out) and Fr(json.load(open(out))['lam']) >= best:
        print('kept the existing certificate (lambda = %s >= %s)' % (json.load(open(out))['lam'], best)); best = None
    if best is not None:
        json.dump(dict(n=n, lam=str(best), Qz=Qz, eps=eps,
                       blocks=[dict(flip=nm, type=list(tp), index=li, Z=[[str(v) for v in r] for r in Z]) for (nm, (tp, li)), Z in zip(keys, Zint)]),
                  open(out, 'w'))
        print('wrote %s with lambda = %s (%.0fs)' % (os.path.relpath(out, HERE), best, time.time()-t0), flush=True)
