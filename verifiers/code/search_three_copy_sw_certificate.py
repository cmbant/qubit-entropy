#!/usr/bin/env python3
"""Search for exact dual certificates of the Schur-Weyl three-copy relaxation (writes
data/three_copy_sw_certificates.json).  Not a verifier: certify_three_copy_schur_weyl_bounds.py re-checks.

Statement: sec:threecopy-sw (in three_copy_schur_weyl.tex)
Recipe: exact gamma map (three_copy_schur_weyl.exact_gamma_map) -> floating-point SDP with every block
congruence-centred by an exact rational LDL at the Haar moment and then at points nearer the optimum (the
blocks are badly conditioned in the monomial basis for n >= 15), with all centred block traces rewarded by eps -> centred multipliers
rationalised (dyadic, stored with the rational centre) and mapped back by the exact congruence
Z_b = L^{-T} Y'_b L^{-1} (three_copy_schur_weyl.gamma_multipliers), so Z stays positive definite -> exact residual
d = oo - lam*on - Gamma^*(Z), whose omega-blocks must be positive definite.  The largest passing lam of
the candidates is stored.
Usage: search_three_copy_sw_certificate.py n[:k] eps lam1,lam2,...|auto   (k-body purity, default k = floor(n/2))
Environment: THREECOPY_CACHE (gamma-map directory), THREECOPY_SOLVER=SDPA, THREECOPY_DATA (output file),
THREECOPY_CHECKPOINT (resume file for the re-centring rounds; <file>.rec keeps the finished solve for the exact checks).
"""
import json, os, sys, time
from fractions import Fraction as Fr
import numpy as np, cvxpy as cp, flint
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import three_copy_schur_weyl as sw
SOLVER = os.environ.get('THREECOPY_SOLVER', 'CLARABEL')
from three_copy_schur_weyl import Q, ffloat

def haar_exact(st, on):
    """Haar moment (each omega block (W Gram W^T)^{-1}), normalised exactly to on.x = 1."""
    x = [Q(0)]*st.NP
    for bi, (a, nu, W, WT, co) in enumerate(st.blocks['om']):
        MH = (W*st.gram(nu)*WT).inv()
        for p, (b, k, l) in enumerate(st.params):
            if b == bi: x[p] = MH[k, l]
    N = sum((on[p]*x[p] for p in range(st.NP)), Q(0))
    return [v/N for v in x]

def omega_unit(st, p):
    b, k, l = st.params[p]; m = st.size('om', b); E = flint.fmpq_mat(m, m); E[k, l] = 1; E[l, k] = 1
    return b, E

def centred_problem(st, G, on, oo, xc):
    """Every block congruence-centred EXACTLY at the rational point xc: block(xc) = L D L^T, and the float
    tensor is D^{-1/2} L^{-1} block(e_p) L^{-T} D^{-1/2}, equal to the identity at x = xc.
    Returns None if some block is not positive definite at xc."""
    blocks = []
    def centre(Ms, M0):
        if not sw.is_pd(M0): return None
        Li, D = sw.ldl(M0); s = np.array([1/np.sqrt(ffloat(d)) for d in D]); LiT = Li.transpose()
        T = np.array([[[ffloat(v) for v in r] for r in (Li*M*LiT).tolist()] if M is not None else np.zeros((len(D), len(D))) for M in Ms])
        return T*s[None, :, None]*s[None, None, :], Li, s
    om = [[None]*st.NP for _ in st.blocks['om']]
    for p in range(st.NP):
        b, E = omega_unit(st, p); om[b][p] = E
    for bi in range(len(st.blocks['om'])):
        M0 = sum((om[bi][p]*xc[p] for p in range(st.NP) if om[bi][p] is not None), flint.fmpq_mat(st.size('om', bi), st.size('om', bi)))
        c = centre(om[bi], M0)
        if c is None: return None
        blocks.append(('om', bi) + c)
    for bi, arr in G.items():
        M0 = sum((arr[p]*xc[p] for p in range(st.NP) if xc[p] != 0), flint.fmpq_mat(arr[0].nrows(), arr[0].nrows()))
        c = centre(arr, M0)
        if c is None: return None
        blocks.append(('ga', bi) + c)
    return blocks

def solve(st, G, on, oo, eps=0.0, tol=1e-9, rounds=3, t=0.9, log=None):
    """min oo.x s.t. on.x = 1 and every block PSD, with all centred block traces rewarded by eps in the last round.  Round 1 is
    centred at the Haar moment; each later round re-centres exactly at (1-t) x_centre + t x_previous.
    Returns (value, status, {gamma block: (L^{-1}, s, Y)}, centre) where Z_b = L^{-T} diag(s) Y diag(s) L^{-1}
    and L D L^T is the gamma block at the (rational) centre."""
    onf = np.array([ffloat(v) for v in on]); oof = np.array([ffloat(v) for v in oo]); NP = st.NP
    xc = haar_exact(st, on); h = float(oof @ np.array([ffloat(v) for v in xc])); best = None; r0 = 0
    ck = os.environ.get('THREECOPY_CHECKPOINT')      # resume file for long runs: (next round, centre)
    if ck and os.path.exists(ck):
        r0, cen = json.load(open(ck)); xc = [Q(Fr(v)) for v in cen]
        if log: log('  resuming at round %d from %s' % (r0+1, ck))
    def one(xc, e):
        blocks = centred_problem(st, G, on, oo, xc)
        if blocks is None: return None
        xcf = np.array([ffloat(v) for v in xc]); dg = {(b, k): xcf[p] for p, (b, k, l) in enumerate(st.params) if k == l}
        sc = np.array([np.sqrt(abs(dg[(b, k)]*dg[(b, l)])) for (b, k, l) in st.params])    # x_p = sc_p y_p
        y = cp.Variable(NP); cons = [(sc*onf) @ y == 1]; gl = []; pen = 0
        for name, bi, T, Li, s in blocks:
            m = T.shape[1]; M = cp.reshape(y @ (T*sc[:, None, None]).reshape(NP, -1), (m, m), order='C'); Ms = (M+M.T)/2
            c = Ms >> 0; cons.append(c)
            pen = pen+cp.trace(Ms)        # every block: the omega residual keeps a margin eps in centred coordinates
            if name == 'ga': gl.append((bi, Li, s, c))
        pr = cp.Problem(cp.Minimize((sc*oof/h) @ y-e*pen), cons)   # objective relative to the Haar value h
        try:
            if SOLVER == 'SDPA':     # multiprecision SDPA-GMP (pip sdpa-multiprecision), for n >= 15
                pr.solve(solver='SDPA', mpfPrecision=256, epsilonStar=1e-30, epsilonDash=1e-30, maxIteration=400, print='no')
            else:
                pr.solve(solver='CLARABEL', max_iter=1000, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
        except cp.error.SolverError:
            return None
        if y.value is None: return None
        return (pr.value*h, pr.status, {bi: (Li, s, h*(c.dual_value+e*np.eye(len(s)))) for bi, Li, s, c in gl}, xc), sc*y.value   # undo the 1/h
    best_e = 0.0
    for r in range(r0, rounds):
        e = eps if r == rounds-1 else 0.0    # the eps reward biases the value far from the centre: last round only
        res = one(xc, e)
        if res is None: break
        best, x = res; best_e = e
        if log: log('  round %d: value %.12f (%s)' % (r+1, best[0], best[1]))
        xc = [Q(Fr(float((1-t)*ffloat(v)+t*w))) for v, w in zip(xc, x)]
        if ck and r < rounds-1: json.dump([r+1, [str(Fr(int(v.p), int(v.q))) for v in xc]], open(ck, 'w'))
    if best is not None and eps > 0 and best_e == 0:   # a later round failed: redo the last good centre with eps
        res = one(best[3], eps)
        if res is not None:
            best = res[0]
            if log: log('  eps round at the last good centre: value %.12f (%s)' % (best[0], best[1]))
    return best

def centred_multipliers(Y):
    """Y'_b = diag(s) Y_b diag(s), symmetrised in floating point: exact dyadic rationals (stored as float.hex)."""
    out = {}
    for bi, (Li, s, Yb) in Y.items():
        Yp = Yb*s[:, None]*s[None, :]; out[bi] = (Yp+Yp.T)/2
    return out

def adjoint(st, G, Zq):
    """c_p = sum_blocks tr[Gamma_b(e_p) Z_b]."""
    c = [Q(0)]*st.NP
    for bi, Z in Zq.items():
        for p in range(st.NP):
            A = G[bi][p]; m = A.nrows()
            c[p] += sum((A[i, j]*Z[i, j] for i in range(m) for j in range(m) if A[i, j] != 0), Q(0))
    return c

if __name__ == '__main__':
    nk = sys.argv[1].split(':'); n, eps = int(nk[0]), float(sys.argv[2]); k = int(nk[1]) if len(nk) > 1 else n//2
    key = str(n) if k == n//2 else '%d:%d' % (n, k); lams = [Fr(s) for s in sys.argv[3].split(',') if s != 'auto']; t0 = time.time()
    st = sw.Structure(n); on, oo = st.functionals(k); cache = os.environ.get('THREECOPY_CACHE')
    cf = os.path.join(cache, 'gamma_n%d.pkl' % n) if cache else None
    if cf and os.path.exists(cf):
        import pickle
        G = {bi: [flint.fmpq_mat([[Q(*e) for e in r] for r in M]) for M in arr] for bi, arr in pickle.load(open(cf, 'rb')).items()}
    else:
        G = sw.exact_gamma_map(st, log=lambda s: print(s, flush=True))
        if cf:
            import pickle
            pickle.dump({bi: [[[(int(M[i, j].p), int(M[i, j].q)) for j in range(M.ncols())] for i in range(M.nrows())] for M in arr] for bi, arr in G.items()}, open(cf, 'wb'))
    print('n=%d: %d parameters, exact gamma map (%.0fs)' % (n, st.NP, time.time()-t0), flush=True)
    ck = os.environ.get('THREECOPY_CHECKPOINT'); rf = ck + '.rec' if ck else None
    if rf and os.path.exists(rf):     # the solve finished in an earlier run: go straight to the exact checks
        val, status, rec = json.load(open(rf)); print('solve record loaded from %s' % rf, flush=True)
    else:
        val, status, Y, xc = solve(st, G, on, oo, eps=eps, log=lambda s: print(s, flush=True))
        print('perturbed SDP value %.12f (%s), min centred multiplier eigenvalue %.3e' % (val, status, min(np.linalg.eigvalsh(y[2]).min() for y in Y.values())), flush=True)
        rec = dict(eps=eps, centre=[str(Fr(int(v.p), int(v.q))) for v in xc],
                   Y={str(bi): [[float(v).hex() for v in r] for r in M] for bi, M in centred_multipliers(Y).items()})
        if rf: json.dump([val, status, rec], open(rf, 'w'))
    Zq = sw.gamma_multipliers(st, G, rec)
    if 'auto' in sys.argv[3].split(','):    # the SDP value truncated to 4..8 significant digits
        e = int(np.floor(np.log10(val)))
        lams += [Fr(int(np.floor(val*10**(d-1-e))), 10**(d-1-e)) for d in range(4, 9)]
    c = adjoint(st, G, Zq); best = None
    for lam in lams:
        ok = all(sw.is_pd(M) for M in Zq.values()) and all(sw.is_pd(D) for D in sw.omega_residual(st, on, oo, Q(lam), c))
        print('lambda %s = %.10f: %s' % (lam, float(lam), 'PASS' if ok else 'fail'), flush=True)
        if ok and (best is None or lam > best): best = lam
    out = os.environ.get('THREECOPY_DATA', os.path.join(HERE, '..', 'data', 'three_copy_sw_certificates.json'))
    data = json.load(open(out)) if os.path.exists(out) else {}
    if best is not None and (key not in data or Fr(data[key]['lam']) < best):
        data[key] = dict(lam=str(best), **rec)
        json.dump(data, open(out, 'w')); print('stored %s lambda = %s (%.0fs)' % (key, best, time.time()-t0))
