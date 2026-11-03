#!/usr/bin/env python3
"""FC-0--FC-4: exact certificate for the four-copy lower bound on the mean three-body purity of seven qubits.

Statement: thm:fourcopy (in four_copy_bound.tex)
The certificate (data/four_copy_certificate_n7.json, from search_four_copy_certificate.py) gives integer
positive definite multipliers Z_gamma for the flip blocks.  For every omega parameter p the exact residual
  d_p = oo_p - lam on_p - c_p/(12^7 Qz),   c_p = sum_gamma <Z_gamma, Q'^T H (12 T)^{(x)7} X_p Q'>,
is computed (four_copy_exact.py), and each omega block [d_kl] is shown positive definite.  For the twirled
moment X of any unit z, x = parameters of X, the identity
  pi_3(z) - lam = <d, x> + sum_gamma <Z_gamma, gamma(x)>/Qz,   <d, x> = sum_blocks <[d_kl], omega(x)>
then gives pi_3 >= lam, because omega(x) >= 0 and gamma(x) >= 0 for genuine states.

FC-0 the exact site data (matrix units, flip maps, 12T integral);
FC-1 regression on genuine three-qubit states, in the exact coordinates: the S_3-averaged twirled moment
     lies in the parameter space, tr X = 1, the objective is the mean one-body purity, omega(x) >= 0,
     the flip maps reproduce the directly computed flipped moments, and the flip blocks are >= 0
     (numerical; this checks the construction, not the bound);
FC-2 the modular adjoint contraction against a direct forward computation (n = 3, 4, random integer Z);
FC-3 the seven-qubit certificate: Z_gamma positive definite (Sylvester, exact) and every omega residual
     block positive definite (exact, CRT under a rigorous magnitude bound);
FC-4 the bound.
"""
import itertools, json, math, os, sys, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import four_copy_exact as fx
t0 = time.time()

# ---------------------------------------------------------------- FC-0
for T in (fx.T1, fx.T2):
    assert len(T) == 14 and all(isinstance(v, Fr) for r in T for v in r)
assert all(fx.leading_minors_positive([[Fr(int(v)) for v in row] for row in fx.GRAM[t]]) for t in (1,2))
print('FC-0 PASS: integer S_4 irreps and matrix units (Schur orthogonality, exact); flip maps T1, T2 exact with 12T integral;'
      ' Gram forms positive definite: %s' % all(np.linalg.eigvalsh(fx.GRAM[t].astype(float)).min() > 0 for t in (1, 2)), flush=True)

# ---------------------------------------------------------------- helpers for genuine states
E16 = [sum(float(c[k])*fx.OP[k] for k in range(24)) for c in fx.EMB]                 # matrix units as 16x16
G = np.array([[np.sum(fx.OP[k]*E) for E in E16] for k in range(24)])                # tr(P_pi^T E_c)
Gp = np.linalg.pinv(G)
def moment(zs):
    """Integer-coordinate tensor (14^n) of the U(2)^n twirl of |z1 z2 z3 z4><z1 z2 z3 z4| (z_c in (C^2)^n)."""
    n = int(round(math.log2(zs[0].size)))
    Psi = np.einsum('a,b,c,d->abcd', *zs).reshape((2,)*(4*n))                        # axes (copy, site)
    Psi = Psi.transpose([c*n+i for i in range(n) for c in range(4)]).reshape((16,)*n)  # site-major, copy bits
    b = np.zeros((24,)*n, dtype=complex)
    for w in itertools.product(range(24), repeat=n):
        V = Psi
        for i, k in enumerate(w): V = np.moveaxis(np.tensordot(fx.OP[k].astype(float), V, axes=([1], [i])), 0, i)
        b[w] = np.sum(V*Psi.conj())                                                   # tr(P_w^T |Psi><Psi|)
    X = b
    for i in range(n): X = np.moveaxis(np.tensordot(Gp, X, axes=([1], [i])), 0, i)
    return X
def apply_T(Ti, X):
    for k in range(X.ndim): X = np.moveaxis(np.tensordot(Ti.astype(float)/12, X, axes=([1], [k])), 0, k)
    return X
def block(X, tp): return fx.from_sub(X[tuple(fx.SL[t] for t in fx.tlist(tp))], tp)
def Fmat(bl, k, l):
    H = fx.kronlist([fx.GRAM[t] for t in fx.tlist(bl['tp'])]).astype(float); A = np.array(bl['adj'], dtype=float)
    R = [r.astype(float) for r in bl['R']]
    if bl['triv']: return np.ones((1, 1))
    return R[k] @ A @ R[l].T @ H + R[l] @ A @ R[k].T @ H
def purity(z, n, kk):
    psi = z.reshape((2,)*n); tot = 0
    for S in itertools.combinations(range(n), kk):
        M = psi.transpose(list(S)+[i for i in range(n) if i not in S]).reshape(2**kk, -1); r = M @ M.conj().T
        tot += np.real(np.trace(r @ r))
    return tot/math.comb(n, kk)

# ---------------------------------------------------------------- FC-1
n = 3; B3 = fx.bases(n); bl3, pa3 = fx.omega_data(n, B3); on3, oo3 = fx.obj_norm(n, 1, bl3, pa3)
eps = np.kron(np.kron([[0, 1], [-1, 0]], [[0, 1], [-1, 0]]), [[0, 1], [-1, 0]])
rng = np.random.default_rng(7); worst = dict(res=0, on=0, oo=0, om=np.inf, flip=0, gam=np.inf)
for trial in range(3):
    z = rng.normal(size=8)+1j*rng.normal(size=8); z /= np.linalg.norm(z); zt = eps @ z.conj()
    X = moment([z, z, z, z])
    worst['flip'] = max(worst['flip'], np.abs(apply_T(fx.T1i, X)-moment([z, z, z, zt])).max(),
                        np.abs(apply_T(fx.T2i, X)-moment([z, z, zt, zt])).max())
    Xs = sum(X.transpose(p) for p in itertools.permutations(range(n))).real/math.factorial(n)
    x = np.zeros(len(pa3))
    for tp in fx.types(n):
        idx = [i for i, (bi, k, l) in enumerate(pa3) if bl3[bi]['tp'] == tp]; Xb = block(Xs, tp)
        if not idx: worst['res'] = max(worst['res'], np.abs(Xb).max()); continue
        A = np.array([Fmat(bl3[pa3[i][0]], pa3[i][1], pa3[i][2]).ravel() for i in idx]).T
        sol = np.linalg.lstsq(A, Xb.ravel(), rcond=None)[0]; x[idx] = sol
        worst['res'] = max(worst['res'], np.abs(A @ sol-Xb.ravel()).max())
    worst['on'] = max(worst['on'], abs(np.dot([float(v) for v in on3], x)-1))
    worst['oo'] = max(worst['oo'], abs(np.dot([float(v) for v in oo3], x)-purity(z, n, 1)))
    for bi, bl in enumerate(bl3):
        m = bl['m']; E = np.zeros((m, m))
        for i, (bj, k, l) in enumerate(pa3):
            if bj == bi:
                if k == l: E[k, k] = x[i]
                else: E[k, l] = E[l, k] = x[i]/2
        worst['om'] = min(worst['om'], np.linalg.eigvalsh(E).min())
    for name, Ti in (('g1', fx.T1i), ('g2', fx.T2i)):
        Y = apply_T(Ti, Xs)
        for tp in fx.types(n):
            H = fx.kronlist([fx.GRAM[t] for t in fx.tlist(tp)]).astype(float)
            for (_, _, _, Q) in B3[(tp, name)]:
                M = Q.T.astype(float) @ H @ block(Y, tp) @ Q.astype(float); M = (M+M.T)/2
                worst['gam'] = min(worst['gam'], np.linalg.eigvalsh(M).min()/max(1, np.abs(M).max()))
ok1 = worst['res'] < 1e-10 and worst['on'] < 1e-10 and worst['oo'] < 1e-10 and worst['om'] > -1e-10 and worst['flip'] < 1e-10 and worst['gam'] > -1e-10
assert ok1, worst
print('FC-1 %s (numerical regression, 3 random three-qubit states): parametrisation residual %.1e, |tr X - 1| %.1e, '
      '|objective - purity| %.1e, min eig omega %.1e, flip maps vs direct flipped moments %.1e, min eig flip blocks %.1e'
      % ('PASS' if ok1 else 'FAIL', worst['res'], worst['on'], worst['oo'], worst['om'], worst['flip'], worst['gam']), flush=True)

# ---------------------------------------------------------------- FC-2
def forward_c(n, B, blocks, params, Zint, keys):
    out = []
    for (bi, k, l) in params:
        bl = blocks[bi]; X = np.zeros((fx.NC,)*n); sub = fx.to_sub(Fmat(bl, k, l), bl['tp'])
        for s, p in fx.orderings(bl['tp']).items():
            order = [0]*n
            for j in range(n): order[p[j]] = j
            X[tuple(fx.SL[t] for t in s)] += sub.transpose(order)
        tot = 0.0
        for name, Ti in (('g1', fx.T1i), ('g2', fx.T2i)):
            Y = apply_T(Ti, X)*12**n
            for (nm, (tp, li)), Z in zip(keys, Zint):
                if nm != name: continue
                Q = B[(tp, name)][li][3].astype(float); H = fx.kronlist([fx.GRAM[t] for t in fx.tlist(tp)]).astype(float)
                tot += np.sum(np.array(Z, dtype=float)*(Q.T @ H @ block(Y, tp) @ Q))
        out.append(tot)
    return out
worst2 = 0
for n in (3, 4):
    B = B3 if n == 3 else fx.bases(n); bl, pa = (bl3, pa3) if n == 3 else fx.omega_data(n, B)
    keys, Zint = [], []
    for name in ('g1', 'g2'):
        for tp in fx.types(n):
            for li, (_, _, _, Q) in enumerate(B[(tp, name)]):
                A = rng.integers(-3, 4, size=(Q.shape[1],)*2); keys.append((name, (tp, li))); Zint.append((A @ A.T).tolist())
    ce = fx.c_exact(n, B, bl, pa, Zint, keys); cf = forward_c(n, B, bl, pa, Zint, keys)
    worst2 = max(worst2, max(abs(a-b)/max(1, abs(b)) for a, b in zip(ce, cf)))
print('FC-2 %s: modular adjoint contraction + CRT equals the direct forward computation (n = 3, 4; random integer Z), '
      'max relative difference %.1e' % ('PASS' if worst2 < 1e-9 else 'FAIL', worst2), flush=True)
assert worst2 < 1e-9, worst2

# ---------------------------------------------------------------- FC-3
cert = json.load(open(os.path.join(HERE, '..', 'data', 'four_copy_certificate_n7.json')))
n = cert['n']; lam = Fr(cert['lam']); Qz = cert['Qz']
keys = [(b['flip'], (tuple(b['type']), b['index'])) for b in cert['blocks']]
Zint = [[[int(v) for v in r] for r in b['Z']] for b in cert['blocks']]
zpd = all(fx.leading_minors_positive([[Fr(v) for v in r] for r in Z]) for Z in Zint)
assert zpd
B = fx.bases(n); print('  n=7 rational bases built (%.0fs)' % (time.time()-t0), flush=True)
assert all(len(Z) == B[(tp, nm)][li][3].shape[1] for (nm, (tp, li)), Z in zip(keys, Zint))
blocks, params = fx.omega_data(n, B); on, oo = fx.obj_norm(n, 3, blocks, params)
assert n == 7 and lam == Fr(211,1600) and Qz > 0 and len(params) == 167
c = fx.c_exact(n, B, blocks, params, Zint, keys, log=lambda s: print('  '+s, flush=True))
ok, mn = fx.exact_check(blocks, params, on, oo, lam, c, 12**n*Qz)
assert ok
print('FC-3 %s: %d flip multipliers, all positive definite: %s; %d omega parameters in %d blocks, every residual block '
      'positive definite: %s (smallest eigenvalue %.3e)' % ('PASS' if zpd and ok else 'FAIL', len(Zint), zpd, len(params),
                                                           len(blocks), ok, mn), flush=True)
print('FC-4 %s: every pure seven-qubit state has mean three-body purity pi_3 >= %s = %.6f  (%.0fs)'
      % ('PASS' if zpd and ok and ok1 and worst2 < 1e-9 else 'FAIL', lam, float(lam), time.time()-t0), flush=True)
