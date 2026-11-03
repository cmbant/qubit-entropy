#!/usr/bin/env python3
"""NV-1. EXPLORE (numerical evidence, not a proof): minimum of G = delta - C* d^2 over critical states at fixed tau.

Real-density model of lem:fqe-critical-real-density: A = (1+s)/2 U1 U1^T + (1-s)/2 U2 U2^T, s = sqrt(tau), U orthonormal
4x2 frame; delta from the three diagonals of H_j A H_j^T, d^2 from the sixteen plane functions F_P.  For each tau:
20000 random frames, the 25 lowest polished by Nelder-Mead.  The theorem (thm:fqe-p1-sharp) says G >= 0 with equality only at
tau = 0; the numbers below show G_min(tau)/tau close to 0.30 for all tau.
"""
import numpy as np
from scipy.optimize import minimize
import critical_cells as cc

C = float(cc.CSTAR.mid())
EMAX = float(cc.EMAX.mid())
H = [np.array([[float(x) for x in row] for row in rows]) for rows in cc.HROWS]
V = []
for (k, oth, sg) in cc.PLANES:
    Vk = np.zeros((4, 2))
    Vk[k, 0] = 1
    for n, j in enumerate(oth):
        Vk[j, 1] = sg[n]/np.sqrt(3)
    V.append(Vk)


def frame(x):
    a, b = x[..., :4], x[..., 4:]
    a = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b-(a*b).sum(-1, keepdims=True)*a
    return a, b/np.linalg.norm(b, axis=-1, keepdims=True)


def G(x, t):
    a, b = frame(x)
    s = np.sqrt(t)
    A = (1+s)/2*np.einsum('...i,...j->...ij', a, a)+(1-s)/2*np.einsum('...i,...j->...ij', b, b)
    E = 0
    for Hj in H:
        l = np.clip(np.einsum('ab,...bc,ac->...a', Hj, A, Hj), 1e-300, None)
        E = E-(l*np.log2(l)).sum(-1)/3
    F = 0
    for Vk in V:
        m = np.einsum('ik,...ij,jl->...kl', Vk, A, Vk)
        det = m[..., 0, 0]*m[..., 1, 1]-m[..., 0, 1]**2
        F = np.maximum(F, 0.5*(m[..., 0, 0]+m[..., 1, 1])+np.sqrt(np.clip(det, 0, None)))
    return EMAX-E-C*(2-2*np.sqrt(F))


rng = np.random.default_rng(5)
print('tau        min G        min G / tau')
for t in (0.0, 0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    X = rng.normal(size=(20000, 8))
    g = G(X, t)
    best = 9
    for i in np.argsort(g)[:25]:
        r = minimize(lambda x: float(G(x, t)), X[i], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14, maxiter=3000))
        best = min(best, r.fun)
    print('%-8g %12.3e %10.3f' % (t, best, best/t) if t > 0 else '%-8g   zero (|G| < 1e-12, the equality points)' % t)
    if t == 0:
        assert abs(best) < 1e-12
