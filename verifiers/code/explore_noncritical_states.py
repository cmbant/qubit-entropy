#!/usr/bin/env python3
"""NV-2. EXPLORE (numerical evidence, not a proof): delta - C* d^2 over non-critical pure four-qubit states.

A state is psi = (A_1 x ... x A_4) phi_v / norm with phi_v = (M x M) sum_i v_i |ii> a critical normal form and
A_i = exp(x_i . sigma) positive local filters (the SLOCC polar part; 6 + 12 real parameters = the 18 directions of the LU
quotient).  delta from the three balanced cut entropies; the orbit fidelity F = max over local unitaries and the two HS
orbits of |<HS|U psi>|^2 by BFGS from several starts (validated against the 32-ray formula on critical states).
(1) the equality state with random filters of size eps: min G/eps^2;  (2) random filters of size 0.05-0.4 on critical states within 0.05-0.4 (normal-form coordinates) of the equality state.
"""
import itertools
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]])/np.sqrt(2)
CST = 0.19057894701272688786
EMAX = 1+0.5*np.log2(3)
SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]).astype(complex)]


def phi_of_v(v):
    X = np.zeros((4, 4), complex)
    for i in range(4):
        X[i, i] = v[i]
    return (M@X@M.T).reshape(2, 2, 2, 2)


def delta(psi):
    out = []
    for cut in [(0, 1), (0, 2), (0, 3)]:
        rest = [i for i in range(4) if i not in cut]
        s = np.linalg.svd(np.transpose(psi, list(cut)+rest).reshape(4, 4), compute_uv=False)**2
        s = s[s > 1e-300]
        out.append(-(s*np.log2(s)).sum())
    return EMAX-np.mean(out)


eta = np.array([1/np.sqrt(2), 1j/np.sqrt(6), 1j/np.sqrt(6), 1j/np.sqrt(6)])
HS = (phi_of_v(eta), phi_of_v(eta.conj()))


def su2(q):
    a, b, c, d = q/np.linalg.norm(q)
    return np.array([[a+1j*b, c+1j*d], [-c+1j*d, a-1j*b]])


def apply(Us, psi):
    return np.einsum('ia,jb,kc,ld,abcd->ijkl', Us[0], Us[1], Us[2], Us[3], psi)


def fidelity(psi, rng, starts=4):
    psi = psi/np.linalg.norm(psi)
    best = 0
    inits = [np.tile([1, 0, 0, 0.], 4)]+[rng.normal(size=16) for _ in range(starts)]
    for Href in HS:
        for x0 in inits:
            f = lambda x: -abs(np.vdot(Href, apply([su2(x[4*i:4*i+4]) for i in range(4)], psi)))**2
            r = minimize(f, x0, method='BFGS', options=dict(gtol=1e-10))
            best = max(best, -r.fun)
    return best


def F32(v):
    best = 0
    for i in range(4):
        for eps in itertools.product((1, -1), repeat=3):
            e = np.zeros(4, complex)
            e[i] = 1/np.sqrt(2)
            for n, j in enumerate([k for k in range(4) if k != i]):
                e[j] = 1j*eps[n]/np.sqrt(6)
            best = max(best, abs(np.vdot(e, v))**2)
    return best


def state(x):
    v = x[:4]+1j*x[4:8]
    v = v/np.linalg.norm(v)
    psi = apply([expm(sum(x[8+3*i+k]*SIG[k] for k in range(3))) for i in range(4)], phi_of_v(v))
    return psi/np.linalg.norm(psi)


rng = np.random.default_rng(1)
v = rng.normal(size=4)+1j*rng.normal(size=4)
v /= np.linalg.norm(v)
print('orbit fidelity (BFGS) vs 32-ray formula on a critical state: %.9f %.9f' % (fidelity(phi_of_v(v), rng, 8), F32(v)))
a, b = np.cos(np.pi/8)/np.sqrt(2), np.sin(np.pi/8)/np.sqrt(2)
v0 = np.array([a, 1j*a, b, 1j*b])
phi0 = phi_of_v(v0)


def G(psi, rr):
    return delta(psi)-CST*(2-2*np.sqrt(fidelity(psi, rr, 2)))


assert abs(G(phi0, rng)) < 1e-9
print('G at the equality state: 0 (|G| < 1e-9)')
for eps in (0.02, 0.05, 0.1, 0.2):
    res = []
    for _ in range(12):
        Us = []
        for i in range(4):
            c = rng.normal(size=3)
            c /= np.linalg.norm(c)
            Us.append(expm(eps*sum(ci*s for ci, s in zip(c, SIG))))
        res.append(G(apply(Us, phi0), rng)/eps**2)
    print('random filters of size %.2f at the equality state: min G/eps^2 = %.2f, median %.2f' % (eps, min(res), np.median(res)))

# (2) filtered states near the equality state
def random_near_apex(rng, scale):
    """equality-state coordinates v0 plus a random perturbation of size `scale` in the normal-form variables"""
    v = v0+scale*(rng.normal(size=4)+1j*rng.normal(size=4))
    return v/np.linalg.norm(v)


t0 = time.time()
rows = []
for scale in (0.05, 0.15, 0.4):
    for eps in (0.05, 0.15, 0.4):
        gs = []
        for _ in range(10):
            v = random_near_apex(rng, scale)
            Us = []
            for i in range(4):
                c = rng.normal(size=3)
                c /= np.linalg.norm(c)
                Us.append(expm(eps*sum(ci*s_ for ci, s_ in zip(c, SIG))))
            psi = apply(Us, phi_of_v(v))
            psi = psi/np.linalg.norm(psi)
            d2 = 2-2*np.sqrt(fidelity(psi, rng, 3))
            gs.append((delta(psi)-CST*d2, delta(psi)/d2))
        gs = np.array(gs)
        print('critical part within %.2f of the equality state, filters of size %.2f: min G = %.4f, min delta/d^2 = %.4f (10 states)'
              % (scale, eps, gs[:, 0].min(), gs[:, 1].min()))
        assert gs[:, 0].min() > -1e-9
print('(%.0f s)' % (time.time()-t0))
