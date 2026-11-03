#!/usr/bin/env python3
"""EX: exact (sympy) tie and gradient identities at the eight chart equality points (critical_sharp_coefficient.tex, Lemma
fqe-critical-local-cone).  No floating point and no ball arithmetic is used for the statements below.

(S) V4 symmetry at (0,0,0,1): for z -> (s1 z1, s2 z2, s1 s2 z3, z4, s2 z5, s1 z6) (z = (P,Q,P',1-Q',w1,w2)) the chart matrix
    satisfies  T(sigma y) = D T(y) D',  D = diag(1,s1,s2), D' = diag(1,s1 s2,1)  as an identity in the six variables; D C^{ji} D' is
    again a signed permutation matrix C^{ji'} of the same cut j (so delta is invariant), and for every plane p there is a plane
    pi(p) with  tr m_p(D T D') = tr m_pi(p)(T),  det m_p(D T D') = det m_pi(p)(T)  as polynomials in the nine entries of T
    (so F_p(sigma y) = F_pi(p)(y) and G_p(sigma y) = G_pi(p)(y)).
(A) at each of the eight points (four apexes x two parity sectors): exactly four planes attain the maximal F_P, all others are
    strictly smaller; F* = max F satisfies 4F* = (1+1/sqrt3)(1+sqrt(2/3)); the twelve probabilities are (2+-sqrt2)/8 and 1/4, so
    delta = EMAX - E = A0 - (1+2H2(p))/3 with p = (2+sqrt2)/4; hence delta = C* (2-2 sqrt F*) with C* DEFINED by this tie, i.e.
    G_P = 0 at the apex for the four tied planes.
(G) the sum over the four tied planes of the gradient of G_P = delta - C*(2 - 2 sqrt F_P) at the apex vanishes in all six
    coordinates, as an identity in C*, ln p, ln(1-p), ln 2 (the log-probabilities enter only through these symbols).
"""
import itertools
import sys
import sympy as sp
from critical_cells import _C, PLANES, _H1, _H2M

s3, s2 = sp.sqrt(3), sp.sqrt(2)
RAYS = [_C[(j, i)] for j in range(3) for i in range(4)]
Hm = [sp.eye(4), sp.Matrix(_H1)/2, sp.Matrix(_H2M)/2]
RAYROW = [Hm[j].row(i).T for j in range(3) for i in range(4)]
P, Q, Pp, Qp, w1, w2 = ys = sp.symbols('P Q Pp Qp w1 w2', real=True)
Ts = sp.Matrix(3, 3, sp.symbols('T0:9'))
a_, b_, l2 = sp.symbols('lnp ln1mp ln2')
x_ = sp.Symbol('x')


def tof(T):
    return [sum(sg*T[a, b] for a, b, sg in ray) for ray in RAYS]


def mplane(p, T):
    k, oth, sg = PLANES[p]
    m = sp.eye(2)/4
    for t, h in zip(tof(T), RAYROW):
        u = sp.Matrix([h[k], (sg[0]*h[oth[0]]+sg[1]*h[oth[1]]+sg[2]*h[oth[2]])/s3])
        m += t*u*u.T/4
    return m


def chart(sg):
    r = sp.Matrix([1, P, Q])/sp.sqrt(1+P**2+Q**2)
    xa = sp.Matrix([P, -1, 0])/sp.sqrt(P**2+1)
    ya = r.cross(xa)
    s = sp.Matrix([sg, Pp, Qp])/sp.sqrt(1+Pp**2+Qp**2)
    xb = sp.Matrix([Pp, -sg, 0])/sp.sqrt(Pp**2+1)
    yb = s.cross(xb)
    return r*s.T+w1*(xa*xb.T-ya*yb.T)+w2*(xa*yb.T+ya*xb.T)


def is_zero(e):
    e = sp.simplify(e)
    return e == 0 or sp.minimal_polynomial(e, x_) == x_


# ---- (S) -----------------------------------------------------------------------------------------------------------------
for sg in (1, -1):
    Tc = chart(sg)
    for e1, e2 in itertools.product((1, -1), repeat=2):
        sig = {P: e1*P, Q: e2*Q, Pp: e1*e2*Pp, w1: e2*w1, w2: e1*w2}
        D, Dp = sp.diag(1, e1, e2), sp.diag(1, e1*e2, 1)
        diff = (Tc.subs(sig, simultaneous=True)-D*Tc*Dp).applyfunc(sp.simplify)
        assert diff == sp.zeros(3, 3), (sg, e1, e2, diff)
for e1, e2 in itertools.product((1, -1), repeat=2):
    D, Dp = sp.diag(1, e1, e2), sp.diag(1, e1*e2, 1)
    rho = []
    for ray in RAYS:
        C = sp.zeros(3, 3)
        for a, b, s in ray:
            C[a, b] = s
        C2 = D*C*Dp
        hit = [q for q, r2 in enumerate(RAYS) if all(C2[a, b] == s for a, b, s in r2) and sum(abs(x) for x in C2) == 3]
        assert len(hit) == 1
        rho.append(hit[0])
    assert all(rho[r]//4 == r//4 for r in range(12)) and sorted(rho) == list(range(12))      # same cut, bijection
    TD = D*Ts*Dp
    inv = [(sp.expand(mplane(p, Ts).trace()), sp.expand(mplane(p, Ts).det())) for p in range(16)]
    for p in range(16):
        m = mplane(p, TD)
        key = (sp.expand(m.trace()), sp.expand(m.det()))
        assert sum(1 for q in range(16) if all(sp.expand(u-v) == 0 for u, v in zip(key, inv[q]))) >= 1, (e1, e2, p)
print('[PASS] EX-S exact V4 symmetry at (0,0,0,1) in both sectors: T(sigma y) = D T D\' identically in (P,Q,P\',Q\',w1,w2); the twelve rays are '
      'permuted inside their cut (delta invariant); every plane is sent to a plane with the same trace and determinant of m_P (polynomial '
      'identity in the nine entries of T), so G_P(sigma y) = G_pi(P)(y) for all four sign choices')

# ---- (A), (G) ------------------------------------------------------------------------------------------------------------
Fstar = s2/12+sp.Rational(1, 4)+sp.sqrt(6*s2+9)/12
assert is_zero((1+1/s3)*(1+sp.sqrt(sp.Rational(2, 3)))/4-Fstar)
pval = (2+s2)/4
lamlog = {sp.Rational(1, 4): -2*l2, (2+s2)/8: a_-l2, (2-s2)/8: b_-l2}


def lnlam(v):
    for val, sym in lamlog.items():
        if is_zero(v-val):
            return sym
    raise AssertionError(v)


Cst = sp.Symbol('Cstar')
APEXES = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]
for sg in (1, -1):
    Tc = chart(sg)
    t_sym = tof(Tc)
    for ap in APEXES:
        pt = dict(zip(ys, list(ap)+[0, 0]))
        T0 = Tc.subs(pt)
        lam = [sp.simplify((1+x)/4) for x in tof(T0)]
        L = [lnlam(v) for v in lam]
        E = -sum(v*l for v, l in zip(lam, L))/(3*l2)                     # (1/3) sum_j H(lambda_j.) , log2 = ln/ln2
        H2 = -(pval*a_+(1-pval)*b_)/l2
        assert sp.simplify(E-(4+2*H2)/3) == 0                              # delta = EMAX - E = A0 - (1+2 H2)/3
        Fs = []
        for p in range(16):
            m = mplane(p, T0)
            Fs.append(sp.simplify(m.trace()/2+sp.sqrt(m.det())))
        tied = [p for p in range(16) if is_zero(Fs[p]-Fstar)]
        assert len(tied) == 4, (sg, ap, tied)
        for p in range(16):
            if p not in tied:
                assert (Fs[p]-Fstar).is_negative, (sg, ap, p, Fs[p])
        dt = [[sp.simplify(sp.diff(x, y).subs(pt)) for y in ys] for x in t_sym]
        tot = [0]*6
        for p in tied:
            m = mplane(p, Tc)
            m0 = m.subs(pt)
            for k, y in enumerate(ys):
                dm = m.diff(y).subs(pt)
                dF = dm.trace()/2+(m0.adjugate()*dm).trace()/(2*sp.sqrt(m0.det()))
                dd2 = -dF/sp.sqrt(Fs[p])                                   # d(2-2 sqrt F)
                tot[k] += sum((1+L[i])*dt[i][k] for i in range(12))/(12*l2)-Cst*dd2
        for k in range(6):
            assert sp.simplify(tot[k]) == 0, (sg, ap, k, sp.simplify(tot[k]))
        print('[PASS] EX-A/G sector %+d apex %s: tied planes %s (F = F* exactly, all others strictly smaller), probabilities (2+-sqrt2)/8 and 1/4, '
              'delta = C*(2-2 sqrt F*) with C* defined by the tie, and the sum of the four tied gradients is 0 in all six coordinates'
              % (sg, ap, tied))
print('[PASS] EX 4F* = (1+1/sqrt3)(1+sqrt(2/3)) exactly (minimal polynomial of the difference is x); all eight equality points checked')
