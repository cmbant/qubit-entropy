#!/usr/bin/env python3
"""FI-1--FI-5: the determining-point proof of the spin-flip gradient identity (FGI).

Proved in: sup_general_envelope.tex#lem:fqe-flip-gradient (supplement; the point-evaluation proof of (FGI)). The direct
proof in the supplement (flip_gradient_proof.tex) is checked by verify_envelope_flip_gradient.py.

Notation: w = x (x) y (x) t (three copies), U_i = S_i prod_{j!=i} L_j w, V_i = S_i prod_{j!=i}(L_j+M_j) w,
pair vectors P_ij = (R_i^{(i)} - R_j^{(j)}) acting as in (S). For a state z put zt = eps^{(x)4} conj(z) and let
O(x,y,t) be the residual (sqrt(3/8)(V_i - 8U_i))_i (+) (sqrt(3/2) P_ij)_{i<j} of w = x(x)y(x)t, symmetrised in x, y;
the pair vectors of w use R_i[a,b] = sum x[a,..] (eps^{(x)4} t)[b,..] acting on y. Then H(z) = ||O(z,z,zt)||^2 by (S),
and with A = O(z,z,zt), C = O(z,z,z), D = O(z,zt,zt), X_i = (V_i - 8U_i)(z,z,zt):

  (FGI)  <A,C> + 2<D,A> = (9/8) sum_i X_i^T eps^{(x)6} X_i + s * lambda,
         s = z^T eps^{(x)4} z,  lambda = (5/4)N^2 + (11/8)W_1 + (3/4)P - (19/8)|s|^2.

Both sides are local-unitary invariant polynomials of bidegree (4,2) in (z, conj z). Conj z is treated as an independent
vector w; the conjugate of a vector built from (z,w) is the same construction with z and w swapped.
FI-1: the SU(2)^4-invariants of bidegree (4,2) form a space of dimension 19 (character constant term) [exact].
FI-2: 19 explicit contraction invariants (eps between two z-legs or two w-legs, delta between a z-leg and a w-leg, at
      every site) have an evaluation matrix of exact rank 19 at 24 integer points [exact, over Q]. Hence an invariant of
      bidegree (4,2) that vanishes at those points vanishes identically.
FI-3: (S) in this normalisation: ||A||^2 = F - (3/4)NW_1 exactly at every point [exact].
FI-4: (FGI) holds exactly at every point, hence identically [exact].
FI-5: negative regressions: the identity fails at the points if 9/8 is replaced by 1, or if lambda is dropped [exact].
"""
# log: verify_flip_gradient_interpolation.txt
from collections import defaultdict
from fractions import Fraction
from itertools import permutations
import numpy as np
import sympy as sp

res = []


def check(tag, ok, msg):
    res.append(bool(ok))
    print(('[PASS] ' if ok else '[FAIL] ') + tag + ' ' + msg)


EPS = np.array([[0, 1], [-1, 0]], dtype=object)
LOP = np.zeros((2, 2, 2, 2), dtype=object)   # L|xyt> = eps_yt |x>,   index [out, x, y, t]
MOP = np.zeros((2, 2, 2, 2), dtype=object)   # M|xyt> = eps_xt |y>
for a in range(2):
    for b in range(2):
        for c in range(2):
            LOP[a, a, b, c] += EPS[b, c]
            MOP[b, a, b, c] += EPS[a, c]
SYM6 = np.zeros((2,) * 6, dtype=object)       # 6 * symmetriser of three slots
for b in np.ndindex(2, 2, 2):
    for p in permutations(range(3)):
        SYM6[tuple(b[p[k]] for k in range(3)) + b] += 1
LET = 'abcdefghijkl'


def e4(t):
    return np.einsum('ae,bf,cg,dh,efgh->abcd', EPS, EPS, EPS, EPS, t)


def uv(x, y, t):
    """6U_i, 6V_i of w = x(x)y(x)t (not symmetrised); output legs in site order."""
    w = np.einsum('abcd,efgh,ijkl->aeibfjcgkdhl', x, y, t)   # site-major (x1,y1,t1,x2,...)
    out = []
    for i in range(4):
        pair = []
        for op in (LOP, LOP + MOP):
            spec, ops, outs = [], [], ''
            for s in range(4):
                a, b, c = LET[3 * s:3 * s + 3]
                if s == i:
                    o = chr(65 + 3 * s) + chr(66 + 3 * s) + chr(67 + 3 * s)
                    spec.append(o + a + b + c)
                    ops.append(SYM6)
                else:
                    o = chr(65 + 3 * s)
                    spec.append(o + a + b + c)
                    ops.append(op)
                outs += o
            pair.append(np.einsum(LET + ',' + ','.join(spec) + '->' + outs, w, *ops, optimize='greedy'))
        out.append(pair)
    return out


def rz(x, t, y, i):
    xb = e4(t)
    xm = np.moveaxis(x, i, 0).reshape(2, -1)
    bm = np.moveaxis(xb, i, 0).reshape(2, -1)
    r = xm.dot(bm.T)
    return np.moveaxis(np.tensordot(r, y, axes=([1], [i])), 0, i)


def raw(x, y, t):
    """(6(V_i - 8U_i))_i and pair vectors P_ij of x(x)y(x)t, unsymmetrised in x, y."""
    xs = [v - 8 * u for u, v in uv(x, y, t)]
    rv = [rz(x, t, y, i) for i in range(4)]
    ps = [rv[i] - rv[j] for i in range(4) for j in range(i + 1, 4)]
    return xs, ps


def blocks(z, w):
    """Blocks of A, C and 2D for the formal pair (z, w = conj z)."""
    zt = e4(w)
    A = raw(z, z, zt)
    C = raw(z, z, z)
    d1, d2 = raw(z, zt, zt), raw(zt, z, zt)
    D2 = ([a + b for a, b in zip(d1[0], d2[0])], [a + b for a, b in zip(d1[1], d2[1])])
    return A, C, D2


def dot(a, b):
    return sum((int(u) * int(v) for u, v in zip(a.ravel(), b.ravel())), 0)


def eps_all(x):
    for leg in range(x.ndim):
        x = np.moveaxis(np.tensordot(EPS, x, axes=([1], [leg])), 0, leg)
    return x


def marg(z, w, keep):
    rest = [s for s in range(4) if s not in keep]
    zz = np.transpose(z, list(keep) + rest).reshape(2 ** len(keep), -1)
    ww = np.transpose(w, list(keep) + rest).reshape(2 ** len(keep), -1)
    return zz.dot(ww.T)


def tr(m):
    return sum((m[k, k] for k in range(m.shape[0])), 0)


def scalars(z, w):
    N = dot(z, w)
    P = T = 0
    for j in (1, 2, 3):
        R = marg(z, w, [0, j])
        R2 = R.dot(R)
        P += tr(R2)
        T += tr(R2.dot(R))
    S1 = sum((tr(marg(z, w, [i]).dot(marg(z, w, [i]))) for i in range(4)), 0)
    W1 = 2 * S1 - 4 * N * N
    s, sb = dot(z, e4(z)), dot(w, e4(w))
    F = 17 * N * P - 12 * T - 12 * N ** 3
    return N, P, T, W1, s, sb, F


def sides(z, w, coef=Fraction(9, 8), lam_on=True):
    A, C, D2 = blocks(z, w)
    Ab, Cb, D2b = blocks(w, z)            # formal conjugates
    lhs = Fraction(3, 8 * 36) * sum((dot(Ab[0][i], C[0][i]) + dot(D2b[0][i], A[0][i]) for i in range(4)), 0) \
        + Fraction(3, 2) * sum((dot(Ab[1][k], C[1][k]) + dot(D2b[1][k], A[1][k]) for k in range(6)), 0)
    main = coef / 36 * sum((dot(A[0][i], eps_all(A[0][i])) for i in range(4)), 0)
    N, P, T, W1, s, sb, F = scalars(z, w)
    lam = Fraction(5, 4) * N * N + Fraction(11, 8) * W1 + Fraction(3, 4) * P - Fraction(19, 8) * s * sb
    hA = Fraction(3, 8 * 36) * sum((dot(Ab[0][i], A[0][i]) for i in range(4)), 0) \
        + Fraction(3, 2) * sum((dot(Ab[1][k], A[1][k]) for k in range(6)), 0)
    return lhs, main + (s * lam if lam_on else 0), hA, F - Fraction(3, 4) * N * W1


# ---------------- FI-1: dimension of the invariant space ----------------
def pmul(A, B):
    C = defaultdict(Fraction)
    for ka, va in A.items():
        for kb, vb in B.items():
            C[tuple(x + y for x, y in zip(ka, kb))] += va * vb
    return {k: v for k, v in C.items() if v}


def powsum(k):
    P = {(0, 0, 0, 0): Fraction(1)}
    for i in range(4):
        e, f = [0] * 4, [0] * 4
        e[i], f[i] = k, -k
        P = pmul(P, {tuple(e): Fraction(1), tuple(f): Fraction(1)})
    return P


ps = [None] + [powsum(k) for k in range(1, 5)]
hs = [{(0, 0, 0, 0): Fraction(1)}]
for mdeg in range(1, 5):           # Newton: m h_m = sum_k p_k h_{m-k}
    acc = defaultdict(Fraction)
    for k in range(1, mdeg + 1):
        for key, val in pmul(ps[k], hs[mdeg - k]).items():
            acc[key] += val / mdeg
    hs.append({k: v for k, v in acc.items() if v})
weyl = {(0, 0, 0, 0): Fraction(1)}
for i in range(4):
    e = [0] * 4
    e[i] = 2
    weyl = pmul(weyl, {(0, 0, 0, 0): Fraction(1), tuple(e): Fraction(-1)})
dim42 = pmul(pmul(hs[4], hs[2]), weyl).get((0, 0, 0, 0), 0)
dim22 = pmul(pmul(hs[2], hs[2]), weyl).get((0, 0, 0, 0), 0)
check('FI-1', dim42 == 19 and dim22 == 8,
      f'dim of SU(2)^4 invariants: bidegree (4,2) = {dim42}, (2,2) = {dim22} (purities: 8)')

# ---------------- points and contraction invariants ----------------
rng = np.random.default_rng(20261029)
NPTS = 24
pts = [(rng.integers(-2, 3, size=(2,) * 4).astype(object), rng.integers(-2, 3, size=(2,) * 4).astype(object))
       for _ in range(NPTS)]
DELTA = np.eye(2, dtype=int).astype(object)


def matchings(legs):
    if not legs:
        yield []
        return
    a = legs[0]
    for k in range(1, len(legs)):
        b = legs[k]
        for rest in matchings(legs[1:k] + legs[k + 1:]):
            yield [(a, b)] + rest


MATCH = list(matchings(list(range(6))))        # legs 0..3: copies of z, 4..5: copies of w


def network(ms, z, w):
    tens = [z, z, z, z, w, w]
    letters = [[chr(97 + 4 * c + s) for s in range(4)] for c in range(6)]
    spec = [''.join(letters[c]) for c in range(6)]
    ops = list(tens)
    for s, m in enumerate(ms):
        for a, b in MATCH[m]:
            ops.append(DELTA if (a < 4) != (b < 4) else EPS)
            spec.append(letters[a][s] + letters[b][s])
    return np.einsum(','.join(spec) + '->', *ops, optimize='greedy')


cols, nets = [], []
trial = np.random.default_rng(7)
while len(nets) < 19:
    ms = tuple(int(v) for v in trial.integers(0, 15, size=4))
    col = [int(network(ms, z, w)) for z, w in pts]
    if sp.Matrix([c for c in cols] + [col]).T.rank() == len(cols) + 1:
        cols.append(col)
        nets.append(ms)
E = sp.Matrix(cols).T
check('FI-2', E.shape == (NPTS, 19) and E.rank() == 19,
      f'{len(nets)} contraction invariants, evaluation matrix {E.shape[0]}x{E.shape[1]} of exact rank {E.rank()}')

vals = [sides(z, w) for z, w in pts]
check('FI-3', all(v[2] == v[3] for v in vals), f'||A||^2 = F - (3/4)NW_1 exactly at all {NPTS} points')
check('FI-4', all(v[0] == v[1] for v in vals),
      f'(FG) exact at all {NPTS} points; sample lhs = {vals[0][0]}, rhs = {vals[0][1]}')
bad1 = sum(1 for z, w in pts[:4] if (lambda v: v[0] != v[1])(sides(z, w, coef=Fraction(1))))
bad2 = sum(1 for z, w in pts[:4] if (lambda v: v[0] != v[1])(sides(z, w, lam_on=False)))
check('FI-5', bad1 == 4 and bad2 == 4, f'coefficient 1 fails at {bad1}/4 points, lambda dropped fails at {bad2}/4')

print('ALL PASS' if all(res) else 'FAILURES')
raise SystemExit(0 if all(res) else 1)
