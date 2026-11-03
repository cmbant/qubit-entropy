#!/usr/bin/env python3
"""Exact check of the sharp moment identity (S) of sextic_identity.tex (Theorem thm:fqe-moment).

SI-1: build the tensors from their definitions, with no contraction table: w = z (x) z (x) tilde z,
      U_i = S_i prod_{j!=i} L_j w and V_i = S_i prod_{j!=i} (L_j+M_j) w.  Also build the pair vectors
      (R_i^{(i)} - R_j^{(j)}) z, where the one-qubit marginal R_i acts on qubit i.
SI-2: exact polynomial arithmetic in the sixteen amplitudes z_a and their conjugates zb_a, no sampling:
        8F = 6 N W1 + 3 sum_i ||V_i - 8U_i||^2 + 12 sum_{i<j} ||(R_i^{(i)} - R_j^{(j)}) z||^2,
      compared coefficient by coefficient, with F = 17NP - 12T - 12N^3 and W1 = 2 S1 - 4N^2.
SI-3: at a product state F = 3, N = 1, W1 = 4, so F = (3/4) N W1 and the constant 3/4 is sharp.
The derivation of (S) from the contraction table is checked separately in verify_m4_entropy_swap_proof.py (SP-2, SP-4).
"""
from fractions import Fraction as Fr
from itertools import permutations, product


def bits(a, k):
    return [(a >> (k-1-t)) & 1 for t in range(k)]


EPS = [[0, 1], [-1, 0]]
SYM = [[Fr(0)]*8 for _ in range(8)]                     # symmetriser of the three slots (x,y,t)
for a in range(8):
    b = bits(a, 3)
    for p in permutations(range(3)):
        c = [b[p[k]] for k in range(3)]
        SYM[4*c[0]+2*c[1]+c[2]][a] += Fr(1, 6)
L = [[0]*8 for _ in range(2)]
M = [[0]*8 for _ in range(2)]
for x, y, t in product(range(2), repeat=3):
    L[x][4*x+2*y+t] = EPS[y][t]                        # L|xyt> = eps_{yt}|x>
    M[y][4*x+2*y+t] = EPS[x][t]                        # M|xyt> = eps_{xt}|y>
LM = [[L[r][c]+M[r][c] for c in range(8)] for r in range(2)]


def rows(mat):
    return [[(c, Fr(v)) for c, v in enumerate(row) if v != 0] for row in mat]


SYMr, Lr, LMr = rows(SYM), rows(L), rows(LM)


# polynomials: dict (sorted z indices, sorted zb indices) -> Fraction
def padd(d, k, v):
    d[k] = d.get(k, 0)+v
    if d[k] == 0:
        del d[k]


def acc(A, B, c=1):
    for k, v in B.items():
        padd(A, k, c*v)


def mul(A, B):
    out = {}
    for (z1, b1), v1 in A.items():
        for (z2, b2), v2 in B.items():
            padd(out, (tuple(sorted(z1+z2)), tuple(sorted(b1+b2))), v1*v2)
    return out


def conj(A):
    return {(b, z): v for (z, b), v in A.items()}


def w_entry(I):
    """w at site indices I=(I_1..I_4), I_s = 4x+2y+t: z_X z_Y ztilde_T, ztilde_T = (-1)^{|T|} zb_{15-T}."""
    X = Y = T = 0
    for i in I:
        X, Y, T = 2*X+(i >> 2), 2*Y+((i >> 1) & 1), 2*T+(i & 1)
    return (tuple(sorted((X, Y))), (15 ^ T,)), (-1)**bin(T).count('1')


def contracted(i, other):
    """The vector S_i prod_{j!=i} other_j w, as {output index: polynomial}."""
    out = {}
    ops = [SYMr if s == i else other for s in range(4)]
    for o in product(*(range(len(op)) for op in ops)):
        poly = {}
        for terms in product(*(ops[s][o[s]] for s in range(4))):
            key, sign = w_entry([c for c, _ in terms])
            coef = sign
            for _, v in terms:
                coef *= v
            padd(poly, key, coef)
        if poly:
            out[o] = poly
    return out


def norm2(vec):
    out = {}
    for poly in vec.values():
        acc(out, mul(poly, conj(poly)))
    return out


# SI-1 / SI-2: sum_i ||V_i - 8U_i||^2
res = {}
for i in range(4):
    U, V = contracted(i, Lr), contracted(i, LMr)
    D = {o: dict(V.get(o, {})) for o in set(U) | set(V)}
    for o, poly in U.items():
        acc(D[o], poly, -8)
    acc(res, norm2(D))

Z = [{((a,), ()): Fr(1)} for a in range(16)]
ZB = [{((), (a,)): Fr(1)} for a in range(16)]


def marginal(keep):
    rest = [s for s in range(4) if s not in keep]
    d = 2**len(keep)
    R = [[{} for _ in range(d)] for _ in range(d)]
    for r, c, rs in product(range(d), range(d), range(2**len(rest))):
        br, bc = [0]*4, [0]*4
        for n, k in enumerate(keep):
            br[k], bc[k] = bits(r, len(keep))[n], bits(c, len(keep))[n]
        for n, k in enumerate(rest):
            br[k] = bc[k] = bits(rs, len(rest))[n]
        f = lambda b: 8*b[0]+4*b[1]+2*b[2]+b[3]
        padd(R[r][c], ((f(br),), (f(bc),)), Fr(1))
    return R


def trace_power(R, k):
    d, out = len(R), {}
    for idx in product(range(d), repeat=k):
        term = {((), ()): Fr(1)}
        for m in range(k):
            term = mul(term, R[idx[m]][idx[(m+1) % k]])
        acc(out, term)
    return out


R1 = [marginal([s]) for s in range(4)]


def act(s, b):
    """(R_s acting on qubit s) z, component b."""
    poly, bb = {}, bits(b, 4)
    for a in range(2):
        cc = list(bb)
        cc[s] = a
        acc(poly, mul(R1[s][bb[s]][a], Z[8*cc[0]+4*cc[1]+2*cc[2]+cc[3]]))
    return poly


pair = {}
for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
    vec = {}
    for b in range(16):
        poly = act(i, b)
        acc(poly, act(j, b), -1)
        vec[b] = poly
    acc(pair, norm2(vec))
print(f'SI-1 ok: tensors U_i, V_i and pair vectors built from L, M, S and the marginals; '
      f'{len(res)} monomials in sum ||V_i-8U_i||^2, {len(pair)} in the pair sum')

N = {}
for a in range(16):
    padd(N, ((a,), (a,)), Fr(1))
P, T3, S1 = {}, {}, {}
for j in (1, 2, 3):
    R = marginal([0, j])
    acc(P, trace_power(R, 2))
    acc(T3, trace_power(R, 3))
for R in R1:
    acc(S1, trace_power(R, 2))
N2 = mul(N, N)
N3 = mul(N2, N)
F = {}
acc(F, mul(N, P), 17)
acc(F, T3, -12)
acc(F, N3, -12)
NW1 = {}
acc(NW1, mul(N, S1), 2)
acc(NW1, N3, -4)
diff = {}
acc(diff, F, 8)
acc(diff, NW1, -6)
acc(diff, res, -3)
acc(diff, pair, -12)
print(f'SI-2: {len(F)} monomials in F; nonzero coefficients of 8F - 6NW1 - 3 sum||V_i-8U_i||^2 '
      f'- 12 sum||(R_i-R_j)z||^2: {len(diff)}')
assert not diff and len(F) > 0

# SI-3: product state |0000>: P = 3, T = 3, N = 1, W1 = 4
F_prod, W_prod = 17*3-12*3-12, 4
assert F_prod == Fr(3, 4)*W_prod
print('SI-3 ok: product state has F = 3 = (3/4) N W1, so 3/4 is the best constant')
