"""Exact polynomial ingredients for the closest-HS orbit certificate.

All coefficients are rational; a magic matrix is represented as sqrt(3) X+iY.
The positive constraints and normal-slice coverage are proved in the paper.
"""
from itertools import combinations
import sympy as s
from sympy.polys.rings import ring

NAMES = 't u1 u2 u3 a1 a2 a3 e12 e13 e23 b12 b13 b23 p1 p2 p3 q1 q2 q3'.split()
PR, *XS = ring(NAMES, s.QQ)
ZERO = PR.zero
PAIRS = list(combinations(range(1, 4), 2))
CHARS = [0]*7+[3, 5, 6]*2+[1, 2, 4]*2


def transpose(A):
    return [list(row) for row in zip(*A)]


def matmul(A, B):
    return [[sum((A[i][k]*B[k][j] for k in range(len(B))), ZERO)
             for j in range(len(B[0]))] for i in range(len(A))]


def add(A, B, a=1, b=1):
    return [[a*x+b*y for x, y in zip(r, q)] for r, q in zip(A, B)]


def norm(A):
    return sum((x*x for r in A for x in r), ZERO)


def trace(A):
    return sum((A[i][i] for i in range(len(A))), ZERO)


def defects(X, Y):
    A = add(matmul(X, transpose(X)), matmul(Y, transpose(Y)), 3, 1)
    B = add(matmul(Y, transpose(X)), matmul(X, transpose(Y)), 1, -1)
    C = add(matmul(transpose(X), X), matmul(transpose(Y), Y), 3, -1)
    D = add(matmul(transpose(X), Y), matmul(transpose(Y), X))
    n = 3*norm(X)+norm(Y)
    sr, si = trace(C), 2*sum((x*y for r, q in zip(X, Y) for x, y in zip(r, q)), ZERO)
    Nr = add(matmul(X, C), matmul(Y, D), 1, -1)
    Ni = add(matmul(X, D), matmul(Y, C), 3, 1)
    Nr = add(add(Nr, X, 1, -sr/2), Y, 1, si/2)
    Ni = add(add(Ni, Y, 1, -sr/2), X, 1, -3*si/2)
    p2 = trace(matmul(A, A))-3*trace(matmul(B, B))
    p3 = trace(matmul(matmul(A, A), A))-9*trace(matmul(matmul(A, B), B))
    Q = 27*n*p2-18*p3-18*(3*norm(Nr)+norm(Ni))-4*n**3
    W = 12*(norm(B)+norm(add(matmul(transpose(X), Y), matmul(transpose(Y), X), 1, -1)))
    tau = sr*sr+3*si*si
    return n, Q, Q+s.QQ(5, 2)*n*(W+tau), W, tau


X, Y = [[[ZERO for j in range(4)] for i in range(4)] for _ in range(2)]
t, U, A = XS[0], XS[1:4], XS[4:7]
X[0][0], Y[0][0] = t, sum(U, ZERO)
for j in range(1, 4):
    X[j][j], Y[j][j] = U[j-1], t+A[j-1]
    X[0][j], X[j][0] = XS[12+j], XS[15+j]/3
    Y[0][j], Y[j][0] = XS[15+j], 3*XS[12+j]
for k, (i, j) in enumerate(PAIRS):
    X[i][j], X[j][i] = XS[7+k]+XS[10+k], XS[7+k]-XS[10+k]
    Y[i][j] = Y[j][i] = -XS[7+k]/2
n, Q, R, W, tau = defects(X, Y)
h0 = 6*t+sum(A, ZERO)
e = n-h0*h0/6


def height_hessian():
    z = [[ZERO for j in range(4)] for i in range(4)]
    gens = []
    for j in range(1, 4):
        G = [r[:] for r in z]
        G[0][j], G[j][0] = PR.one, -PR.one
        gens.extend([(G, z, 0), (z, G, 0)])
    for i, j in PAIRS:
        G = [r[:] for r in z]
        G[i][j], G[j][i] = PR.one, -PR.one
        gens.append((G, z, 0))
    gens.append((z, z, 1))  # Phase generator i*sqrt(3).
    def act(g, V):
        L, RR, p = g
        xx, yy = V
        return (add(add(matmul(L, xx), matmul(xx, RR), 1, -1), yy, 1, -p),
                add(add(matmul(L, yy), matmul(yy, RR), 1, -1), xx, 1, 3*p))
    cx, cy = [r[:] for r in z], [r[:] for r in z]
    cx[0][0] = PR.one
    for i in range(1, 4):
        cy[i][i] = PR.one
    H = []
    for g in gens:
        row = []
        gx, gy = act(g, (cx, cy))
        assert 3*sum((a*b for r, q in zip(X, gx) for a, b in zip(r, q)), ZERO)+sum((a*b for r, q in zip(Y, gy) for a, b in zip(r, q)), ZERO) == 0
        for gg in gens:
            v, w = act(g, act(gg, (cx, cy))), act(gg, act(g, (cx, cy)))
            tx, ty = add(v[0], w[0], s.QQ(1, 2), s.QQ(1, 2)), add(v[1], w[1], s.QQ(1, 2), s.QQ(1, 2))
            row.append(-3*sum((a*b for r, q in zip(X, tx) for a, b in zip(r, q)), ZERO)-sum((a*b for r, q in zip(Y, ty) for a, b in zip(r, q)), ZERO))
        H.append(row)
    return H


def lower_frame(sig0, sig, cp, sp):
    # cp=cos(phi), sp=sin(phi)/sqrt(3); hence cp^2+3sp^2=1.
    assert cp*cp+3*sp*sp == 1
    rb, ib = sig0*Y[0][0]+sig*sum((Y[i][i] for i in range(1, 4)), ZERO), -sig0*X[0][0]-sig*sum((X[i][i] for i in range(1, 4)), ZERO)
    L = []
    for i in range(1, 4):
        row = []
        for j in range(1, 4):
            xx, yy = (X[i][j]+X[j][i])/2, Y[i][j]
            val = -cp*(3*xx-sig*yy)-3*sp*(yy+sig*xx)
            if i == j:
                val += h0-cp*rb-3*sp*ib
            row.append(val)
        L.append(row)
    return L
