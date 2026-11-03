#!/usr/bin/env python3
"""NSL: the closest-point normal slice of supplement Section sec:fqe-global-overlap.

Proved in: global_overlap_certificate.tex, paragraph "Coverage by the normal slice".  A magic
matrix is Z = sqrt(3) X + i Y (32 real coordinates) and Z_* = diag(sqrt3, i, i, i)/sqrt6.  The
real height sqrt6 Re<Z_*, Z> = 3 X00 + sum_j Y_jj is maximised over phase and O(4) x O(4) at a
closest representative.  Exact sympy linear algebra checks:
  NSL-1  the 13 first-order conditions (6 left, 6 right generators, phase) have rank 10, and are
        equivalent to Y00 = sum X_jj, X_j0 = Y_0j/3, Y_j0 = 3 X_0j, Y_3 symmetric;
  NSL-2  simultaneous lower rotations fix Z_*, preserve these conditions and act on
        M = Sym(X_3) + 2 Y_3 by conjugation, so a residual rotation makes M diagonal (3 conditions);
  NSL-3  the 13 linear conditions cut out a 19-dimensional space, equal to the span of the
        19-coordinate slice of the paper (orbit_certificate_polynomials.py).
The maximisation itself, and positivity of the closest-point constraints, are analytic.
"""
import sympy as s
from orbit_certificate_polynomials import X as XP, Y as YP, NAMES

x = s.Matrix(4, 4, lambda i, j: s.Symbol(f'x{i}{j}', real=True))
y = s.Matrix(4, 4, lambda i, j: s.Symbol(f'y{i}{j}', real=True))
V = list(x) + list(y)
Z = s.sqrt(3)*x + s.I*y
Zs = s.diag(s.sqrt(3), s.I, s.I, s.I)/s.sqrt(6)


def height(W):
    return s.expand(s.sqrt(6)*s.re((Zs.H*W).trace()))


assert s.expand(height(Z) - (3*x[0, 0] + sum(y[j, j] for j in range(1, 4)))) == 0


def gen(i, j):
    G = s.zeros(4, 4)
    G[i, j], G[j, i] = 1, -1
    return G


GENS = [gen(i, j) for i in range(4) for j in range(i+1, 4)]
conds = [height(G*Z) for G in GENS] + [height(-Z*G) for G in GENS] + [height(s.I*Z)]
A = s.Matrix([[s.diff(c, v) for v in V] for c in conds])
assert A.rank() == 10
stated = ([y[0, 0] - sum(x[j, j] for j in range(1, 4))]
          + [x[j, 0] - y[0, j]/3 for j in range(1, 4)]
          + [y[j, 0] - 3*x[0, j] for j in range(1, 4)]
          + [y[j, k] - y[k, j] for j in range(1, 4) for k in range(j+1, 4)])
S = s.Matrix([[s.diff(c, v) for v in V] for c in stated])
assert S.rank() == 10 and A.col_join(S).rank() == 10
print('NSL-1 PASS: the 13 first-order conditions have rank 10 and equal the ten stated relations')

# NSL-2: infinitesimal simultaneous lower rotations.
for (j, k) in [(1, 2), (1, 3), (2, 3)]:
    G = gen(j, k)
    assert s.simplify(G*Zs - Zs*G) == s.zeros(4, 4)
    dx, dy = G*x - x*G, G*y - y*G           # derivative of O Z O^T at O = I
    sub = dict(zip(V, list(dx) + list(dy)))
    # the stationary space is invariant: each stated relation maps into the span of the relations
    img = s.Matrix([[s.diff(c.subs(sub, simultaneous=True), v) for v in V] for c in stated])
    assert S.col_join(img).rank() == 10
    M = (x[1:, 1:] + x[1:, 1:].T)/2 + 2*y[1:, 1:]
    dM = (dx[1:, 1:] + dx[1:, 1:].T)/2 + 2*dy[1:, 1:]
    g = G[1:, 1:]
    Ysym = {y[a, b]: y[b, a] for a in range(1, 4) for b in range(a+1, 4)}
    assert s.expand((dM - (g*M - M*g)).subs(Ysym)) == s.zeros(3, 3)
print('NSL-2 PASS: lower rotations fix Z_*, preserve stationarity and conjugate Sym(X_3)+2Y_3')

gauge = [(x[j, k] + x[k, j])/2 + 2*y[j, k] for j in range(1, 4) for k in range(j+1, 4)]
E = S.col_join(s.Matrix([[s.diff(c, v) for v in V] for c in gauge]))
assert E.rank() == 13 and 32 - E.rank() == 19
coords = s.symbols(NAMES, real=True)
conv = lambda p: s.sympify(p.as_expr(*coords)) if hasattr(p, 'as_expr') else s.sympify(p)
slice_vals = [conv(XP[i][j]) for i in range(4) for j in range(4)] + \
             [conv(YP[i][j]) for i in range(4) for j in range(4)]
J = s.Matrix([[s.diff(e, c) for c in coords] for e in slice_vals])
assert J.rank() == 19 and (E*J).is_zero_matrix
print('NSL-3 PASS: the stationary and gauge conditions cut out exactly the 19-dimensional slice span')
