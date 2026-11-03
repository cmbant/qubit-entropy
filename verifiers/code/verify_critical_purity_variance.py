#!/usr/bin/env python3
"""CV-1--CV-5: exact algebra for prop:fqe-critical-variance.

Proved in: renyi_compatibility.tex. The arbitrary-plane parametrization,
fixed-variance product argument and exhaustive equality inference are analytic.
CV-4 checks the two generic polynomial factorizations used in that argument.
"""

import sympy as s
from itertools import product

L = [
    s.Matrix(x)
    for x in (
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
        [[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],
        [[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    )
]
R = [
    s.Matrix(x)
    for x in (
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
        [[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
    )
]
I = s.eye(4)
for i in range(3):
    assert L[i].T == -L[i] and R[i].T == -R[i]
    for j in range(3):
        assert L[i] * R[j] - R[j] * L[i] == s.zeros(4)
        assert L[i] * L[j] + L[j] * L[i] == (-2 * I if i == j else s.zeros(4))
        assert R[i] * R[j] + R[j] * R[i] == (-2 * I if i == j else s.zeros(4))
K = [[-L[i] * R[j] for j in range(3)] for i in range(3)]
assert all(k.T == k and s.trace(k) == 0 for row in K for k in row)
# A fixed plane checks orientation and sign conventions; the general Hodge
# two-plane argument is analytic, rather than inferred from this example.
B = s.zeros(4)
B[0, 1] = 1
B[1, 0] = -1
star = s.zeros(4)
star[2, 3] = 1
star[3, 2] = -1
assert B + star == -L[0] and B - star == -R[0]
assert (I - (B + star) * (B - star)) / 2 == s.diag(1, 1, 0, 0)
print(
    "CV-1 PASS: quaternion relations, commuting actions and projector sign convention"
)

H = [
    I,
    s.Matrix([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]) / 2,
    s.Matrix([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2,
]
expected = [
    {(0, 0), (1, 1), (2, 2)},
    {(0, 2), (1, 0), (2, 1)},
    {(0, 1), (1, 2), (2, 0)},
]
for h, pairs in zip(H, expected):
    assert h * h.T == I
    diags = {}
    for i, j in product(range(3), repeat=2):
        diagonal = tuple((h * K[i][j] * h.T).diagonal())
        if any(diagonal):
            assert (i, j) in pairs and all(abs(x) == 1 for x in diagonal)
            diags[i, j] = diagonal
    assert set(diags) == pairs
    rows = list(zip(*(diags[key] for key in sorted(diags))))
    assert all(s.prod(row) == 1 for row in rows)
    assert len(set(rows)) == 4
print("CV-2 PASS: all three exact measurement triads and tetrahedral sign rows")

c = s.symbols("c0:3")
u = sum(t * t for t in c)
k = s.prod(c)
probs = [
    (1 + sum(e * t for e, t in zip(eps, c))) / 4
    for eps in product((-1, 1), repeat=3)
    if s.prod(eps) == 1
]
assert s.expand(sum(t * t for t in probs) - (1 + u) / 4) == 0
assert s.expand(sum(t**3 for t in probs) - (1 + 3 * u + 6 * k) / 16) == 0
A0, A1, B0, B1 = s.symbols("A0 A1 B0 B1")
A = [A0, A1, 1 - A0 - A1]
BB = [B0, B1, 1 - B0 - B1]
e2 = lambda v: sum(v[i] * v[j] for i in range(3) for j in range(i + 1, 3))
X, Y = 1 - 3 * e2(A), 1 - 3 * e2(BB)
uv = [sum(A[i] * BB[(i + j) % 3] for i in range(3)) for j in range(3)]
assert s.expand(sum(uv) - 1) == 0
assert s.expand(sum(t * t for t in uv) - (1 + 2 * X * Y) / 3) == 0
kk = s.symbols("k")
p = [(1 + t) / 4 for t in uv]
T = sum((1 + 3 * t + 6 * kk) / 16 for t in uv)
G = s.Rational(3, 4) - T - sum(t * t for t in p)
assert s.expand(G - (1 - X * Y - 27 * kk) / 24) == 0
assert (
    s.expand(12 * G - (5 - 12 * T - 12 * (sum(t * t for t in p) - s.Rational(1, 3))))
    == 0
)
print("CV-3 PASS: spectral moments, cyclic purity identity and variance defect algebra")

# The fixed-e2 product bound and geometric-mean transfer are algebraic.
r = s.symbols("r", nonnegative=True)
triple = [(1 + 2 * r) / 3, (1 - r) / 3, (1 - r) / 3]
assert s.expand(1 - 3 * e2(triple) - r**2) == 0
assert s.expand(27 * s.prod(triple) - (1 - 3 * r**2 + 2 * r**3)) == 0
h = (1 - r) ** 2 * (1 + 2 * r)
discriminant = s.prod((A[i] - A[j]) ** 2 for i in range(3) for j in range(i + 1, 3))
assert s.expand(4 * X**3 - (27 * s.prod(A) - 1 + 3 * X) ** 2 - 27 * discriminant) == 0
xi, zeta = s.symbols("xi zeta")
g = lambda t: (1 - t) ** 2 * (1 + 2 * t)
q = xi * zeta
dd = (xi - zeta) ** 2
ww = (1 - xi**2) * (1 - zeta**2)
positive_remainder = dd * (ww * (6 * q * (q + 2) + 2 * dd) + dd * (1 + 2 * q) ** 2)
assert s.expand(g(xi * zeta) ** 2 - g(xi**2) * g(zeta**2) - positive_remainder) == 0
assert s.expand((1 - h) / 2 - (s.Rational(3, 2) * r**2 - r**3)) == 0
assert (
    s.expand(s.Rational(3, 2) * r**2 - r**3 - r**4 / 2 - r**2 * (1 - r) * (3 + r) / 2)
    == 0
)
print(
    "CV-4 PASS: cubic discriminant, nonnegative geometric-mean remainder and sharp defect"
)

# Equality family, including all intermediate variances.  The second spectrum
# is a tetrahedral vector (sqrt(a*b), sqrt(a*b), b); its square root is kept
# formal until reducing its defining polynomial.
a, b, w = s.symbols("a b w")
first = [s.Rational(1, 2), a / 2, b / 2, b / 2]
second = [(1 + b + 2 * w) / 4, (1 + b - 2 * w) / 4, (1 - b) / 4, (1 - b) / 4]
reduce_w = lambda expr: s.rem(
    s.Poly(s.expand(expr), w), s.Poly(w * w - a * b, w)
).as_expr()
family = [first, second, second]
sub = {a: (1 + 2 * r) / 3, b: (1 - r) / 3}
purity = [s.expand(reduce_w(sum(z * z for z in cut)).subs(sub)) for cut in family]
third = [s.expand(reduce_w(sum(z**3 for z in cut)).subs(sub)) for cut in family]
assert s.expand(sum(purity) - 1) == 0
assert s.expand(sum((p - s.Rational(1, 3)) ** 2 for p in purity) - r**4 / 24) == 0
assert s.expand(5 - 12 * sum(third) - (s.Rational(3, 2) * r**2 - r**3)) == 0
for cut in family:
    assert s.expand(sum(cut).subs(sub) - 1) == 0
hs = sorted([s.Rational(1, 2)] + [s.Rational(1, 6)] * 3)
assert all(
    sorted(
        [
            z.subs({a: s.Rational(1, 3), b: s.Rational(1, 3), w: s.Rational(1, 3)})
            for z in cut
        ]
    )
    == hs
    for cut in family
)
assert sorted([z.subs({a: 1, b: 0}) for z in first]) == [
    0,
    0,
    s.Rational(1, 2),
    s.Rational(1, 2),
]
assert all(z.subs({a: 1, b: 0, w: 0}) == s.Rational(1, 4) for z in second)
# Every claimed rank-two projector is realized, not inferred from examples.
aa = s.symbols("aa0:3")
bb = s.symbols("bb0:3")
complex_left = sum((aa[i] * L[i] for i in range(3)), s.zeros(4))
complex_right = sum((bb[i] * R[i] for i in range(3)), s.zeros(4))
C = -complex_left * complex_right
assert (C * C - s.eye(4) * sum(z * z for z in aa) * sum(z * z for z in bb)).applyfunc(
    s.expand
) == s.zeros(4)
assert s.trace(C) == 0
print(
    "CV-5 PASS: full equality-family moments, HS/cluster endpoints and projector attainability"
)
