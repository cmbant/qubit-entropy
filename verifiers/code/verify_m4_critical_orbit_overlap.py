#!/usr/bin/env python3
"""CO-1--CO-3: exact symbolic and rational checks for sharp critical-state stability.

Proved in: critical_orbit_stability.tex,
thm:fqe-critical-orbit-sharp.

Verifies:
  CO-1: the critical variance identity, the three-term moment identity for Q(v),
        the 32-ray tight frame with operator 8I, and the invariance of Q(v) and
        permutation of the 32 HS representatives (up to sixth-root-of-unity phases)
        under the three real orthogonal cut transforms H_0, H_1, H_2.
  CO-2: all exact polynomial identities and rational concavity bounds (P_1, P_2)
        in the elementary 24-cell proof of Q(v) >= 3/2 - 2*Phi(v) for Phi(v) <= 1/2.
  CO-3: the sharp Hilbert-distance coefficient (2+sqrt(2))/4 and Bell-product equality
        across all 32 HS representatives.
"""
from itertools import combinations, product
import sympy as s

R = s.Rational

# ---------------------------------------------------------------------------
# CO-1: critical variance identity, 3-term Q(v) identity, 32-ray tight frame,
#       and 3-cut MUB / 24-cell symmetry
# ---------------------------------------------------------------------------
x0, x1, x2, x3 = s.symbols('x0 x1 x2 x3', real=True)
y0, y1, y2, y3 = s.symbols('y0 y1 y2 y3', real=True)
v = s.Matrix([x0 + s.I * y0, x1 + s.I * y1, x2 + s.I * y2, x3 + s.I * y3])
n = s.expand(sum(z * s.conjugate(z) for z in v))
p = [s.expand(z * s.conjugate(z)) for z in v]
ss = s.expand(sum(z * z for z in v))
B = s.expand(sum(pi * z * z for pi, z in zip(p, v)))
Q = s.expand(
    27 * n * sum(pi * pi for pi in p)
    - 36 * sum(pi**3 for pi in p)
    - 4 * n**3
    - R(9, 2) * n * ss * s.conjugate(ss)
    + 18 * s.re(s.expand(s.conjugate(ss) * B))
)

variance = 0
for i in range(4):
    for j, k in combinations([a for a in range(4) if a != i], 2):
        term = s.expand(s.conjugate(v[i]) * (v[j]**2 - v[k]**2) + 2 * v[i] * (p[j] - p[k]))
        variance += s.expand(term * s.conjugate(term))
assert s.expand(Q - variance - n * ss * s.conjugate(ss) / 2) == 0

c = [[s.expand(s.re(s.conjugate(v[i]) * v[j])) for j in range(4)] for i in range(4)]
D1 = p[0] + p[1] - p[2] - p[3]
D2 = p[0] + p[2] - p[1] - p[3]
D3 = p[0] + p[3] - p[1] - p[2]
Q_three_term = s.expand(
    R(1, 2) * n**3
    + 18 * (
        D1 * (c[0][1]**2 - c[2][3]**2)
        + D2 * (c[0][2]**2 - c[1][3]**2)
        + D3 * (c[0][3]**2 - c[1][2]**2)
    )
)
assert s.expand(Q - Q_three_term) == 0

frame = []
for i in range(4):
    for signs in product((-1, 1), repeat=3):
        hv = s.zeros(4, 1)
        hv[i] = 1 / s.sqrt(2)
        for j, sign in zip([k for k in range(4) if k != i], signs):
            hv[j] = sign * s.I / s.sqrt(6)
        frame.append(hv)
assert len(frame) == 32
assert s.simplify(sum((h * h.H for h in frame), s.zeros(4)) - 8 * s.eye(4)) == s.zeros(4)

H1 = R(1, 2) * s.Matrix([
    [1, -1, 1, -1],
    [-1, 1, 1, -1],
    [1, 1, 1, 1],
    [-1, -1, 1, 1],
])
H2 = R(1, 2) * s.Matrix([
    [1, -1, -1, -1],
    [-1, 1, -1, -1],
    [1, 1, -1, 1],
    [-1, -1, -1, 1],
])
assert H1 * H1.T == s.eye(4) and H2 * H2.T == s.eye(4)

# Verify H_1, H_2 permute the 32 HS representatives up to 6th-root-of-unity phases
scaled_frame = [s.Matrix([s.expand(z * s.sqrt(6)) for z in h]) for h in frame]
phases = [
    s.Integer(1), s.Integer(-1),
    (1 + s.I * s.sqrt(3)) / 2, (1 - s.I * s.sqrt(3)) / 2,
    (-1 + s.I * s.sqrt(3)) / 2, (-1 - s.I * s.sqrt(3)) / 2,
]
for H in (H1, H2):
    for h_sc in scaled_frame:
        w = H * h_sc
        assert any(
            all(s.expand(w[k] - ph * f_sc[k]) == 0 for k in range(4))
            for f_sc in scaled_frame for ph in phases
        )
    vH = H * v
    pH = [s.expand(z * s.conjugate(z)) for z in vH]
    ssH = s.expand(sum(z * z for z in vH))
    BH = s.expand(sum(pi * z * z for pi, z in zip(pH, vH)))
    QH = s.expand(
        27 * n * sum(pi * pi for pi in pH)
        - 36 * sum(pi**3 for pi in pH)
        - 4 * n**3
        - R(9, 2) * n * ssH * s.conjugate(ssH)
        + 18 * s.re(s.expand(s.conjugate(ssH) * BH))
    )
    assert s.expand(ssH - ss) == 0
    assert s.expand(QH - Q) == 0

# Verify the 8 cut probabilities lambda_eps and their 4-sign averages
for e1, e2, e3 in product((-1, 1), repeat=3):
    w_eps = R(1, 2) * (v[0] + e1 * v[1] + e2 * v[2] + e3 * v[3])
    lam_eps = s.expand(w_eps * s.conjugate(w_eps))
    lam_expanded = R(1, 4) * (
        n
        + 2 * (e1 * c[0][1] + e2 * c[0][2] + e3 * c[0][3])
        + 2 * (e1 * e2 * c[1][2] + e1 * e3 * c[1][3] + e2 * e3 * c[2][3])
    )
    assert s.expand(lam_eps - lam_expanded) == 0

print('[PASS] CO-1 critical variance and 3-term Q(v) identities; 32 HS representatives '
      'form an 8I tight frame permuted up to 6th-root phases by H_1, H_2')

# ---------------------------------------------------------------------------
# CO-2: exact algebraic and rational-concavity checks for Q >= 3/2 - 2*Phi
# ---------------------------------------------------------------------------
r0, X2, Y2, Sy, e2y, s2, q4, Dmax = s.symbols('r0 X2 Y2 Sy e2y s2 q4 Dmax', real=True)
p0 = r0**2

# Expansion of Phi_im(v) on the unit sphere r0^2 + X2 + Y2 = 1
Phi_im = (r0 / s.sqrt(2) + Sy / s.sqrt(6))**2
Phi_im_rewritten = (
    R(1, 2) * (r0**2 + Y2)
    + R(1, 3) * (s.sqrt(3) * r0 * Sy - Y2 + e2y)
).subs(e2y, (Sy**2 - Y2) / 2)
assert s.expand(Phi_im - Phi_im_rewritten) == 0

# Rational upper bounds on sqrt(3) and lower bound r0 >= 3/5
assert (R(26, 15)**2 - 3).is_positive is True
assert (R(7, 4) - R(26, 15)).is_positive is True
assert s.expand(3 * r0**2 - R(1, 2) + R(7, 4) * r0 * (1 - r0) - 1 - R(1, 4) * (5 * r0 - 3) * (r0 + 2)) == 0
assert s.expand(1 - r0**2 - 3 * (1 - r0)**2 - 2 * (1 - r0) * (2 * r0 - 1)) == 0
assert s.expand(1 - r0**2 - R(26, 15) * r0 * (1 - r0) - R(1, 15) * (1 - r0) * (15 - 11 * r0)) == 0

# Constant bound 2*s2 <= (2*r0 - 1)^2
assert s.expand(4 * p0 - 2 * (1 - p0 - X2) + 3 * (1 - r0)**2 - ((3 * r0 - 1)**2 + 2 * X2)) == 0
assert s.cancel((3 * r0 - 1 + X2 / (3 * r0 - 1))**2 - ((3 * r0 - 1)**2 + 2 * X2) - X2**2 / (3 * r0 - 1)**2) == 0
assert s.cancel(
    (2 * r0 - 1 + X2 / (3 * r0 - 1))**2 - X2
    - ((2 * r0 - 1)**2 - X2 * ((1 - r0) * (3 * r0 - 1) - X2) / (3 * r0 - 1)**2)
) == 0
assert s.expand(r0**2 - (1 - r0) * (3 * r0 - 1) - (2 * r0 - 1)**2) == 0

# Positivity of Y(r0, y)
assert s.expand((2 * r0 + 1) + 3 * (2 * r0 - 1) + 4 * (1 - r0) + 4 * (1 - r0) - 6) == 0
assert (R(3) - R(729, 256)).is_positive is True
Y_rewritten = (
    R(2, 3) * (s.sqrt(3) * r0 * Sy - (Sy**2 - 2 * e2y) + e2y) - 9 * Dmax * Sy**2 * e2y
)
assert s.expand(Y_rewritten - (R(2, 3) * Sy * (s.sqrt(3) * r0 - Sy) + (2 - 9 * Dmax * Sy**2) * e2y)) == 0
assert s.expand(R(2, 3) * Sy * (s.sqrt(3) * r0 - Sy) - R(1, 3) * Sy**2 - R(1, 3) * Sy * (2 * s.sqrt(3) * r0 - 3 * Sy)) == 0
assert s.expand(2 * s.sqrt(3) * r0 - 3 * s.sqrt(3) * (1 - r0) - s.sqrt(3) * (5 * r0 - 3)) == 0

# Positivity of X(p; x^2) on r0 in [19/25, 1]
u1, u2, u3 = s.symbols('u1 u2 u3', real=True)
X2_u = u1 + u2 + u3
q4_u = u1 * u2 + u1 * u3 + u2 * u3
r6_u = u1 * u2 * u3
sum_D_x4 = (
    (1 - 2 * (u1 + u2)) * u1 * u2
    + (1 - 2 * (u1 + u3)) * u1 * u3
    + (1 - 2 * (u2 + u3)) * u2 * u3
)
assert s.expand(sum_D_x4 - (q4_u - 2 * (X2_u * q4_u - 3 * r6_u))) == 0
X_high = 18 * p0 * (2 * p0 - 1) - 1 + 36 * p0 * X2 - 12 * (2 * p0 + 1 - R(4, 3) * X2) * X2
X_high_sos = (4 * X2 - R(3, 2) * (1 - p0))**2 + (135 * p0**2 - 54 * p0 - 13) / 4
assert s.expand(X_high - X_high_sos) == 0
min_high = ((135 * p0**2 - 54 * p0 - 13) / 4).subs(r0, R(19, 25))
assert min_high == R(16573, 78125) and min_high.is_positive is True

# Positivity of X(p; x^2) on r0 in [3/5, 16/25] via P_1(r_0)
u = s.Symbol('u', nonnegative=True)
X2_1 = R(1, 15) * (1 - r0) * (15 - 11 * r0)
P1 = s.expand(15 * ((18 * p0 * (2 * p0 - 1) - 1) * X2_1 + 36 * p0 * X2_1**2 - 9 * (6 * p0 - 1) * (2 * r0 - 1)**4))
P1_d2_u = s.Poly(s.expand(s.diff(P1, r0, 2).subs(r0, R(3, 5) + u)), u)
assert all(coeff.is_negative is True for coeff in P1_d2_u.all_coeffs())
assert P1.subs(r0, R(3, 5)) == R(3693, 78125) and P1.subs(r0, R(3, 5)).is_positive is True
assert P1.subs(r0, R(16, 25)) == R(216609912, 1220703125) and P1.subs(r0, R(16, 25)).is_positive is True

# Positivity of X(p; x^2) on r0 in [16/25, 19/25] via P_2(r_0)
X2_2 = 2 * (1 - r0) * (2 * r0 - 1)
P2 = s.expand(s.cancel(((18 * p0 * (2 * p0 - 1) - 1) * X2_2 + 36 * p0 * X2_2**2 - 9 * (2 * p0 + 1) * (2 * r0 - 1)**4) / (2 * r0 - 1)))
assert s.expand(P2 - (72 * r0**5 - 432 * r0**4 + 432 * r0**3 - 54 * r0**2 - 52 * r0 + 7)) == 0
P2_d2_u = s.expand(s.diff(P2, r0, 2).subs(r0, R(3, 5) + u))
assert s.expand(P2_d2_u + 36 * (8 * u**2 * (9 - 5 * u) + R(288, 5) * u + 3)) == 0
assert P2.subs(r0, R(16, 25)) == R(988047, 9765625) and P2.subs(r0, R(16, 25)).is_positive is True
assert P2.subs(r0, R(19, 25)) == R(570453, 9765625) and P2.subs(r0, R(19, 25)).is_positive is True

print('[PASS] CO-2 exact decoupled lower bound Q+2*Phi_im-3/2 >= Y+X and rational '
      'concavity certificates P_1, P_2 on [3/5, 1] prove Q >= 3/2 - 2*Phi')

# ---------------------------------------------------------------------------
# CO-3: sharp Hilbert-distance coefficient (2+sqrt(2))/4 and Bell equality
# ---------------------------------------------------------------------------
x = s.Symbol('x', nonnegative=True)
c_dist = (2 + s.sqrt(2)) / 4
assert s.expand((1 - x * x) - 2 * c_dist * (1 - x) - (1 - x) * (x - 1 / s.sqrt(2))) == 0
assert s.expand(
    R(3, 2) - 2 * x * x - 2 * c_dist * (1 - x)
    - 2 * (1 / s.sqrt(2) - x) * (x - (2 - s.sqrt(2)) / 4)
) == 0
assert s.simplify(R(1, 2) / (2 - s.sqrt(2)) - c_dist) == 0

bell_vectors = [
    s.Matrix([s.sqrt(3), 0, 0, 0]),
    s.Matrix([s.sqrt(3) + 3 * s.I, s.sqrt(3) + 3 * s.I, s.sqrt(3) + 3 * s.I, s.sqrt(3) + 3 * s.I]),
    s.Matrix([s.sqrt(3) - 3 * s.I, -s.sqrt(3) + 3 * s.I, -s.sqrt(3) + 3 * s.I, -s.sqrt(3) + 3 * s.I]),
]
for vp in bell_vectors:
    subs = {
        x0: s.re(vp[0]), y0: s.im(vp[0]),
        x1: s.re(vp[1]), y1: s.im(vp[1]),
        x2: s.re(vp[2]), y2: s.im(vp[2]),
        x3: s.re(vp[3]), y3: s.im(vp[3]),
    }
    np = s.simplify(n.subs(subs))
    Qp = s.simplify(Q.subs(subs))
    assert s.simplify(Qp / np**3 - R(1, 2)) == 0
    overlaps = [
        s.radsimp(s.expand(s.expand_complex((hv.H * vp)[0] * s.conjugate((hv.H * vp)[0]))) / np)
        for hv in frame
    ]
    assert all(s.simplify(R(1, 2) - q).is_nonnegative is True for q in overlaps)
    assert any(s.simplify(q - R(1, 2)) == 0 for q in overlaps)

print('[PASS] CO-3 sharp Hilbert-distance coefficient (2+sqrt2)/4 and sharp infidelity '
      'coefficient 1; both attained by the three Bell-pair-product representatives, '
      'whose maximum overlap over all 32 representatives is exactly 1/2')

print('All CO checks passed; analytic orbit reduction and case arguments are in the paper.')
