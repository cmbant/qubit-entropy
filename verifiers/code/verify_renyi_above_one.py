#!/usr/bin/env python3
"""RA-1--RA-5: exact checks for the analytic Renyi bound and purity corollary.

Homes: renyi_above_one.tex (near-one bound), renyi_scalar_slack.tex (RA),
and renyi_hs_cluster_crossing.tex (RA-4).
RA-1 checks the fourth-derivative identity, three endpoint numerators,
fifteen printed coarse Taylor bounds and their tails; no parameter grid.
The concavity, derivative-shape and Hermite arguments are in the paper.
All enclosures use directed integer arithmetic; no floating point is used.
"""

from functools import lru_cache
from math import factorial
from pathlib import Path
import sympy as sp
from rational_intervals import Box, Q, SCALE, logarithm, log_interval, power

P, R = Q(1, 6), Q(1, 2)


@lru_cache(None)
def f(q, x):
    if x == 0:
        return Box(0)
    return -x * logarithm(x) if q == 1 else (power(x, q) - x) / (1 - q)


def fp(q, x):
    return -logarithm(x) - 1 if q == 1 else (q * power(x, q - 1) - 1) / (1 - q)


def fpp(q, x):
    return Box(-1 / x) if q == 1 else -q * power(x, q - 2)


@lru_cache(None)
def coefficients(q):
    slope = (f(q, R) - f(q, P)) / (R - P)
    a = (fp(q, P) + fp(q, R) - 2 * slope) / (R - P) ** 2
    b = (fp(q, R) - fp(q, P)) / (2 * (R - P)) - Q(3, 2) * a * (P + R)
    c = fp(q, P) - 3 * a * P**2 - 2 * b * P
    d = f(q, P) - a * P**3 - b * P**2 - c * P
    return a, b, c, d, slope


qq, x = sp.symbols("qq x", positive=True)
mm, bb = sp.symbols("mm bb", real=True)
ff = (x - x**qq) / (qq - 1)
gg = ff - mm * x - bb
aa = 2 * (qq + 1) / (3 - qq) * (1 / (qq - 1) - mm)
dd = 4 * (2 * qq - 1) * (3 - 2 * qq) / ((qq - 1) * (3 - qq) * (2 - qq))
phi = 2 * bb + aa * x - dd * x**qq
factor4 = qq * (2 - qq) * (3 - qq) * x ** (qq - 4)
assert sp.simplify(sp.diff(gg**2, x, 4) / factor4 - phi) == 0

r0, E, F0, L2, L3 = sp.symbols("r0 E F0 L2 L3")
L6 = L2 + L3
delta = r0 * (2 - r0) * (1 - r0)
N0 = 6 * (1 - r0) * E - 6 + 5 * r0 + 7 * r0**2
Np = (2 + r0) * (1 - r0) * (3 * E - 1) - 4 * (1 + r0) * (1 - 4 * r0**2)
N1 = (2 + r0) * (1 - r0) * (3 * F0 - E) - 4 * (1 + r0) * (1 - 4 * r0**2)
chain = lambda n: sp.diff(n, r0) + sp.diff(n, E) * L3 * E
J0 = sp.expand((chain(N0) - L6 * N0) * delta - N0 * sp.diff(delta, r0))
Jp = sp.expand((chain(Np) - L6 * Np) * delta - Np * sp.diff(delta, r0))
J1 = sp.expand(
    (sp.diff(N1, r0) - sp.diff(N1, E) * L6 * E - sp.diff(N1, F0) * L2 * F0) * delta
    - N1 * sp.diff(delta, r0)
)

# Connect the three endpoint expressions to the actual secant coefficients.
ep, es = sp.symbols("ep es")  # p^r and s^r
slope = (R - P - (R * es - P * ep)) / (r0 * (R - P))
intercept = (es - ep) / (4 * r0)
a0 = 2 * (2 + r0) / (2 - r0) * (1 / r0 - slope)
d0 = 4 * (1 - 4 * r0**2) / (r0 * (2 - r0) * (1 - r0))
assert (
    sp.simplify(
        2 * intercept + a0 * P - d0 * P * ep - ep * N0.subs(E, es / ep) / (3 * delta)
    )
    == 0
)
assert sp.simplify(a0 - (1 + r0) * d0 * ep - ep * Np.subs(E, es / ep) / delta) == 0
assert sp.simplify(a0 - (1 + r0) * d0 - N1.subs({E: ep, F0: es}) / delta) == 0
assert sp.expand(Np.subs(E, 1 + r0) - 11 * r0**2 - 13 * r0**3) == 0

# Five atanh terms and the written elementary tail give the coarse intervals.
for number, low, high in [(2, Q(69, 100), Q(7, 10)), (3, Q(109, 100), Q(11, 10))]:
    z = Q(number - 1, number + 1)
    total = 2 * sum(z ** (2 * j + 1) / (2 * j + 1) for j in range(5))
    tail = 2 * z**11 / (11 * (1 - z * z))
    assert low < total < total + tail < high
intervals = [
    Box(ends=(Box(Q(69, 100)).lo, Box(Q(7, 10)).hi)),
    Box(ends=(Box(Q(109, 100)).lo, Box(Q(11, 10)).hi)),
]


def coarse(poly):
    out = Box(0)
    for powers, coeff in sp.Poly(poly, L2, L3).terms():
        out += (
            Q(int(coeff.p), int(coeff.q))
            * intervals[0] ** powers[0]
            * intervals[1] ** powers[1]
        )
    return out


bounds = [[4, -6, 15, -14, 0], [27, -2, -33, 52, -29], [4, 86, -75, 2, 0]]
tails = [3, 3, 14]
assert Q(4, 3) ** 4 > 3 and Q(8, 5) ** 4 > 6 and Q(6, 5) ** 4 > 2
for i, (J, lower, Mtail) in enumerate(zip([J0, Jp, J1], bounds, tails)):
    alpha = L3 if i < 2 else -L6
    terms = {
        E: sum((alpha * r0) ** j / sp.factorial(j) for j in range(7)),
        F0: sum((-L2 * r0) ** j / sp.factorial(j) for j in range(7)),
    }
    series = sp.Poly(sp.expand(J.subs(terms)), r0)
    assert series.nth(0) == series.nth(1) == 0
    for j, bound in zip(range(2, 7), lower):
        assert coarse(series.nth(j)).lo >= Box(bound).hi
    rest = J.subs({E: 0, F0: 0})
    assert sp.Poly(rest, r0).degree() <= 6
    assert sp.expand(J - rest - J.coeff(E) * E - J.coeff(F0) * F0) == 0
    tail = Q(0)
    for symbol, ab, eb in [
        (E, Q(11, 10) if i < 2 else Q(9, 5), Q(4, 3) if i < 2 else Q(8, 5)),
        (F0, Q(7, 10), Q(6, 5)),
    ]:
        for (j,), coeff in sp.Poly(J.coeff(symbol), r0).terms():
            assert j <= 5
            val = coarse(coeff)
            tail += (
                eb
                * Q(max(abs(val.lo), abs(val.hi)), SCALE)
                * ab ** (7 - j)
                / factorial(7 - j)
            )
    assert tail < Mtail

root = Path(__file__).resolve().parents[1]
home = next(
    p
    for p in [
        root / "renyi_scalar_slack.tex",
        root.parent / "paper" / "renyi_scalar_slack.tex",
    ]
    if p.exists()
).read_text()
table = home.split("% RA-TAYLOR-BEGIN\n")[1].split("% RA-TAYLOR-END")[0]
printed = [
    [int(v) for v in line.rstrip("\\").split("&")[1:]]
    for line in table.strip().splitlines()
]
assert printed == [row + [tail] for row, tail in zip(bounds, tails)]
assert 4 - Q(6, 4) - Q(14, 4**3) - Q(3, 4**5) > 2
assert 27 - Q(2, 4) - Q(33, 4**2) - Q(29, 4**4) - Q(3, 4**5) > 24
assert 86 - Q(75, 4) > 0 and 4 - Q(14, 4**5) > 3

# One elementary algebraic maximum; all root bounds use integer fourth powers.
assert 6 * Q(319, 500) ** 4 < 1 and 2 * Q(841, 1000) ** 4 > 1
upper = Q(203, 500) + Q(36 * 9**4, 35 * 20**4) * Q(377, 200) ** 5
assert upper < Q(141, 100)
print(
    "RA-1 PASS: fourth-derivative and endpoint algebra; 15 printed Taylor bounds; tails 3,3,14; scalar bound 141/100 without a parameter grid"
)

assert Q(141, 200) ** 2 * 2 == Q(19881, 20000) < 1
print("RA-2 PASS: logarithm absorption (141 sqrt(2)/200)^2 = 19881/20000 < 1")

m, r, L, S, Sstar, D, Hh = sp.symbols("m r L S Sstar D Hh")
# M-m=-r(S-Sstar); the logarithm remainder is bounded analytically.
assert sp.expand(-(m - r * (S - Sstar) - m) / (r * m) - (S - Sstar) / m) == 0
# After Cauchy--Schwarz: (S-Sstar)^2 <= (141/25)D, S=Hh-D.
upper = (Hh - D - Sstar) / m + r * sp.Rational(141, 50) * D / (m * L)
assert (
    sp.expand(upper - (Hh - Sstar) / m + (1 - r * sp.Rational(141, 50) / L) * D / m)
    == 0
)
alpha, a, b, Psum, Tsum, F = sp.symbols("alpha a b Psum Tsum F")
gap = -b * (Psum - 1) - a * (Tsum - sp.Rational(5, 12))
assert (
    sp.expand(
        (gap - alpha * (Psum - 1) - a * F / 12).subs(
            {b: -alpha - sp.Rational(17, 12) * a, F: 17 * Psum - 12 * Tsum - 12}
        )
    )
    == 0
)
print(
    "RA-3 PASS: logarithm conversion, slack absorption and summed moment-defect identities"
)


# Exact mandatory negative regression, as a corollary of the parameter result:
# R_q(M4) < 5/3 iff m_q > 2^(-5(q-1)/3). The cluster's cut spectra
# and its constant 5/3 value are checked by the existing EF-2 verifier.
def hs_moment(q):
    return power(R, q) + 3 * power(P, q)


# Coarse bracket by integer comparisons. With m_q = 2^-q(1 + 3^(1-q)):
# m_{3/2} = 2^(-3/2)(1+1/sqrt3) < 2^(-5/6) iff (1+1/sqrt3)^3 = 2+10/(3 sqrt3) < 4 iff 100 < 108;
# m_{8/5} = 2^(-8/5)(1+3^(-3/5)) > 1/2 iff 2^(3/5) < 1+3^(-3/5), from
# 2^(3/5) < 47/31 < 1+3^(-3/5), i.e. 8*31^5 < 47^5 and 27*16^5 < 31^5.
s3, h = sp.sqrt(3), sp.Rational(1, 2)
assert sp.expand(sp.radsimp(h ** sp.Rational(3, 2) + 3 * (h / 3) ** sp.Rational(3, 2)
                  - h ** sp.Rational(3, 2) * (1 + 1 / s3))) == 0
assert sp.radsimp(sp.expand((1 + 1 / s3) ** 3) - (2 + 10 / (3 * s3))) == 0
assert 10 ** 2 < 3 * 6 ** 2  # 10/(3 sqrt3) < 2
assert 8 * 31 ** 5 < 47 ** 5 and 27 * 16 ** 5 < 31 ** 5
# Fine bracket: q_s = 1587/1000 < q_* < 1589/1000 < q_u = 8/5.
for qq, sign in ((Q(1587, 1000), -1), (Q(1589, 1000), 1)):
    d = hs_moment(qq) - power(Q(2), -Q(5, 3) * (qq - 1))
    assert (d.hi < 0) if sign < 0 else (d.lo > 0)
print(
    "RA-4 PASS: integer comparisons at 3/2 (100 < 108) and 8/5 (via 47/31);"
    " exact HS/cluster comparison at 1587/1000 and 1589/1000;"
    " 1587/1000 < q_* < 1589/1000"
)

# RA-5: monotonicity transfers the q=5/4 defect to all larger orders.
a0, b0, _, _, _ = coefficients(Q(5, 4))
alpha0 = -b0 - Q(17, 12) * a0
m0 = hs_moment(Q(5, 4))
ln2 = logarithm(Q(2))
assert (20 * alpha0 - 39 * m0 * ln2).lo > 0
assert (4 * log_interval(m0) + Q(313, 180) * ln2).lo > 0
print("RA-5 PASS: exact purity-defect constants, cap-free P >= 10/9")
