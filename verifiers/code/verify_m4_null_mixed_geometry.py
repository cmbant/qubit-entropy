#!/usr/bin/env python3
"""MS: sharp mixed-moment orbit comparison on the full minimum-purity locus.

Checks the elementary opposite-parity reduction (MS-1), the same-parity
parameterization, monotonicity and polynomial minorants (MS-2), the three
univariate polynomial decomposes closing the same-parity triangle (MS-3), and
the sharp interpolation and cluster equality (MS-4) in
null_mixed_orbit_geometry.tex. No external certificate data or floating-point
proof input.
"""
from itertools import product
import sympy as s

R, SQ = s.Rational, s.sqrt(3)
a, b, c, L = s.symbols('a b c L', real=True)
B = 9-R(9, 2)*SQ
base = L-1-R(3, 8)*(L*L-1)**2
f = s.expand(base+(3*L-B)*L*(L*L-2)/9)
assert s.simplify(f-(-L**4/24+(SQ/2-1)*L**3+L*L/12+(3-SQ)*L-R(11, 8))) == 0
assert s.simplify(f.subs(L, s.sqrt(2))-s.sqrt(2)+R(11, 8)) == 0
assert s.simplify(f.subs(L, SQ)) == 0
assert s.simplify(s.diff(f, L, 2)+L*L/2-6*(SQ/2-1)*L-R(1, 6)) == 0
ll, e2 = a+b+c, a*b+b*c+c*a
assert s.expand(a**4+b**4+c**4-(a*a+b*b+c*c)**2+2*e2**2-4*ll*a*b*c) == 0
assert s.expand(ll**3+9*a*b*c-4*ll*e2
                -(a-b)**2*(a+b-c)-c*(a-c)*(b-c)) == 0
p, q, u, v = s.symbols('p q u v')
A0, beta = 1-1/SQ, 1/(2*SQ)
fr, fs = 2-A0*p+beta*p*u, 2-A0*q+beta*q*v
S = (p+q-p*q)/2+p*q*(1-u)*(1-v)/6
endpoint = 4/(4-SQ)
rem = s.Poly(s.expand(S-endpoint*(1-fr*fs/4)), p, q, u, v)
assert all(max(mm) <= 1 for mm in rem.monoms())
assert all(s.simplify(rem.as_expr().subs(dict(zip((p, q, u, v), bits)))).is_nonnegative
           for bits in product((0, 1), repeat=4))
print('[PASS] MS-1 strengthened sphere norm algebra and all 16 multiaffine vertices')

# MS-2: same-parity parameterization, affine minorant X_lin, monotonicity, and polynomial minorants.
x, y, wt = s.symbols('x y wt', real=True)
Gideal = s.groebner([a*a+b*b+c*c-1], a, b, c, domain=s.QQ)
x_sub, wt_sub = SQ*c, 3*a*b
X_exact = 2 - 2*x**2*(1 - x**2/3) - R(2, 3)*wt**2
X_lin = 2 - 2*x**2*(1 - x**2/3) - (1 - x**2/3)*wt
assert Gideal.reduce(s.expand(X_exact.subs({x: x_sub, wt: wt_sub}) - (3*(a**4+b**4+c**4) - 1)))[1] == 0
assert s.expand((X_exact - X_lin) - R(2, 3)*wt*((3 - x**2)/2 - wt)) == 0
assert Gideal.reduce(s.expand((X_exact - X_lin).subs({x: x_sub, wt: wt_sub}) - 3*a*b*(a - b)**2))[1] == 0
assert s.expand(X_lin - (R(1, 2)*(1 - x**2)**2 + (1 - x**2/3)*((3 - x**2)/2 - wt))) == 0

# Nonnegativity of partial derivatives in wt_r, wt_s:
# ((sqrt(3) + 1) / (18 * sqrt(2) * (4 - sqrt(3))))^2 > (1/24)^2
lhs_sq = ((SQ + 1)/(18*(4 - SQ)))**2 / 2
rhs_sq = R(1, 576)
assert s.simplify(lhs_sq - rhs_sq) > 0
assert s.simplify((2*SQ/(SQ + 1))*(1 + 1/SQ - y/3) - (2 - R(2, 3)*y) - (SQ - 1)*y/3) == 0

wt_low = x*(5 - 2*x)/3
assert s.expand((3 - 2*x**2) - ((5 - 2*x)/3)**2 - R(2, 9)*(1 - x)*(1 + 11*x)) == 0
X_b = R(1, 9)*(1 - x)*(18 + 3*x - 13*x**2)
assert s.expand(X_lin.subs(wt, wt_low) - X_b - R(4, 9)*x**2*(1 - x)**2) == 0

S_b = 1 + x/2 - R(7, 20)*x**2
rad_diff = s.expand((9 + 10*x - 7*x**2) - 9*S_b**2)
assert s.expand(rad_diff - x*(10 - 7*x)*(63*(x - R(5, 7))**2 + R(55, 7))/400) == 0
print('[PASS] MS-2 exact same-parity parameterization, monotonicity in wt, and polynomial minorants')

# MS-3: three univariate polynomials closing P(x, y) >= 0 on 0 <= y <= x <= 1.
f_hat = 1 + S_b/SQ + x/3
g_hat = 1 + S_b.subs(x, y)/SQ - y/3
Y_b = X_b.subs(x, y)
B_const = 1 + 1/SQ
P_xy = (1 + x**2*y**2)/2 + X_b*Y_b/24 + endpoint*f_hat*g_hat/4 - endpoint

q2 = (1 + 8*SQ/9) + (R(16, 9) + 97*SQ/270)*y - (R(13, 9) + 13*SQ/27)*y**2
assert s.expand(s.radsimp(g_hat - B_const*Y_b/2 - y*q2/2)) == 0
assert s.simplify(q2.subs(y, 0)) > 0 and s.simplify(q2.subs(y, 1)) > 0

q1 = (4 - SQ)*(endpoint*B_const*s.diff(f_hat, x)/8 + s.diff(X_b, x)/24)
q1_sos = (
    (R(11, 20) - R(3, 5)*x)**2
    + (5*SQ/24 - R(1189, 3600))
    + (17*SQ/540 - R(133, 2700))*x
    + (R(163, 450) - 13*SQ/72)*x**2
)
assert s.expand(s.radsimp(q1 - q1_sos)) == 0
for coeff in (5*SQ/24 - R(1189, 3600), 17*SQ/540 - R(133, 2700), R(163, 450) - 13*SQ/72):
    assert s.simplify(coeff) > 0

dP_dx = s.diff(P_xy, x)
dP_rem = s.expand(s.radsimp(dP_dx - Y_b*q1/(4 - SQ) - x*y**2 - endpoint*s.diff(f_hat, x)*(y*q2/2)/4))
assert dP_rem == 0

P_diag = s.expand(s.radsimp(P_xy.subs(x, y)))
assert s.simplify(P_diag.subs(y, 0)) == 0
q3_scaled = s.expand(s.radsimp((4 - SQ)*s.cancel(P_diag/y)))
q3_sos = (
    (R(13, 25) - R(26, 25)*y - y**2/10)**2
    + (11*SQ/18 - R(5896, 5625))
    + (R(737, 7500) - 19*SQ/360)*y
    + (R(173411, 202500) - 79*SQ/162)*y**2
    + (R(756757, 486000) - 419*SQ/972)*y**3
    - (R(21043, 24300) - 52*SQ/243)*y**4
    + 169*(4 - SQ)/1944 * y**5
)
assert s.expand(s.radsimp(q3_scaled - q3_sos)) == 0
c0 = 11*SQ/18 - R(5896, 5625)
c1 = R(737, 7500) - 19*SQ/360
c2 = R(173411, 202500) - 79*SQ/162
c3 = R(756757, 486000) - 419*SQ/972
c4 = R(21043, 24300) - 52*SQ/243
c5 = 169*(4 - SQ)/1944
assert all(s.simplify(c) > 0 for c in (c0, c1, c2, c3, c5))
assert s.simplify(c3 - R(4, 5)) > 0 and s.simplify(R(1, 2) - c4) > 0
print('[PASS] MS-3 exact univariate decompositions for q1(x), q2(y), and diagonal q3(y) on [0, 1]')

th = s.symbols('theta', real=True)
f0 = (2+SQ)/6
a0, c0 = 3/(4-SQ), (3+SQ)/4
assert s.simplify((1-24*th)*a0+24*th*endpoint-(1+8*th)*a0) == 0
assert s.simplify((R(1, 2)+4*th)-(1+8*th)*a0*(1-f0)) == 0
assert s.simplify((R(1, 2)+4*th)-(1+8*th)*c0*(1-1/SQ)) == 0
print('[PASS] MS-4 sharp interpolation and cluster equality for every theta in the interval')
print('All MS checks passed; both parity sectors and the interpolation family are verified analytically.')
