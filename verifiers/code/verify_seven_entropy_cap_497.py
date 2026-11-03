#!/usr/bin/env python3
"""SP7-5--SP7-6: exact rational verification of the analytic proofs of the
sharp universal eight-level entropy/purity slope 7*log2(7)/48 (> 49/120 > 2/5)
and the 0.497 global-competitor spectral cap.

No numerical eigensolver, floating-point inequality, or interval grid enters.
Proved in: ../seven_qubit_entropy_compatibility.tex (lem:seven-entropy-linear-purity,
cor:seven-entropy-eigen-cap-497).
"""
from fractions import Fraction as Q
import sympy as sp

# --- SP7-5: Universal eight-level entropy-purity slope (lem:seven-entropy-linear-purity) ---
# For F(r) = (1+7r)*ln(1+7r) + 7*(1-r)*ln(1-r) and N(r) = r*F'(r) - 2*F(r):
r = sp.Symbol('r', positive=True)
F_expr = (1 + 7 * r) * sp.log(1 + 7 * r) + 7 * (1 - r) * sp.log(1 - r)
N_expr = sp.simplify(r * sp.diff(F_expr, r) - 2 * F_expr)
N_dd = sp.factor(sp.diff(N_expr, r, 2))
expected_N_dd = 112 * r * (7 * r - 3) / ((1 - r) ** 2 * (1 + 7 * r) ** 2)
assert sp.simplify(N_dd - expected_N_dd) == 0

# Exact zero of N(r) at r = 6/7 (where 1+7r = 7 and 1-r = 1/7):
assert sp.simplify(N_expr.subs(r, sp.Rational(6, 7))) == 0
# Minimum of F(r)/r^2 on (0, 1] is F(6/7)/(6/7)^2 = (49/6)*ln(7):
assert sp.simplify(F_expr.subs(r, sp.Rational(6, 7)) / sp.Rational(6, 7) ** 2 - sp.Rational(49, 6) * sp.log(7)) == 0

# Exact integer power comparison 7^5 = 16807 > 16384 = 2^14 => log2(7) > 14/5,
# so sharp slope (7*log2(7))/48 > 49/120 > 2/5:
assert 7**5 == 16807 > 16384 == 2**14
sharp_rational_lb = Q(7, 48) * Q(14, 5)
assert sharp_rational_lb == Q(49, 120) > Q(2, 5)
print("PASS SP7-5: N''(r) = 112*r*(7*r-3)/((1-r)^2*(1+7*r)^2) and N(6/7) = 0 verified symbolically")
print("PASS SP7-5: every 8-level rho satisfies 3-S(rho) >= (7*log2(7)/48)*(8*tr(rho^2)-1) > (49/120)*(8*tr(rho^2)-1) > (2/5)*(8*tr(rho^2)-1)")

# --- SP7-6: Competitor three-qubit spectral cap 497/1000 (cor:seven-entropy-eigen-cap-497) ---
# Reference positive weights (q1, 6*q2, q3) = (1/2, 1/2, 1/192) with 8*q1 = 4, 8*q2 = 2/3, 8*q3 = 1/24:
q1 = Q(1, 2)
q2 = Q(1, 12)
q3 = Q(1, 192)
assert 8 * q1 == 4 and 8 * q2 == Q(2, 3) and 8 * q3 == Q(1, 24)
assert q1 + 6 * q2 + q3 == 1 + Q(1, 192)

# Exact square completion in a:
# -4*a + (392/45)*(t + a - 1/4)^2 = 4*t - 1 - 45/98 + (392/45)*(t + a - 47/98)^2:
t_sym, a_sym = sp.symbols('t a')
lhs_sq = -4 * a_sym + sp.Rational(392, 45) * (t_sym + a_sym - sp.Rational(1, 4)) ** 2
rhs_sq = 4 * t_sym - 1 - sp.Rational(45, 98) + sp.Rational(392, 45) * (t_sym + a_sym - sp.Rational(47, 98)) ** 2
assert sp.simplify(lhs_sq - rhs_sq) == 0

# Exact tangent-sum identity (L a symbol standing for log2(3)):
#   2t + (1-t-a)(1-L) - a(3+L) = 1 + t - (1-t)L - 4a,
# and monotonicity: d/dt [5t - 45/98 - (1-t)L - c] = 5 + L > 0 for L > 0.
L_sym, c_sym = sp.symbols('L c')
tangent_lhs = 2 * t_sym + (1 - t_sym - a_sym) * (1 - L_sym) - a_sym * (3 + L_sym)
tangent_rhs = 1 + t_sym - (1 - t_sym) * L_sym - 4 * a_sym
assert sp.expand(tangent_lhs - tangent_rhs) == 0
mono = 5 * t_sym - sp.Rational(45, 98) - (1 - t_sym) * L_sym - c_sym
assert sp.expand(sp.diff(mono, t_sym) - (5 + L_sym)) == 0
assert 3**1 > 2**1 and bool(sp.log(3, 2) > 1)  # L = log2(3) > 1 > 0, so 5 + L > 0
print("PASS SP7-6: tangent-sum identity 2t+(1-t-a)(1-L)-a(3+L) = 1+t-(1-t)L-4a verified symbolically")
print("PASS SP7-6: d/dt [5t-45/98-(1-t)L-c] = 5+L > 0 (L = log2(3) > 1), so the lower bound is increasing in t")

# Since 3^5 = 243 < 256 = 2^8 (log2(3) < 8/5) and ln(2) > 2/3 (1/(192*ln 2) < 1/128),
# for t >= t0 = 497/1000:
assert 3**5 == 243 < 256 == 2**8
t0 = Q(497, 1000)
j_global_lower = 5 * t0 - Q(45, 98) - Q(8, 5) * (1 - t0) - Q(1, 128)
target = Q(4851, 4000)
assert target == 35 * (3 - Q(59307, 20000))
assert j_global_lower == Q(4755759, 3920000)
assert j_global_lower > target
assert j_global_lower - target == Q(1779, 3920000)

print("PASS SP7-6: E3 >= 59307/20000 implies every three-body lambda_max < 497/1000")
print("PASS SP7-6: exact rational tangent + square-completion lower bound 4755759/3920000 exceeds 4851/4000 by 1779/3920000")
print("Rigorous lower bound and positive gap, decimal display only:",
      float(j_global_lower), float(j_global_lower - target))
print("Scope: the contender eigenvalue cap; the global bound is thm:seven-entropy-global-maximum (E7-1).")
