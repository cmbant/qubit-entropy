#!/usr/bin/env python3
"""JI-1--JI-3: exact algebra for the elementary joint-purity deduction.

Proved in: purity_variance_certificate.tex, (JI) and (JB).
The fixed-sum extremum of a convex function and equality arguments are
analytic in the paper; this verifier checks only algebra and tight data.
No floating-point arithmetic or sampling enters these checks.
"""
import sympy as s
P, F, U = s.symbols('P F U', real=True)
T = (17*P - 12 - F)/12
G = 3*P - s.Rational(9, 4) - T - U
V = U - P**2/3
Q = F**2 + s.Rational(9, 2)*(P - 1) - 6*V
assert s.expand(12*G - F*(1 - 2*F)
                - 2*(P - 1)*(3 - 2*P) - 2*Q) == 0
assert s.expand(12*G - (19*P - 15 + F - 12*U)) == 0
print('JI-1 PASS: exact implication identity and sextic-purity algebra')

boxes = [(P - s.Rational(1, 2))**2 + s.Rational(1, 8),
         1 + (P - s.Rational(5, 4))**2 + s.Rational(1, 16),
         2 + (P - 2)**2]
residuals = [F - s.Rational(1, 2) + (P - 1)*(19 - 12*P),
             F + (2*P - 3)*(31 - 12*P)/2,
             F - 12*P**2 + 67*P - 87]
for bound, target in zip(boxes, residuals):
    assert s.expand((12*G).subs(U, bound) - target) == 0
assert residuals[0].subs({F: s.Rational(1, 2), P: 1}) == 0
assert residuals[1].subs({F: 0, P: s.Rational(3, 2)}) == 0
assert residuals[2].subs({F: 0, P: s.Rational(9, 4)}) == 3
assert residuals[2].subs({F: 0, P: 3}) == 6
print('JI-2 PASS: all three purity-box residual formulas and boundary signs')

hs = [s.Rational(1, 2)] + [s.Rational(1, 6)]*3
cluster = [s.Rational(1, 2)]*2 + [s.Rational(0)]*2
uniform = [s.Rational(1, 4)]*4
pure = [s.Rational(1)] + [s.Rational(0)]*3
for label, cuts in [('HS', [hs]*3),
                    ('cluster', [cluster, uniform, uniform]),
                    ('Bell product', [pure, uniform, uniform])]:
    pp = sum((sum(v*v for v in sp) for sp in cuts), s.S(0))
    tt = sum((sum(v**3 for v in sp) for sp in cuts), s.S(0))
    uu = sum((sum(v*v for v in sp)**2 for sp in cuts), s.S(0))
    ff = 17*pp - 12*tt - 12
    assert s.factor(G.subs({P: pp, F: ff, U: uu})) == 0
    assert s.factor(Q.subs({P: pp, F: ff, U: uu})) == 0
assert s.prod([s.Rational(1, 2), s.Rational(1, 4), s.Rational(1, 4)]) == s.Rational(1, 32)
print('JI-3 PASS: exact tight data for HS, cluster, and Bell; Rényi-2 purity-product endpoint')
