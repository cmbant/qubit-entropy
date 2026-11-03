#!/usr/bin/env python3
"""QV-1--QV-3: calibration and elementary purity-box algebra for (QV) and (QK).

Proved in: sup_renyi.tex#sec:sup-qk (proofs of thm:fqe-sharp-quadratic-variance and thm:fqe-quadratic-variance).
QV-1 also checks the residual identity (QR), an auxiliary identity not stated in the paper.
The convex box extremum and normal-form attainability are analytic there.
"""
import sympy as s
P, F, U, r, x = s.symbols('P F U r s', real=True)
V = U-P**2/3
G = F/12+s.Rational(19,12)*P-s.Rational(5,4)-U
eps = P-1
H = F**2+3*P*(P-1)-6*V
assert s.expand(H-F*(F-s.Rational(1,2))-5*eps*(eps-s.Rational(1,2))-6*G) == 0
assert s.factor((s.Rational(3,2)*x*x-x**3)**2-x**4/4) == x**4*(x-2)*(x-1)
print('QV-1 PASS: residual identity and the proved minimum-purity reduction')

boxes = [(P-s.Rational(1,2))**2+s.Rational(1,8),
         1+(P-s.Rational(5,4))**2+s.Rational(1,16),
         2+(P-2)**2]
strong = [-(P-s.Rational(3,2))**2,
          (P-s.Rational(3,2))*(s.Rational(21,2)-P),
          (P-s.Rational(9,4))*(s.Rational(75,4)-P)+s.Rational(99,16)]
weak = [4*(P-s.Rational(9,8))*(s.Rational(3,2)-P),
        4*(P-s.Rational(3,2))*(s.Rational(27,8)-P),
        (P-s.Rational(9,4))*(s.Rational(39,2)-4*P)+s.Rational(27,8)]
for box, a, b in zip(boxes, strong, weak):
    assert s.expand(H.subs(U,box)-F**2-a) == 0
    assert s.expand((F**2+s.Rational(9,2)*(P-1)-6*V).subs(U,box)-F**2-b) == 0
print('QV-2 PASS: all box formulas; strong and weak residual domains')

spectra = [[(1+r)/2,(1-r)/2,0,0],[s.Rational(1,4)]*4,[s.Rational(1,4)]*4]
p = [sum(z*z for z in sp) for sp in spectra]
purity = sum(p)
sextic = 17*purity-12*sum(sum(z**3 for z in sp) for sp in spectra)-12
variance = sum(z*z for z in p)-purity**2/3
assert s.expand(purity-1-r*r/2) == 0
assert s.expand(sextic-(1-r*r)/2) == 0
assert s.factor(variance-(1+2*r*r)**2/24) == 0
assert s.factor(sextic**2+3*purity*(purity-1)-6*variance) == 0
assert s.cancel((6*variance-sextic**2)/(purity*(purity-1))) == 3
assert s.cancel((6*variance-sextic**2)/(purity-1)).subs(r,1) == s.Rational(9,2)
print('QV-3 PASS: full cluster-to-Bell equality family and both forced coefficients')
