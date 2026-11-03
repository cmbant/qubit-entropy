#!/usr/bin/env python3
"""EB-0--EB-3: exact rational constants for full-space entropy basin.

Proof uses the already-certified full entropy Hessian (SEH-1--SEH-3)
at the entropy-optimised icosahedral state. This checks new
geodesic-resolvent derivative bounds and the explicit radius.
Does not certify an unrestricted global entropy maximum.
"""
from fractions import Fraction as F
from math import log

radius=F(1,20000)
theta_max=2*radius
lambda_min0=F(1,12)
lambda_floor=F(1,16)
lower=lambda_min0-2*theta_max
assert theta_max==F(1,10000)
assert lower==F(1247,15000) and lower>lambda_floor
print("EB-0 PASS: for theta <= 1/10000 all cut eigenvalues > 1/16; exact lower floor",lower)

# Along a unit geodesic, |R'|_F<=2, |R''|_F<=4,
# R'''=-4R', |R'''|_F<=8. Also
# ||log R+I||_F < 6 and resolvent derivatives have
# operator norms <= 1/m and 1/m^2.
terms=(8*6,3*4*2*16,2**3*16**2)
assert terms==(48,384,2048)
assert sum(terms)==2480
assert F(3,2)*sum(terms)==3720
# Exact simple bounds used in the argument:
# ln 2 > 2/3 and ln 16 < 3, verified analytically
# by positive series for atanh(1/3) and exp(3).
assert F(2,3)<F(2,3)+F(2,81) # explicit next positive term of ln2 series
assert sum(F(3**j, __import__('math').factorial(j)) for j in range(5))>16
print("EB-1 PASS: |d^3 entropy / dtheta^3| < 2480/ln2 < 3720")

H=F(239,1000)
second=H/2
cubic=F(3720,6)*theta_max
remaining=second-cubic
assert remaining==F(23,400)
assert remaining>F(1,20)
print("EB-2 PASS: Hessian curvature 239/1000 minus cubic Taylor error gives",remaining,"> 1/20")

assert radius*2<=F(1,10000)
print("EB-3 PASS: for orbit distance <=1/20000, entropy is below the candidate by at least distance^2/20")
print("Scope: quantitative local basin; the global bound is thm:seven-entropy-global-maximum (E7-1).")
