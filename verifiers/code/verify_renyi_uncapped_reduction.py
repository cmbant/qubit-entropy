#!/usr/bin/env python3
"""UR-1--UR-4: exact algebra behind analytic uncapped reductions.

Proved in: renyi_uncapped_reduction.tex. The positivity, divided-difference,
majorization and Jensen steps are proved analytically there; no sampled
or floating-point condition is used as a proof input.
"""
import sympy as s
x, r, c, A, B = s.symbols('x r c A B', real=True)
q = 1 + r
# A=x^(q-2), B=(1-x)^(q-2), c=3^(1-q).
g = A*x*x + c*B*(1-x)**2
gprime = q*(A*x-c*B*(1-x))
gsecond = q*r*(A+c*B)
residual = q*(c*A*B*(r+2*x*(1-x)) - A*A*x*x - c*c*B*B*(1-x)**2)
assert s.expand(g*gsecond-gprime*gprime-residual) == 0
print('UR-1 PASS: exact logarithmic-curvature factorization')
u = s.symbols('u', positive=True)
t = u**5/3
poly = u**10-5*u**8+16*u**5-5*u**2+9
endpoint = s.Rational(3,5)*(t+2+1/t)-(u**3+u**-3-2)
assert s.expand(5*u**5*endpoint-poly)==0
assert s.factor(s.diff(poly,u)-10*u*(u**3*(u**5-4*u**3+8)-1))==0
assert 110592 < 49*3125
print('UR-2 PASS: 8/5-endpoint reduction and strictly positive polynomial data')
a,b,c1,d=s.symbols('a b c1 d')
lambda_=[a,b,c1,d]
e2=sum(lambda_[i]*lambda_[j] for i in range(4) for j in range(i+1,4))
e3=sum(lambda_[i]*lambda_[j]*lambda_[k] for i in range(4) for j in range(i+1,4) for k in range(j+1,4))
p=sum(v*v for v in lambda_)
t3=sum(v**3 for v in lambda_)
assert s.factor((2*e2-(1-p)).subs(d,1-a-b-c1))==0
assert s.factor((6*e3-(1-3*p+2*t3)).subs(d,1-a-b-c1))==0
z=s.symbols('z')
gpoly=s.prod((z-v) for v in lambda_)
assert s.Poly(gpoly,z).all_coeffs()==[1,-sum(lambda_),e2,-e3,s.prod(lambda_)]
print('UR-3 PASS: fixed-trace second/third moments determine the quartic up to det')

# The unrestricted spin-flip cap gives tau+W1/2 >= (2 lambda_1-1)^2.
# In 2 Delta = W1+tau-(2 p_k-1), the resulting lower bound is
# W1/2 plus four times the product sum of the smaller roots.
rest_pairs = b*c1+b*d+c1*d
assert s.factor(((2*a-1)**2 - (2*p-1) -4*rest_pairs).subs(d,1-a-b-c1)) == 0
W1 = s.symbols('W1')
assert s.factor((W1 + (2*a-1)**2 - W1/2 - (2*p-1)
                 - 2*(W1/4 + 2*rest_pairs)).subs(d,1-a-b-c1)) == 0
spectrum = [a,1-a,s.S(0),s.S(0)]
p_binary = sum(v*v for v in spectrum)
tau_binary = (2*a-1)**2
assert s.expand(2*p_binary-1-tau_binary) == 0
print('UR-4 PASS: unrestricted crossed-purity W1/4 term and critical rank-two saturation identity')
