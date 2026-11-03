#!/usr/bin/env python3
"""BC-1--BC-3: exact algebra for the analytic grouping/cubic entropy bound.

Proved in: renyi_cubic_bound.tex. No numerical sign tests or subdivisions.
Log-sum, geometric convexity, equality and order transfer are proved in text.
"""
from fractions import Fraction as F
import sympy as s

a, u, v, t, q, L, P, T, FF = s.symbols('a u v t q L P T F', positive=True)
w = a*a*(3-2*a)
assert s.expand(1-w-(1-a)**2*(1+2*a)) == 0
assert s.expand(w-a-a*(1-a)*(2*a-1)) == 0
assert s.expand(s.diff(w, a)-6*a+6*a*a) == 0
J = q*(w*s.log(a)+(1-w)*s.log(1-a))-w*s.log(w)-(1-w)*s.log(1-w)+q*a*(1-a)
target = s.diff(w, a)*(q*s.log(a/(1-a))-s.log(w/(1-w)))
assert s.simplify(s.expand_log(s.diff(J, a)-target, force=True)) == 0
assert s.simplify(J.subs(a, s.Rational(1, 2))-(1-q)*s.log(2)-q/4) == 0
# At q_b: q/(q-1)=4 ln 2, so q/4=(q-1) ln 2 and J(1/2)=0.
assert s.simplify(J.subs(a, s.Rational(1, 2)).subs(q, 4*L/(4*L-1)).subs(s.log(2), L)) == 0
T2 = t*(t+3)/(3*t+1)
assert s.cancel((w/(1-w)).subs(a, t/(1+t))-t*T2) == 0
first = t*s.diff(T2, t)/T2
assert s.cancel(t*s.diff(first, t)-24*t*(t*t-1)/((t+3)**2*(3*t+1)**2)) == 0
assert first.subs(t, 1) == s.Rational(1, 2)
assert s.limit(first, t, s.oo) == 1
lo = F(2, 3)+F(2, 81)
hi = lo+F(2, 5*3**5)*F(9, 8)
assert lo == F(56, 81) and hi == F(1123, 1620) < F(3, 4)
assert F(1, 2) < lo < hi
lo3 = lo+F(2, 1215)
assert 4*lo3/(4*lo3-1) < F(313, 200) < F(1587, 1000)
assert 4*hi/(4*hi-1) > F(3, 2)
print('BC-1 PASS: grouping derivative cancellation, geometric convexity and sharp threshold algebra')

cut = (a*u, a*(1-u), (1-a)*v, (1-a)*(1-v))
p, third = sum(x*x for x in cut), sum(x**3 for x in cut)
e2 = sum(cut[i]*cut[j] for i in range(4) for j in range(i+1, 4))
e3 = sum(cut[i]*cut[j]*cut[k] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4))
cubic = s.Rational(10, 3)-6*p+s.Rational(8, 3)*third
group_bound = 4*a*(1-a)+4*w*u*(1-u)+4*(1-w)*v*(1-v)
assert s.expand(cubic-4*e2-8*e3) == 0
assert s.expand(cubic-group_bound) == 0
for rank, entropy in ((1, 0), (2, 1), (4, 2)):
    assert s.Rational(10, 3)-6*s.Rational(1, rank)+s.Rational(8, 3*rank*rank) == entropy
# Negative control: Shannon entropy at a rational binary spectrum violates CB,
# since H_1(1/4,3/4)=2-(3/4)log_2 3 >3/4 iff 2^5>3^3.
assert 2**5 > 3**3
print('BC-2 PASS: global cubic grouping identity, equality contacts and a lower-order negative control')

sum_bound = 10-6*P+s.Rational(8, 3)*T
assert s.expand(sum_bound.subs(T, (17*P-12-FF)/12)
                -(5-s.Rational(20, 9)*(P-1)-s.Rational(2, 9)*(FF-s.Rational(1, 2)))) == 0
print('BC-3 PASS: unconditional analytic high-F transfer; no variance or chart hypothesis')
