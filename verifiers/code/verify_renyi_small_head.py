#!/usr/bin/env python3
"""SHD: scalar arithmetic in the analytic proof of lem:fqe-small-head-dual.

Proved in: renyi_small_head_dual.tex. It uses the monotonicity of A_q from
lem:fqe-binary-quartic, which is proved there by Cauchy's formula. SHD-1 encloses
the eight node values of A_q and H_q(1/3,2/3); SHD-2 the grouping constants
and dual identities; SHD-3 the Bernstein coefficients of the eight quartics;
SHD-4 the uniform rectangle (the SHD column of tab:fqe-moment-rectangle,
through moment_rectangle.check_rectangle). Grouping and monotone inferences and
domain coverage are analytic.
"""
from fractions import Fraction as F
from math import comb
import sympy as s
from rational_intervals import power, log_interval, LOG2, SCALE
import moment_rectangle as MR

R = s.Rational
q, r = R(1587, 1000), R(587, 1000)
LM, LP = R(693147, 10**6), R(693148, 10**6)
alpha, beta, gamma = R(397, 100), R(3, 20), R(159, 100)
u, v, b, z = s.symbols('u v b z', real=True)
A, B = 2*q*r, R(8, 3)*q*r*(q-2)
C, D5 = R(8, 3)*q*r*(q-2)*(q-3), R(0)
D5 = R(2, 3)*(q-4)*C
# Staircase: A_q >= a_i on [u_i, u_{i+1}] since A_q is increasing and A_q(u_i) >= a_i.
NODES = [R(0), R(1, 100), R(3, 100), R(1, 20), R(2, 25), R(13, 100), R(39, 200), R(47, 200), R(1, 4)]
LEVELS = [R(99, 1000), R(217, 1000), R(289, 1000), R(83, 250), R(377, 1000), R(43, 100), R(12, 25), R(101, 200)]
TARGET = R(8023, 4800)


def binary_ent(point):
    point = F(point)
    return -log_interval(power(point, F(q))+power(1-point, F(q)))/(F(r)*LOG2)


def binary_nodes():
    assert LM < R(LOG2.lo, SCALE) < R(LOG2.hi, SCALE) < LP
    assert 4-q/(r*LM) > LEVELS[0]                       # A_q(0+) = 4 - q/(rL)
    for point, lower in zip(NODES[1:-1], LEVELS[1:]):
        H = binary_ent(point)
        quartic = (4*point*(1-point)-H)/(point*(1-point)*(1-2*point)**2)
        assert R(quartic.lo, SCALE) > lower, point
        if point == R(3, 100):
            assert R(quartic.hi, SCALE) < R(3, 10)      # wrong-level control
    # D_q(1/9) = 1 - H_q(1/3,2/3) >= 249/2000, so 9 D_q(1/9) >= 2241/2000.
    assert R(binary_ent(R(1, 3)).hi, SCALE) < R(1751, 2000)
    assert 9*(1-R(1751, 2000)) == R(2241, 2000)
    assert 1-R(1751, 2000) > R(1, 9)                   # kappa > 0
    print('SHD-1 PASS: eight node values of A_q, H_q(1/3,2/3)<1751/2000 (so D_q(1/9)>1/9)')


def head_and_dual():
    # Grouping factor: with u = t^2/(1+t^2), k_{3/2}(u) = (1+t^2)^2/(1+t^3), and
    # k_{3/2}/(1+2u) = (1+t^2)^3/((1+t^2)^3+t^3(1-t)^3); t(1-t) <= 1/4, 1+t^2 >= 1.
    t = s.symbols('t', real=True)
    assert s.expand((1+t**3)*(1+3*t**2)-(1+t**2)**3-t**3*(1-t)**3) == 0
    assert s.expand(R(1, 4)-t*(1-t)-(t-R(1, 2))**2) == 0
    assert 1/(1+R(1, 4)**3) == R(64, 65)
    assert R(2241, 2000)*R(64, 65) == R(8964, 8125) > R(11, 10)
    p0 = R(1, 2)-u+u*u
    p = p0+v/2
    third = R(1, 4)-R(3, 4)*u+R(3, 4)*u*u+R(3, 4)*(1-u)*v
    hh = s.symbols('h')
    dual = 1+hh-R(11, 10)*(1+2*u)*v+alpha*(p-R(1, 3))
    dual += beta*(17*p-12*third-4)-gamma*(p*p-R(1, 9))
    d = u*(1-u)
    target = (R(1999, 1200)+hh-R(179, 50)*d-R(159, 100)*d*d
              +(R(3, 200)+R(37, 50)*u-R(159, 100)*u*u)*v-R(159, 400)*v*v)
    assert s.expand(dual-target) == 0
    av = R(3, 200)+R(37, 50)*u-R(159, 100)*u*u
    assert min(av.subs(u, 0), av.subs(u, R(1, 4))) > 0
    print('SHD-2 PASS: grouping-factor identity and 64/65 bound, concavity chord constant and dual identities')


def bernstein(poly, lo, hi):
    """Bernstein coefficients on [lo, hi] of a polynomial in u."""
    t = s.symbols('t')
    P = s.Poly(s.expand(poly.subs(u, lo+(hi-lo)*t)), t)
    n = P.degree()
    c = [P.coeff_monomial(t**k) for k in range(n+1)]
    return [sum(R(comb(k, j), comb(n, j))*c[j] for j in range(k+1)) for k in range(n+1)]


def staircase_quartics():
    d = u*(1-u)
    def quartic(a):
        return (R(1999, 1200)+R(21, 50)*d-a*d*(1-4*d)-R(159, 100)*d*d
                +(R(3, 200)+R(37, 50)*u-R(159, 100)*u*u)*u*u)
    smallest = None
    for lo, hi, a in zip(NODES, NODES[1:], LEVELS):
        coeffs = bernstein(TARGET-quartic(a), lo, hi)
        assert len(coeffs) == 5 and min(coeffs) > 0, (lo, hi)
        smallest = min(coeffs) if smallest is None else min(smallest, coeffs[0], min(coeffs))
    assert smallest == R(1, 6400)
    # Negative control: without the node value (a=0) the quartic exceeds the target on [3/100,1/4].
    assert any(min(bernstein(TARGET-quartic(R(0)), lo, hi)) < 0 for lo, hi in zip(NODES[2:], NODES[3:]))
    assert (TARGET-quartic(R(0))).subs(u, R(1, 4)) < 0
    print('SHD-3 PASS: eight staircase quartics have positive Bernstein coefficients (minimum 1/6400)')


def uniform_part():
    X, D = R(3, 32), R(9, 50)
    VV = A+B*v+C*v*v
    WW = C/4+D5*v
    assert B+2*C*D < 0
    assert VV.subs(v, D) > R(17, 10) and A < R(19, 10)
    assert WW.subs(v, D) > -R(3, 50) and C/4 < R(3, 8)
    assert R(17, 10)-X*R(3, 50) > 0
    assert 1+X*R(19, 10)+X*X*R(3, 8) < R(6, 5)
    # Recalled coarse V^2-2W bound from the capped-cubic proof.
    assert A*A-C/2 > R(1706, 625)
    assert abs(2*A*B-2*D5) < R(9, 10)
    assert B*B+2*A*C+2*B*C*D > 5
    assert R(1706, 625)-R(81, 2000) > R(8, 3)
    assert R(8, 3)-2*R(19, 10)*R(3, 50)*X > R(7, 4)*R(6, 5)**2
    assert R(7, 4) > 2*R(21, 50)*gamma                  # m > 2 rho gamma
    assert abs(B)+2*C*D+X*abs(D5) < R(19, 10)
    assert (2*R(7, 5)-X*R(19, 10)**2)/(R(3, 5)*R(7, 10)*R(6, 5)**2) > 4
    assert C > R(7, 5)
    # The SHD column of tab:fqe-moment-rectangle: (MR1) but Q, (MR2) and the
    # point values (MR3) at v_* = 27/200 by the shared exact check.
    col = MR.TABLE['SHD']
    assert (col['X'], col['vmin'], col['vmax'], col['vstar']) == (X, 0, D, R(27, 200))
    assert (col['b'], col['c'], col['gamma']) == (12*beta, alpha+8*beta-gamma/2, gamma)
    assert (col['b'], col['c']) == (R(9, 5), R(35, 8))
    assert (col['Nplus'], col['Vplus'], col['Wminus']) == (R(6, 5), R(19, 10), R(3, 50))
    assert (col['Q'], col['m'], col['S'], col['kappa']) == (R(8, 3), R(7, 4), R(19, 10), 4)
    assert (col['f0'], col['eta'], col['k'], col['L']) == (R(1, 800), R(1, 100), 8, (LM, LP))
    assert tuple(R(e.numerator, e.denominator) for e in MR.coefficients()) == (A, B, C, D5)
    mu = MR.check_rectangle('SHD')
    assert mu == R(1, 800)-R(1, 100)**2/8 > 0
    # Purity and centred cubic at the two ends of the AS geometry.
    assert max((R(1, 2)-u+R(3, 2)*u*u).subs(u, j)
               for j in (R(1, 4), R(1, 3))) == R(11, 32)
    assert max((1-3*u+3*u*u).subs(u, j)
               for j in (R(1, 3), R(1, 2))) < R(11, 32)
    assert X/3 < D*D
    uniform = 2-alpha/12-beta/2+7*gamma/144
    assert uniform == TARGET
    assert TARGET < R(5, 3)+R(1837, 375000)-R(1, 10**5)
    # The head cap is essential to these reductions, and lowering the final bound
    # too far already fails at uniform-four.
    counter = (R(3, 5), R(1, 5), R(1, 10), R(1, 10))
    assert counter[0]-counter[1] > 2*counter[2]
    assert sum(j*j for j in counter) == R(21, 50) > R(11, 32)
    assert uniform > R(167, 100)
    assert R(5,3)+R(1837,375000)-R(1,10**5)-uniform == R(97,10**6)
    print('SHD-4 PASS: uniform rectangle (SHD column, shared check), derivative bounds, point comparisons and full-domain constants')



if __name__ == '__main__':
    binary_nodes()
    head_and_dual()
    staircase_quartics()
    uniform_part()
