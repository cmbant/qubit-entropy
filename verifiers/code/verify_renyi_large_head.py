#!/usr/bin/env python3
"""LHD: exact checks for the large-head chart of the strict band L.

Proved in: renyi_large_head_dual.tex (Lemmas lem:fqe-large-head-curvature and lem:fqe-large-head-dual).
LHD-1  the biased binary derivative J(x,y), its sign pattern, the envelopes of y, and three Bernstein certificates on [0,3/5]
LHD-2  the sign of f_aaa at a = 1/2 (one polynomial with positive coefficients)
LHD-3  the monotonicity in K, the cubic minorant kappa, the logarithm constants and error budgets
LHD-4  the final completed square: two Bernstein certificates of degree 13 and 26
Everything is exact rational arithmetic; powers and logarithms use directed rational intervals.
"""
from fractions import Fraction as F
from math import comb, factorial
import sympy as s
from rational_intervals import Box, power, log_interval, LOG2

Q = s.Rational
q, r = Q(1587, 1000), Q(587, 1000)
alpha, beta, gamma = Q(397, 100), Q(3, 20), Q(159, 100)
z, x, y, qq, t = s.symbols('z x y qq t')
a0, S, d = Q(127, 200), Q(73, 200), Q(27, 200)
target = Q(2507333, 1500000)
LLO = Q(693147, 10**6)
Bz, Cz = (1+2*z*z)/3, (1+6*z*z+2*z**3)/9


def bernstein(P, var, width):
    """Coefficients b_i of P on [0,width] in the basis C(n,i)(v/width)^i(1-v/width)^(n-i)."""
    n = s.Poly(P, var).degree()
    u = s.symbols('u')
    c = s.Poly(s.expand(P.subs(var, width*u)), u).all_coeffs()[::-1]
    c += [0]*(n+1-len(c))
    return [sum(c[j]*Q(comb(i, j), comb(n, j)) for j in range(i+1)) for i in range(n+1)]


def curvature():
    # f_aaaa = h_aaaa - 24 gamma (1+B)^2: only -gamma p^2 contributes, (p^2)'''' = 6 p''^2 and p'' = 2(1+B)
    a, Bs = s.symbols('a Bs')
    p = a*a+Bs*(1-a)**2
    assert s.expand(s.diff(p*p, a, 4)-24*(1+Bs)**2) == 0
    # J(y): (1-x^2)^4 h_aaaa/(16 q/(r L)) with x = 2a-1 and y' = q(1-y^2)/(1-x^2)... in the variable 2a-1
    df = lambda f: s.diff(f, x)+s.diff(f, y)*qq*(1-y*y)/(1-x*x)
    f1 = qq*(y-x)/(1-x*x)
    J = s.cancel(-df(df(df(f1)))*(1-x*x)**4/qq)
    printed = (6*qq**3*y**4-24*qq**2*x*y**3+(-8*qq**3+36*qq*x*x+8*qq)*y*y
               +(24*qq*qq*x-24*x**3-24*x)*y+2*qq**3-36*qq*x*x-8*qq+6*x**4+36*x*x+6)
    assert s.expand(J-printed) == 0
    assert s.expand(s.diff(J, y).subs(y, x)-8*x*(qq-1)*(3*(qq-1)**2*x*x-2*qq*qq+qq+3)) == 0
    assert s.expand(s.diff(J, y, 3)-144*qq*qq*(qq*y-x)) == 0
    assert 3*r*r*Q(3, 5)**2-2*q*q+q+3 < 0 and 8*(q-1) > 0     # J'(x) <= 0 on [0,3/5]
    Jq = lambda xx, yy: printed.subs({qq: q, x: xx, y: yy})
    # the range of K: 3^{-r} <= K <= B^r <= 1; B <= 1/2 forces K < 2/3 because 2^q > 3
    assert (2-power(F(3), F(r))).lo > 0                                                # 3^r < 2: K >= 3^{-r} > 1/2
    assert (power(F(2), F(q))-3).lo > 0                                                # K < 2/3 when B <= 1/2
    # the three envelopes of y; y >= x holds for kappa <= 1 (A) and by a factorisation (B)
    kA = Q(2, 3)
    env = {'A': ((1+x)-kA*(1-x), (1+x)+kA*(1-x), 24*gamma*Q(16, 9)),
           'B': ((1+x)**2-(1-x), (1+x)**2+(1-x), 24*gamma*Q(9, 4)),
           'h': ((1+x)**2-(1-x)**2/2, (1+x)**2+(1-x)**2/2, 24*gamma*Q(16, 9))}
    assert s.expand(env['A'][0]-x*env['A'][1]-(1-kA)*(1-x*x)) == 0
    assert s.expand(env['B'][0]-x*env['B'][1]-(1-x)*(x+x*x)) == 0
    assert s.expand(env['h'][1]-env['h'][0]-(1-x)**2) == 0
    sizes = []
    for tag, (N, D, M) in env.items():
        poly = s.cancel(s.expand((M*r*LLO*(1-x**2)**4/(16*q))*D**4-Jq(x, N/D)*D**4))
        b = bernstein(poly, x, Q(3, 5))
        assert all(c > 0 for c in b), tag
        sizes.append((tag, len(b)-1))
    assert 24*gamma*Q(16, 9) == Q(1696, 25) and 24*gamma*Q(9, 4) == Q(4293, 50)
    # negative control: h_aaaa on the envelope B reaches about 81.0004 (at x = 0), so the constant 80 fails there, exactly
    N, D, _ = env['B']
    assert 80*r*LLO/(16*q) < Jq(0, (N/D).subs(x, 0))
    print('LHD-1 PASS: J(x,y) identities, K range, envelopes; Bernstein certificates of degree',
          ', '.join(f'{n} ({tag})' for tag, n in sizes))


def endpoint():
    k = Q(5247, 10000)
    K = k+(3-3*k-2*q/3)*z*z+(2*q/3-2+2*k)*z**3
    rhs = Q(54, 5)*(1-Cz)+Q(477, 25)*(1-Bz*Bz)
    assert 72*beta == Q(54, 5) and 12*gamma == Q(477, 25)
    P = s.Poly(s.expand(rhs*r*LLO*(1+K)**3-16*q*r*(q+1)*(1-K)*(1+K)**2+16*q**3*(1-K)**3), z)
    quotient, rem = s.div(P, s.Poly(1-z, z))
    assert rem.is_zero and quotient.degree() == 12
    assert all(c > 0 for c in quotient.all_coeffs())
    assert (q+1)-q*q/(3*r) > 0                                   # first term of f_aaa(1/2) increases with v <= 1/3
    assert (power(F(3), -F(r))-F(k)).lo > 0                      # k < K(0) = 3^{-r}
    print('LHD-2 PASS: -f_aaa(1/2,z) >= 0 from (1-z) times a polynomial of degree 12 with positive coefficients')


KAPPA = [Q('0.5238033'), Q('0.0142244'), Q('0.4204281'), Q('0.0384307')]
RHO, U, H0, HL = Q('2.40792594'), Q('2.45774282'), Q('1.10489073'), Q('1.32818582')
E0, EL, EA = Q(1, 10000), Q(6, 10000), Q(1, 10**6)


def scalars():
    lam = d*q/(a0*S)
    assert lam == Q(42849, 46355)
    Rlo, Rhi, Klo = Q(12, 5), Q(5, 2), Q(13, 25)
    # -c (R+K)^2 (1+K) dQ/dK = (1+K)(R+K-lam R t) + t^2 C,  t in [-1,11/9]; both parts are nonnegative
    R, K = s.symbols('R K')
    C = (R+K)*(R-1)-lam*R*(1+K)
    assert Q(11, 9)*lam < Q(113, 100) and Klo > (Q(113, 100)-1)*Rhi            # (R+K) - (11/9) lam R > 0
    assert C.subs({R: Rlo, K: 1}) > 0                                         # C increases in R, decreases in K
    dCR = s.diff(C, R)                                   # = (R-1)+(R+K)-lam(1+K), increasing in R and in K
    assert dCR.subs({R: Rlo, K: Klo}) > 0 and s.diff(dCR, K) == 1-lam and 1-lam > 0
    assert s.diff(C, K) == (1-lam)*R-1 and ((1-lam)*R-1).subs(R, Rhi) < 0
    for lo, hi, root in [(Rlo, Rhi, F(a0)/F(S))]:
        v = power(root, F(q))
        assert Box(F(lo)).hi < v.lo and v.hi < Box(F(hi)).lo                  # R = (127/73)^q in [12/5, 5/2]
    assert Q(11, 9) == (Q(33, 200))/d
    # the cubic Hermite minorant kappa of K at the nodes 3/20, 17/20
    def kval(u):
        return (power(1+2*u, F(q))+2*power(1-u, F(q)))/power(F(3), F(q))
    def kd(u):
        return 2*F(q)*(power(1+2*u, F(r))-power(1-u, F(r)))/power(F(3), F(q))
    lo, hi = Q(3, 20), Q(17, 20)
    inv = s.Matrix([[1, lo, lo**2, lo**3], [0, 1, 2*lo, 3*lo**2], [1, hi, hi**2, hi**3], [0, 1, 2*hi, 3*hi**2]]).inv()
    vals = [kval(F(lo)), kd(F(lo)), kval(F(hi)), kd(F(hi))]
    for i, c in enumerate(KAPPA):
        value = sum((Box(F(inv[i, j]))*vals[j] for j in range(4)), Box())
        assert Box(F(c)).hi < value.lo
    kap = sum(c*z**i for i, c in enumerate(KAPPA))
    assert kap.subs(z, 0) > Klo and kap.subs(z, 1) < 1 and all(c > 0 for c in KAPPA)
    assert q*(q-1)*(q-2)*(q-3) > 0                                            # K'''' > 0
    # constants of the logarithm approximations
    exact = [power(F(a0)/F(S), F(q)), 1/(F(r)*LOG2),
             -log_interval(power(F(a0), F(q))+F(3, 4)*power(F(S), F(q)))/(F(r)*LOG2),
             F(q/r)-log_interval(Box(F(7, 4)))/(F(r)*LOG2)]
    for value, approx in zip(exact, (RHO, U, H0, HL)):
        err = value-Box(F(approx))
        assert -Box(F(2, 10**8)).lo < err.lo and err.hi < Box(F(2, 10**8)).lo
    # (LH2): remainders of ln((1+y)/(1-y)) = 2y + E, |E| <= 2|y|^3/(3(1-y^2)), and rounding
    yl, y0 = Q(704, 10000), Q(383, 10000)
    assert Q(23, 100)/Q(327, 100) < yl
    assert max(Q(23, 100)/(2*Rlo+Q(127, 100)), Q(1, 4)/(2*Rlo+Q(7, 4))) < y0
    for yy, budget in [(yl, EL), (y0, E0)]:
        assert Rhi*2*yy**3/(3*(1-yy*yy))+Q(1, 10**6) < budget
    assert (2+q*max(1/a0, 1/S)+q*Rhi/(a0*S*(Rlo+Klo)**2))*Q(2, 10**8) < EA
    assert Q(33, 200)+Q(33, 200)**2/d == Q(11, 30) < 1                         # |d(t+t^2)| <= 11/30
    assert Q(11, 9)**2 < 2 and abs(1-Q(11, 9)**2) < 1                          # |1-t^2| <= 1 on [-1,11/9]
    print('LHD-3 PASS: monotone in K, minorant kappa, logarithm constants and error budgets')


def final_square():
    def Pgen(a):
        p = a*a+Bz*(1-a)**2
        tt = a**3+Cz*(1-a)**3
        return (alpha+17*beta)*p-12*beta*tt-gamma*p*p-alpha/3-4*beta+gamma/9
    aa = s.symbols('aa')
    v, p0, pl, pa = s.symbols('v p0 pl pa')
    P0, Pl, Pa = Pgen(a0), Pgen(Q(1, 2)), s.diff(Pgen(aa), aa).subs(aa, a0)
    h0 = H0-2*U*(v-Q(3, 4))/(2*RHO+v+Q(3, 4))
    hl = HL-2*U*(v-Q(3, 4))/(v+Q(11, 4))
    ha = -2*q*U*((RHO-v)/(RHO+v)-Q(27, 100))/(1-Q(27, 100)**2)
    E = (v+2*RHO+Q(3, 4))*(v+Q(11, 4))*(v+RHO)
    Fv = (h0+p0)*(1-t**2)+(hl+pl)*t**2+(ha+pa)*d*(t+t**2)+E0+EA+EL*t**2
    poly = s.Poly(s.cancel(E*(target-Fv)), t)
    sub = {v: sum(c*z**i for i, c in enumerate(KAPPA)), p0: P0, pl: Pl, pa: Pa}
    C0, C1, C2 = (s.expand(poly.coeff_monomial(m).subs(sub)) for m in (1, t, t**2))
    assert s.Poly(C2, z).degree() == 13
    b2 = bernstein(C2, z, 1)
    assert all(c >= Q(1, 10) for c in b2)
    det = s.expand(4*C0*C2-C1**2)
    assert s.Poly(det, z).degree() == 26
    bd = bernstein(det, z, 1)
    assert all(c >= Q(3, 100) for c in bd)
    assert Q(4, 5)**2+(1-Q(4, 5))**2/3 == Q(49, 75)          # purity cap 49/75 gives a <= 4/5
    print('LHD-4 PASS: c2 >= 1/10 and 4 c0 c2 - c1^2 >= 3/100 on [0,1] (Bernstein coefficients, degree 13 and 26)')


if __name__ == '__main__':
    curvature()
    endpoint()
    scalars()
    final_square()
