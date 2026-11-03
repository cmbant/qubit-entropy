#!/usr/bin/env python3
"""Exact algebra checks for the analytic single-crossing proof, for every q > 1.

Proved in: renyi_single_crossing.tex (lem:fqe-renyi-single-crossing,
cor:fqe-renyi-order-transfer). No interval coverings or sampling remain.
OT-1: family formula, derivative, contact and gradient identities.
OT-2: Euler identity and the second variation splitting equal coordinates.
OT-3: hyperbolic ratio identity (SR), contact, monotonicity derivative and
      integrating factor. Their signs and the global maximiser argument are
      proved analytically in the home, not certified by these identities.
OT-4: increasing kappa by 1/1000 destroys HS stationarity (exact q = 2 control).
"""
import sympy as sp


def sc1_symbolic():
    q, z, x, C = sp.symbols('q z x C', positive=True)
    kap = sp.symbols('kappa', positive=True)
    beta = kap * q - 1
    a = 1 / (1 + 3 * sp.exp(-z))
    b = a * sp.exp(-z)
    Z = a**q + 3 * b**q
    v = (a**q * sp.log(a) + 3 * b**q * sp.log(b)) / Z
    w = 3 * sp.exp(-q*z) / (1 + 3 * sp.exp(-q*z))
    p = 3 * sp.exp(-z) / (1 + 3 * sp.exp(-z))
    fam = beta * sp.log(1 + 3*sp.exp(-z)) - kap * sp.log(1 + 3*sp.exp(-q*z)) - z*w
    assert sp.simplify(sp.expand_log(v - kap*sp.log(Z) - fam, force=True)) == 0
    derivative = beta*(w-p) + q*z*w*(1-w)
    assert sp.simplify(sp.diff(fam, z) - derivative) == 0
    R = (sp.exp((q-1)*z)-1) * (1+3*sp.exp(-q*z)) / (z*(1+3*sp.exp(-z)))
    assert sp.simplify(derivative - z*w*(1-w)*(q-beta*R)) == 0
    g = x**(q-1) * (q*sp.log(x)+C)
    assert sp.simplify(sp.diff(g, x) - x**(q-2)*((q-1)*(q*sp.log(x)+C)+q)) == 0
    lam = sp.symbols('l1:5', positive=True)
    Z = sum(t**q for t in lam)
    v = sum(t**q * sp.log(t) for t in lam) / Z
    grad = lam[0]**(q-1) * (q*sp.log(lam[0])+1-q*v-kap*q) / Z
    assert sp.simplify(sp.diff(v-kap*sp.log(Z), lam[0]) - grad) == 0
    print('OT-1 one-large family formula, derivative through R(z), gradient and stationarity profile: PASS')


def sc2_variations():
    q, kap = sp.symbols('q kappa', positive=True)
    lam = sp.symbols('l1:5', positive=True)
    Z = sum(x**q for x in lam)
    v = sum(x**q * sp.log(x) for x in lam) / Z
    phi = v - kap * sp.log(Z)
    assert sp.simplify(sum(x * sp.diff(phi, x) for x in lam) - (1 - q * kap)) == 0
    # Along (a+t,a-t,c,d), dZ/dt = dv/dt = 0 at t=0.
    a, c, d = sp.symbols('a c d', positive=True)
    t = sp.symbols('t', real=True)
    Zt = (a+t)**q + (a-t)**q + c**q + d**q
    vt = ((a+t)**q * sp.log(a+t) + (a-t)**q * sp.log(a-t)
          + c**q * sp.log(c) + d**q * sp.log(d)) / Zt
    assert sp.simplify(sp.diff(Zt, t).subs(t, 0)) == 0
    assert sp.simplify(sp.diff(vt, t).subs(t, 0)) == 0
    C = 1 - q * vt.subs(t, 0) - q * kap
    target = 2 * a**(q-2) * ((q-1) * (q * sp.log(a) + C) + q) / Zt.subs(t, 0)
    assert sp.simplify(sp.diff(vt - kap * sp.log(Zt), t, 2).subs(t, 0) - target) == 0
    print("OT-2 Euler identity sum lambda_i d_i phi = -beta and equal-coordinate second variation 2g'(a)/Z: PASS")


def sc3_ratio():
    s, z, L = sp.symbols('s z L', positive=True)
    R = (sp.exp(s*z)-1) * (1+sp.exp(L-(s+1)*z)) / (z*(1+sp.exp(L-z)))
    rhs = sp.sinh(s*z)/z + sp.tanh((z-L)/2) * (sp.cosh(s*z)-1)/z
    assert sp.simplify((R-rhs).rewrite(sp.exp)) == 0
    assert sp.simplify((R.subs(z, L)-sp.sinh(s*L)/L).rewrite(sp.exp)) == 0
    # The derivative of sinh(s z)/z has numerator N(s z), with N(0)=0
    # and N'(x)=x sinh(x)>0 for x>0 (the sign argument is in the home).
    x = sp.symbols('x', real=True)
    N = x * sp.cosh(x) - sp.sinh(x)
    assert sp.simplify(sp.diff(sp.sinh(s*z)/z, z) - N.subs(x, s*z)/z**2) == 0
    assert sp.simplify(N.subs(x, 0)) == 0
    assert sp.simplify(sp.diff(N, x) - x*sp.sinh(x)) == 0
    q = sp.symbols('q', positive=True)
    K = sp.log(q) + sp.log(sp.tanh((q-1)*L/2))
    assert sp.simplify((sp.diff(K, q) - 1/q - L/sp.sinh((q-1)*L)).rewrite(sp.exp)) == 0
    print('OT-3 hyperbolic ratio (SR), contact, sinh(s z)/z derivative, and integrating factor: PASS')


def sc4_negative_control():
    z = sp.symbols('z', positive=True)
    L = sp.log(3)
    q = sp.Integer(2)
    kap = 1/q + L/sp.sinh((q-1)*L) + sp.Rational(1, 1000)
    w = 3*sp.exp(-q*z)/(1+3*sp.exp(-q*z))
    p = 3*sp.exp(-z)/(1+3*sp.exp(-z))
    derivative = (kap*q-1)*(w-p) + q*z*w*(1-w)
    assert sp.simplify(derivative.subs(z, L).rewrite(sp.exp)) == -sp.Rational(1, 2000)
    print("OT-4 negative control: kappa + 1/1000 at q = 2 gives phi'(ln 3) = -1/2000, so HS is not stationary: PASS")


if __name__ == '__main__':
    sc1_symbolic()
    sc2_variations()
    sc3_ratio()
    sc4_negative_control()
    print('ALL OT ALGEBRA CHECKS PASSED (global signs and maximisation proved analytically)')
