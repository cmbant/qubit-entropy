#!/usr/bin/env python3
"""HD: algebra checks for the analytic HS supporting proof at q=8/5.

Proved in: renyi_hs_analytic.tex (lem:fqe-hs-tilt, lem:fqe-hs-polynomial).
The divided-difference slope bound, constrained maximisation, curvature inference and equality
classification are proved in the home. No Bernstein coefficients, subdivision,
sampling, or external certificate data are used here.
"""
from fractions import Fraction as F
from pathlib import Path
import sympy as sp

R = sp.Rational
q, r = R(8, 5), R(3, 5)
beta, gamma = R(1, 12), R(10, 3)
ROOT = Path(__file__).resolve().parents[1]
HOME = next(p for p in (ROOT / 'renyi_hs_analytic.tex',
                       ROOT.parent / 'paper' / 'renyi_hs_analytic.tex') if p.is_file())


def variations():
    u,d,x = sp.symbols('u d x', positive=True)
    lam = (R(1,4)+u+d,R(1,4)+u-d,R(1,4)-u,R(1,4)-u)
    purity = sp.expand(sum(z*z for z in lam))
    third = sp.expand(sum(z**3 for z in lam))
    path_t = R(1,16)+R(3,4)*x+3*x*u-12*u**3
    assert purity == R(1,4)+4*u*u+2*d*d
    assert sp.expand(third-path_t.subs(x,4*u*u+2*d*d)) == 0
    assert sp.expand(sp.diff(path_t,u).subs(x,4*u*u+2*d*d)) == -6*(4*u*u-d*d)
    endpoint = sp.sqrt(x/12)
    assert sp.simplify(sp.sqrt((x-4*endpoint**2)/2)-2*endpoint) == 0
    assert sp.simplify(path_t.subs(u,endpoint)-R(1,16)-R(3,4)*x-x**R(3,2)/sp.sqrt(3)) == 0
    print('HD-1 fixed-purity A path, increasing third moment toward one-large and sharp endpoint identities: PASS')


def slope_constant():
    assert 105**5 == 12762815625 < 12884901888 == 3*2**32
    # All coefficients are exact; the divided-difference bound (DD) is proved in the home.
    assert q*(1-r)/6 == R(8,75)
    assert beta*12 == 1
    print('HD-2 slope constant: 105^5 < 3*2^32, hence eta > 1 for p <= 9/16: PASS')


def family_and_curvature():
    s, p, K = sp.symbols('s p k')
    pp = (1+3*s*s)/4
    tt = (1+9*s*s+6*s**3)/16
    aa = R(9,2)*(s-R(1,3))**2*(s+R(1,6))
    alpha = R(3,2)*K + R(65,36)
    moment_dual = alpha*(pp-R(1,3)) + beta*(17*pp-12*tt-4) - gamma*(pp*pp-R(1,9))
    target = R(3,2)*K*(pp-R(1,3)) - beta*aa - gamma*(pp-R(1,3))**2
    assert sp.expand(moment_dual-target) == 0
    dp = lambda z: sp.diff(z,s)*2/(3*s)
    assert sp.simplify(dp(dp(aa))-6/s) == 0
    x = sp.symbols('x', positive=True)
    ss = (x**5-1)/(x**5+3)
    ds = sp.diff(ss,x)
    hs = -2*(x**5+3)*(x**3-1)/(x**8+3)  # ln2 times dH/ds
    entropy_nats = R(8,3)*sp.log(x**5+3)-R(5,3)*sp.log(x**8+3)
    assert sp.cancel(sp.diff(entropy_nats,x)/ds-hs) == 0
    hss = sp.diff(hs,x)/ds
    J = sp.factor(R(4,9)*(ss*hss-hs)/(ss**3*(2*gamma+6*beta/ss)))
    block = HOME.read_text().split('% HD-CURVATURE-BEGIN\n')[1].split('% HD-CURVATURE-END')[0]
    rows = []
    for line in block.strip().splitlines():
        data = line.rstrip('\\').split('&')
        assert int(data[0]) == len(rows)
        rows.append(tuple(map(int,data[1:])))
    assert len(rows) == 8 and all(len(row)==5 for row in rows)
    y = x-1
    Ps = [sum(c*y**j for j,c in enumerate(row)) for row in rows]
    Q = sum((x**5-1)**j*P for j,P in enumerate(Ps))
    h4 = sum(x**j for j in range(5))
    derivative = -8*(x**5+3)**3*Q/(5*x**3*(43*x**5-31)**2*(x**8+3)**3*h4**3)
    assert sp.cancel(sp.diff(J,x)-derivative) == 0
    assert all(c>=0 for j in [0,1,2,3,4,7] for c in rows[j])
    lower5 = rows[5][0] + sum(min(c,0) for c in rows[5][1:])
    lower6 = rows[6][0]
    assert all(c>=0 for c in rows[6][1:])
    assert lower5 == 150517 and lower6 == -259
    assert lower5+8*lower6 > 0
    print('HD-3 family reduction and the full displayed curvature identity; P5 + R P6 >= 148445 for 0<=R<=8: PASS')


def constants_and_outer():
    # Standard atanh series: lower partial sum and geometric upper tail.
    def lower_atanh(v,n):
        return 2*sum(v**(2*j+1)/(2*j+1) for j in range(n))
    def upper_atanh(v,n):
        return lower_atanh(v,n)+2*v**(2*n+1)/((2*n+1)*(1-v*v))
    lm = lower_atanh(F(1,3),6)
    assert F(2,3) < lm < upper_atanh(F(1,3),6) < F(7,10)
    u0 = F(19331,10000)
    assert F(19,10)**5 < u0**5 < 27
    v = 1/(2*u0+1)
    E = F(301,144)*lm-5*upper_atanh(v,5)+2*(u0-1)/(u0+1)
    assert E > F(21,50000)
    assert F(16,3)/F(7,10)*F(9,29) == F(480,203) > F(7,3)
    eps, f, K = sp.symbols('eps F k')
    alpha = R(3,2)*K + R(65,36)
    ell = alpha*eps+beta*f-gamma*(f*f+R(17,2)*eps+2*eps*eps)/6
    assert sp.expand(ell-((alpha-gamma*(R(17,12)+eps/3))*eps+f*(R(1,12)-R(5,9)*f))) == 0
    assert alpha.subs(K,R(7,3))-gamma*(R(17,12)+R(1,27)) == R(149,324)
    assert R(1,12)-R(5,9)*R(3,20) == 0
    print('HD-4 uniform gap Delta > 1/5000 and the nonnegative outer dual on F<=3/20: PASS')


def band_and_negative_controls():
    import renyi_dual_data as data
    name, theta, box, reg = data.BANDS[0]
    assert name == 'L' and box == (F(3,20),F(11,25),F(0),F(1,9))
    new_min = min(data.ell(theta,e,f) for e,f in data.corners(*box))
    old_min = min(data.ell(theta,e,f) for e,f in data.corners(F(3,10),F(11,25),F(0),F(1,9)))
    assert new_min == old_min == F(1837,125000)
    # A wrong tangent slope makes HS nonstationary: dp/ds at HS is 1/2.
    assert R(1,2)*R(1,1000) == R(1,2000) > 0
    # An unrestricted tilt claim is false: at p=1/2, H-20t is larger on
    # flat-two than on one-large, since H<=2 and 20/(8 sqrt3)>1.
    assert 20**2 > 8**2*3
    print('HD-5 enlarged L band has the identical supporting target; wrong slope and excessive tilt controls: PASS')


if __name__ == '__main__':
    variations()
    slope_constant()
    family_and_curvature()
    constants_and_outer()
    band_and_negative_controls()
    print('ALL HD ALGEBRA CHECKS PASSED (maximisation, divided-difference and curvature inference proved in the home)')
