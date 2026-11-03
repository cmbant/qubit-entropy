#!/usr/bin/env python3
"""DA-1--DA-5: exact contacts, printed tables, coverage and negative controls.

Proved in: renyi_uncapped_covering.tex and renyi_dual_tables.tex. This checks
algebra and data independently of the polynomial sign verifiers. The sphere extremum, determinant reduction and equality inference
are proved in the paper. No tolerance check is used as an identity proof.
"""
from fractions import Fraction as F
from pathlib import Path
import re
import sympy as s
from rational_intervals import power, log_interval, LOG2, SCALE
import renyi_dual_data as CO
import verify_renyi_cluster_sliver as HT

def tex_source(name):
    """The TeX source, in ../paper/ or next to code/."""
    root = Path(__file__).resolve().parents[1]
    return next(p for p in (root/name, root.parent/'paper'/name) if p.exists())


R = s.Rational
a, b, al, be, ga, e, f, K = s.symbols('a b alpha beta gamma e f K')
lam = (a, b, (1-a-b)/2, (1-a-b)/2)
p = sum(x*x for x in lam)
t = sum(x**3 for x in lam)
hs = {a: R(1, 2), b: R(1, 6)}
assert [p.subs(hs), t.subs(hs)] == [R(1, 3), R(5, 36)]
for expr, expected in ((p, (R(2, 3), 0)), (17*p-12*t, (R(10, 3), 0)),
                       (p*p, (R(4, 9), 0))):
    assert [s.diff(expr, x).subs(hs) for x in (a, b)] == list(expected)
alpha = R(3, 2)*(K-R(10, 3)*be+R(4, 9)*ga)
assert s.expand(K-R(2, 3)*alpha-R(10, 3)*be+R(4, 9)*ga) == 0

def ell(eps, moment):
    return al*eps+be*moment-ga*(moment**2+R(17, 2)*eps+2*eps**2)/6

def contact(spec, entropy):
    purity = sum(x*x for x in spec)
    third = sum(x**3 for x in spec)
    return entropy+al*(purity-R(1, 3))+be*(17*purity-12*third-4)-ga*(purity**2-R(1, 9))

w1 = contact((R(1, 2), R(1, 2), 0, 0), 1)
w2 = contact((R(1, 4),)*4, 2)
assert s.expand(w1+2*w2-ell(0, R(1, 2))-5) == 0
assert s.expand((w1-w2).subs({ga: 0, al: 4-8*be})) == 0
sg, dl, q = s.symbols('sigma delta q')
spec = ((sg+dl)/2, (sg-dl)/2, (1-sg)/2, (1-sg)/2)
for expr in (sum(x*x for x in spec), sum(x**3 for x in spec), sum(x**q for x in spec)):
    assert s.expand(expr.subs(dl, -dl)-expr) == 0
print('DA-1 PASS: symbolic HS tangency, cluster contacts and exact evenness')

P, U, defect = s.symbols('P U Q')
quadratic = f*f+R(9, 2)*(P-1)-6*(U-P*P/3)
assert s.expand((6*(U-R(1, 3))-(f*f+R(17, 2)*e+2*e*e)+quadratic).subs(P, 1+e)) == 0
assert s.hessian(ell(e, f), (e, f)) == s.diag(-2*ga/3, -ga/3)
x, z, t2, p2, rho = s.symbols('x z t2 p2 rho')
X = R(1, 2)+e-p2
thirds = R(1, 8)+R(3, 4)*X+rho
assert s.expand(17*(1+e)-12*(t2+thirds)-12-(-1+8*e+9*p2-12*t2-12*rho)) == 0
one_large = ((1+3*z)/4,)+((1-z)/4,)*3
assert s.expand(sum(v*v for v in one_large)-(1+3*z*z)/4) == 0
assert s.expand(sum(v**3 for v in one_large)-(1+9*z*z+6*z**3)/16) == 0
assert s.expand(3*z**3+(-3*z)**3+24*z**3) == 0
assert F(1, 2)+HT.EPS-HT.PH == HT.XM == F(1, 25)
assert 1+HT.EPS-HT.PH-F(1, 4) == HT.P2 == F(29, 100)
assert 2*HT.PH+F(1, 4) > 1+HT.EPS
print('DA-2 PASS: dual, Hessian, global third-moment and sliver algebra')

TABLE = tex_source('renyi_dual_tables.tex')
tex = TABLE.read_text()

def rows(tag):
    block = tex.split('% '+tag+'-ROWS-BEGIN\n')[1].split('% '+tag+'-ROWS-END')[0]
    return [line.rstrip('\\').split('&') for line in block.strip().splitlines()]

assert CO.QA == F(1587, 1000) < F(8, 5)
co_rows = rows('CZ')
assert len(co_rows) == 1 == len(CO.BANDS)
expected = {'L': (F(3, 20), F(11, 25), F(0), F(1, 9))}
for row, band in zip(co_rows, CO.BANDS):
    assert row[0] == band[0]
    assert tuple(F(int(x), 10000) for x in row[1:]) == band[1]
    assert band[2] == expected[row[0]]
    assert band[3] is None
ht_rows = rows('HT')
assert len(ht_rows) == 1 and ht_rows[0][0] == 'HT'
assert tuple(F(int(x), 10000) for x in ht_rows[0][1:]) == HT.THETA_A
assert CO.exact_checks()
analytic_theta = (F(20, 9), F(2, 9), F(0))
sub = dict(zip((al, be, ga), map(s.Rational, analytic_theta)))
assert s.expand((w1-w2).subs(sub)) == 0
assert s.expand((3*w1-ell(0, R(1, 2))-5).subs(sub)) == 0
print('DA-3 PASS: all printed rational multiplier/band entries match the executable data')

# Axis-aligned coverage is determined by its endpoints and the open intervals
# between them. This exact enumeration includes shared boundaries.
def partition(points):
    points = sorted(points)
    return points+[(x+y)/2 for x, y in zip(points, points[1:])]

# Outside L, the analytic cubic envelope implies eps < 3/500 < 1/100.
# Thus the old M selection is entirely redundant, including its boundary.
assert (1-2*F(11, 25))/20 == F(3, 500) < F(1, 100)
assert 20*F(1, 100)+2*F(11, 25) == F(27, 25) > 1
for eps in partition([F(0), F(1, 100), F(1, 9)]):
    for moment in partition([F(3, 20), F(11, 25), F(1, 2)]):
        for high in (False, True):
            standard = any(F0 <= moment <= F1 and e0 <= eps <= e1
                           for name, _, (F0, F1, e0, e1), _ in CO.BANDS)
            hot = F(11, 25) <= moment <= F(1, 2) and eps <= F(1, 100) and high
            analytic = 20*eps+2*moment >= 1
            capped = F(11, 25) <= moment <= F(1, 2) and eps <= F(1, 100) and not high
            assert standard or hot or capped or analytic
for order, sign in ((F(1587, 1000), 1), (F(8, 5), -1)):
    val = log_interval(power(F(1,2),order)+3*power(F(1,6),order))/((1-order)*LOG2)
    assert (F(val.lo,SCALE)>F(5,3) if sign>0 else F(val.hi,SCALE)<F(5,3))
print('DA-4 PASS: exact closed-cell coverage and certified crossing endpoints')

# Wrong multipliers and a shifted contact bound fail by exact rational signs.
assert R(5,3)-2 < 0
sub = dict(zip((al,be,ga),map(s.Rational,analytic_theta)))
shifted = R(5,3)+sub[be]/6-R(1,10000)
assert s.expand(shifted-w1.subs(sub)) == -R(1,10000)
assert w1.subs({al:0,be:0,ga:0}) == 1
assert w2.subs({al:0,be:0,ga:0}) == 2
print('DA-5 PASS: exact wrong-multiplier and shifted-contact negative controls')
