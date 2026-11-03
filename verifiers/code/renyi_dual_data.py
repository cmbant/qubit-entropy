"""Exact multiplier/band data for the canonical Renyi dual table (order q_s = 1587/1000, target 5/3)."""
from fractions import Fraction as F

EPSMAX = F(1, 9)
QA = F(1587, 1000)


def th3(a, b, g):
    return tuple(F(x, 1000) for x in (a, b, g))


BANDS = [
    ('L', th3(3970,150,1590), (F(3,20),F(11,25),F(0),EPSMAX), None),
]


def ell(th, e, f):
    al, be, ga = th
    return al*e+be*f-ga*(f*f+F(17,2)*e+2*e*e)/6


def corners(f0, f1, e0, e1):
    return [(e, f) for e in (e0, e1) for f in (f0, f1)]


def exact_checks():
    return all(min(b[1]) >= 0 for b in BANDS)
