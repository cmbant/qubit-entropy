#!/usr/bin/env python3
"""AH-1, AH-5: the entropy-contender purity band and the joint spin-flip relations.

Proof home: seven_qubit_entropy_joint.tex (prop:seven-competitor-band,
lem:seven-joint-spinflip). AH-1 is exact interval algebra; AH-5 checks both
the computational-basis block diagonalization of lem:seven-joint-spinflip
and exact integer arithmetic at seven determining amplitude ratios.
"""
from fractions import Fraction as F

SCALE = 10**60


class I:
    """Closed rational interval, rounded outwards after each operation."""
    def __init__(self, a, b=None):
        if isinstance(a, I):
            self.lo, self.hi = a.lo, a.hi
            return
        a, b = F(a), F(a if b is None else b)
        assert a <= b
        self.lo = F((a * SCALE).__floor__(), SCALE)
        self.hi = F((b * SCALE).__ceil__(), SCALE)

    def __add__(self, other):
        other = I(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-I(other))

    def __rsub__(self, other):
        return I(other) + (-self)

    def __mul__(self, other):
        other = I(other)
        values = [a * b for a in (self.lo, self.hi)
                  for b in (other.lo, other.hi)]
        return I(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = I(other)
        assert other.lo * other.hi > 0
        return self * I(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return I(other) / self

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        value = I(1)
        for _ in range(n):
            value = value * self
        return value


def log_interval(value):
    value = I(value)
    assert value.lo > 0
    t = (value - 1) / (value + 1)
    radius = max(abs(t.lo), abs(t.hi))
    assert radius < 1
    power, total = t, I(0)
    terms = 90
    for k in range(terms):
        total += 2 * power / (2 * k + 1)
        power *= t * t
    tail = 2 * radius**(2 * terms + 1) / (
        (2 * terms + 1) * (1 - radius * radius))
    return total + I(-tail, tail)


LN2 = log_interval(2)


def f(value):
    return -value * (log_interval(8 * value) - 3 * LN2) / LN2


def root_polynomial(value):
    p, q = value.numerator, value.denominator
    return (2**32 * (q - 2 * p)**15
            - 3**8 * (q + 8 * p)**8 * (q + 28 * p)**7)


lo, hi = F(698287976009, 10**13), F(698287976010, 10**13)
assert root_polynomial(lo) > 0 > root_polynomial(hi)
for _ in range(110):
    mid = (lo + hi) / 2
    if root_polynomial(mid) > 0:
        lo = mid
    else:
        hi = mid
assert root_polynomial(lo) > 0 > root_polynomial(hi)
x = I(lo, hi)
u = (1 - 2 * x) / 30
c, a, b, d = 4 * u, 3 * u + x, 8 * u, u + x
e_star = (195 * f(c) + 40 * f(a) + 15 * f(b) + 30 * f(d)) / 35
p_star = (2096 * x * x - 296 * x + 149) / 1050
p_min, p_max = F(121, 917), F(8693, 64000)
assert p_max == (1 + F(5, 2) * (3 - F(59307, 20000))) / 8
assert p_min < p_star.lo < p_star.hi < p_max
print('PASS AH-1: entropy competitors obey 121/917 <= pi3 < 8693/64000; candidate lies strictly inside')

# The two signed amplitude patterns U, V of the icosahedral family.
import itertools
import numpy as np

U, V = np.zeros(128, dtype=np.int64), np.zeros(128, dtype=np.int64)
negative = {tuple(sorted(e)) for e in [(1, 3), (3, 2), (2, 4), (4, 5), (5, 1)]}
for i, j in itertools.combinations(range(6), 2):
    index = (1 << (5 - i)) + (1 << (5 - j))
    sign = -1 if (i, j) in negative else 1
    U[index], U[127 - index] = sign, -sign
V[63] = V[64] = 1
states = np.stack([U, V], axis=1)
assert np.array_equal(states.T @ states, np.diag([30, 2]))
# Joint spin-flip relations have degree at most six in the real ratio t.
EPS = np.array([[0, 1], [-1, 0]], dtype=np.int64)
J3 = np.kron(np.kron(EPS, EPS), EPS)
I8 = np.eye(8, dtype=np.int64)
for t in range(7):
    z = (U + t * V).reshape((2,) * 7)
    for sites in itertools.combinations(range(7), 3):
        axes = sites + tuple(j for j in range(7) if j not in sites)
        M = z.transpose(axes).reshape(8, 16)
        R = M @ M.T
        assert np.max(np.abs(R)) < 1024  # int64 products below 2**40
        Rt = J3 @ R.T @ J3.T
        even, odd = R + Rt, R - Rt
        if 0 not in sites:
            assert not np.any(odd)
            assert not np.any((even - 8 * I8) @ (even - 2 * (3 + t*t) * I8))
        else:
            assert not np.any(even @ odd - odd @ even)
            assert not np.any(odd @ (even - 12 * I8))
            assert not np.any(2 * (even - 8 * I8) @ (even - 2 * (1 + t*t) * I8)
                              - (5 - t*t) * odd @ odd)
print('PASS AH-5: all 35 joint spin-flip matrix relations, degree <= 6, at seven determining ratios')
