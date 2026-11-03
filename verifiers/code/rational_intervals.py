"""Directed dyadic intervals for exact-rational entropy certificates.

All interval endpoints are integer multiples of 2^-160. Rational powers use
integer root comparisons; logarithms use an atanh series with a rational tail.
Exponentials use a Taylor enclosure on [-1/8,1/8] and repeated squaring.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial
from sympy import integer_nthroot

SCALE = 1 << 160


class Box:
    """Closed intervals whose endpoints are integer multiples of 2^-160."""
    def __init__(self, value=0, *, ends=None):
        if ends is not None:
            self.lo, self.hi = ends
        else:
            v = Q(value) * SCALE
            self.lo = v.numerator // v.denominator
            self.hi = -((-v.numerator) // v.denominator)

    def __add__(self, other):
        other = box(other)
        return Box(ends=(self.lo + other.lo, self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return Box(ends=(-self.hi, -self.lo))

    def __sub__(self, other):
        return self + (-box(other))

    def __rsub__(self, other):
        return box(other) + (-self)

    def __mul__(self, other):
        other = box(other)
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Box(ends=(min(products)//SCALE, -((-max(products))//SCALE)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = box(other)
        if other.hi < 0:
            return (-self)/(-other)
        assert other.lo > 0, 'division interval contains zero'
        reciprocal = Box(ends=(SCALE*SCALE//other.hi,
                              -((-SCALE*SCALE)//other.lo)))
        return self * reciprocal

    def __rtruediv__(self, other):
        return box(other)/self

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out = Box(1)
        for _ in range(n):
            out = out*self
        return out


def box(x):
    return x if isinstance(x, Box) else Box(x)


@lru_cache(None)
def power(x, q):
    """Bracket x^q by checking exact integer powers of the endpoints."""
    x, q = Q(x), Q(q)
    assert x >= 0
    if q < 0:
        return 1/power(x, -q)
    if x == 0:
        assert q > 0
        return Box(0)
    a, b = q.numerator, q.denominator
    numerator = x.numerator**a * SCALE**b
    denominator = x.denominator**a
    k, _ = integer_nthroot(numerator//denominator, b)
    assert k**b * denominator <= numerator < (k+1)**b * denominator
    return Box(ends=(int(k), int(k)+1))


def log_series(x):
    """atanh series on [1,2], with its positive tail enclosed exactly."""
    assert 1 <= x <= 2
    z = Q(x-1, x+1)
    zb = Box(z)
    term = zb
    total = Box(0)
    count = 90
    for k in range(count):
        total += 2*term/Q(2*k+1)
        term *= zb*zb
    tail = 2*z**(2*count+1)/((2*count+1)*(1-z*z))
    return total + Box(ends=(0, box(tail).hi))


LOG2 = log_series(Q(2))


@lru_cache(None)
def logarithm(x):
    assert x > 0
    exponent = 0
    while x < 1:
        x *= 2
        exponent -= 1
    while x > 2:
        x /= 2
        exponent += 1
    return log_series(x) + exponent*LOG2



@lru_cache(None)
def exp_point(k):
    x, splits = Q(k, SCALE), 0
    while abs(x) > Q(1, 8):
        x /= 2
        splits += 1
    xb, term, total = Box(x), Box(1), Box(1)
    for j in range(1, 33):
        term = term*xb/j
        total += term
    tail = box(2*Q(1, 8)**33/factorial(33)).hi
    total = Box(ends=(total.lo-tail, total.hi+tail))
    for _ in range(splits):
        total = total*total
    return total

def exp_interval(x):
    x = box(x)
    return Box(ends=(exp_point(x.lo).lo, exp_point(x.hi).hi))


def log_interval(x):
    x = box(x)
    assert x.lo > 0
    return Box(ends=(logarithm(Q(x.lo, SCALE)).lo,
                     logarithm(Q(x.hi, SCALE)).hi))


def interval_power(x,e):
    x, e = box(x), box(e)
    assert x.lo >= 0
    if x.hi == 0:
        assert e.lo > 0
        return Box(0)
    if x.lo == 0:
        assert e.lo > 0
        upper = exp_interval(log_interval(Box(ends=(x.hi, x.hi)))*e).hi
        return Box(ends=(0, upper))
    return exp_interval(log_interval(x)*e)


def interval(a,b):
    assert a <= b
    return Box(ends=(box(a).lo, box(b).hi))
