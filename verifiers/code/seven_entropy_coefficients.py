"""Shared exact scalar definitions for the seven-qubit entropy certificate."""

import math
from fractions import Fraction
from flint import arb, arb_mat, ctx

ctx.prec = 220
ln2 = arb(2).log()


def f(t):
    return -t * t.log() / ln2


def fp(t):
    return -(t.log() + 1) / ln2


lo, hi = (Fraction("0.0698287976009"), Fraction("0.0698287976010"))


def pol(t):
    p, q = (t.numerator, t.denominator)
    return 2**32 * (q - 2 * p) ** 15 - 3**8 * (q + 8 * p) ** 8 * (q + 28 * p) ** 7


assert pol(lo) > 0 > pol(hi)
for _ in range(180):
    mid = (lo + hi) / 2
    if pol(mid) > 0:
        lo = mid
    else:
        hi = mid


def aa(t):
    return arb(t.numerator) / t.denominator


assert pol(lo) > 0 > pol(hi)
x = arb(aa((lo + hi) / 2), aa((hi - lo) / 2))
u = (1 - 2 * x) / 30
c = 4 * u
a = 3 * u + x
d = u + x
b = 8 * u
C = []
Y = []
for node in [c, a, d]:
    t = 8 * node
    C.append([t**i for i in range(6)])
    Y.append([f(node)])
    C.append([arb(0)] + [i * t ** (i - 1) for i in range(1, 6)])
    Y.append([fp(node) / 8])
P = arb_mat(C).solve(arb_mat(Y))
P = [P[i, 0] for i in range(6)]
# Three exact rational choices; the remaining coefficients enforce contact.
A2 = arb(-115001089107) / 10**12
A3 = arb(18166193766) / 10**12
B1 = arb(14438812336) / 10**12
e = 48 * u
v = 256 * u * u


def eval1(co, t):
    s = arb(0)
    for z in co[::-1]:
        s = s * t + z
    return s


pv = eval1(P, e)
pe = eval1([i * P[i] for i in range(1, 6)], e)
h = (f(b) + f(c)) / 2
he = (fp(b) + fp(c)) / 16
hv = (fp(b) - fp(c)) / (512 * u)
B0 = (v * hv - h + pv) / (v * v) - B1 * e
A1 = (he - pe - v * v * B1) / v - 2 * A2 * e - 3 * A3 * e * e
A0 = hv - 2 * v * (B0 + B1 * e) - A1 * e - A2 * e * e - A3 * e**3
co = P + [A0, A1, A2, A3, B0, B1]
terms = [(i, j) for j in range(3) for i in range(6 - 2 * j)]
CP = [[arb(0) for j in range(6)] for i in range(6)]
# Expand e=4(lambda+mu), v=16(lambda-mu)^2 by the binomial theorem.
for (i, j), cc in zip(terms, co):
    for k in range(i + 1):
        for l in range(2 * j + 1):
            CP[k + l][i + 2 * j - k - l] += (
                cc
                * math.comb(i, k)
                * math.comb(2 * j, l)
                * (-1) ** l
                * 4 ** (i + 2 * j)
            )


E_star = (39 * f(c) + 8 * f(a) + 6 * f(d) + 3 * f(b)) / 7
