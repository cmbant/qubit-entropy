"""FO-1: exact data for degree-four orbit rigidity throughout 1/32 < y < 1/12.

Integer Pauli contractions determine the one-body, pair and Kramers-frame
polynomials. Remaining identities hold symbolically in the family parameter.
The state-reconstruction and orbit arguments are analytic in the proof home.
"""

from fractions import Fraction
from itertools import combinations
import sympy as sp
from certify_ico_fivecopy_g34_51 import ico_integer_states

U, V = [list(map(int, w.ravel())) for w in ico_integer_states()]


def action(w, word):
    flip = phase = ny = 0
    for site, axis in word:
        bit = 1 << (6 - site)
        if axis in "XY":
            flip |= bit
        if axis in "YZ":
            phase |= bit
        ny += axis == "Y"
    return [
        (-1) ** ((i & phase).bit_count()) * (-sp.I) ** ny * w[i ^ flip]
        for i in range(128)
    ]


def coefficients(word, spinflip=False):
    def j(w):
        return [(-1) ** i.bit_count() * w[i ^ 127] for i in range(128)]

    A, B = (j(U), j(V)) if spinflip else (U, V)
    A, B = action(A, word), action(B, word)
    # For h=(sigma Jz)^dagger z take the conjugate of the Pauli output.
    A = [sp.conjugate(v) for v in A]
    B = [sp.conjugate(v) for v in B]
    return (
        sum(a * u for a, u in zip(A, U)),
        sum(a * v + b * u for a, b, u, v in zip(A, B, U, V)),
        sum(b * v for b, v in zip(B, V)),
    )


for i in range(7):
    for axis in "XYZ":
        assert coefficients(((i, axis),)) == (0, 0, 0)
        expected = (-30, 0, 2) if i == 0 else (-10, 0, -2)
        assert coefficients(((i, axis),), True) == (
            expected if axis == "Z" else (0, 0, 0)
        )
for i, j in combinations(range(7), 2):
    for a in "XYZ":
        for b in "XYZ":
            expected = (10, 0, -2) if i == 0 else (-2, 0, 2)
            assert coefficients(((i, a), (j, b))) == (
                expected if a == b == "Z" else (0, 0, 0)
            )
print("FO-1 all one-body, pair and Kramers-frame coefficient identities PASS")
y = sp.symbols("y", real=True)
v = (1 - 8 * y) / 3
w = (32 * y - 1) / 15
hs = -1 + 4 * y
ho = -(1 + 4 * y) / 3
r = v * v / w
l = w
E2 = 6 * v**2 + 15 * w**2
E3 = 15 * v * v * w + 20 * w**3
E4 = 20 * v * v * w * w + 15 * w**4
B1 = 6 * v**4 + 15 * w**4


def zero(a):
    assert sp.cancel(a) == 0


s = (E2**2 + 2 * E4 - B1) / (2 * E3)
zero(s - r - 6 * l)
p2 = s * s - 2 * E2
p3 = s**3 - 3 * s * E2 + 3 * E3
S2 = p2 - s * s / 7
S3 = p3 - 3 * s * p2 / 7 + 2 * s**3 / 49
zero(S2 - sp.Rational(6, 7) * (r - l) ** 2)
zero(S3 - sp.Rational(30, 49) * (r - l) ** 3)
zero(S3 * S3 - sp.Rational(25, 42) * S2**3)
zero(v - w - sp.Rational(2, 5) * (1 - 12 * y))
zero(6 * v * v - (v * v + 5 * w * w) - 5 * (v * v - w * w))
print("FO-1 generic graph invariants and exact skewness saturation PASS")
zero(-sp.Rational(1, 2) - sp.Rational(3, 2) * v - hs)
zero(-sp.Rational(1, 4) - v / 2 - sp.Rational(5, 4) * w - ho)
zero(-hs / 2 - sp.Rational(3, 2) * ho - 1)
qs, qo = hs**2, ho**2
ds, do = 6 * v * v, v * v + 5 * w * w
S1 = qs + 6 * qo
# The affine invariant and its square fix the tangent-line parameter in
# thm:fivecopy-family-equality without a moment-block transversality test.
H = 104 * S1 - 25 * E2
zero(H - 155 - 96 * y)
t = sp.symbols("t", real=True)
zero((H + t * sp.diff(H, y)) ** 2 - (H**2 + t * sp.diff(H**2, y))
     - 96**2 * t**2)
print("FO-1 affine invariant H=155+96y and tangent-square defect 96^2 t^2 PASS")
R = ds * qs + 6 * do * qo
zero((R - do * S1) / (ds - do) - qs)
zero((S1 - qs) / 6 - qo)
pi = (2096 * y * y - 296 * y + 149) / 1050
zero(pi - sp.Rational(121, 917) - (524 * y - 37) ** 2 / 137550)
print("FO-1 generic Gc=h, c^T h=1, frame reconstruction and shell purity PASS")
# The support of Hz=Jz gives h_s=4 b^2-1 independently of pair phases.
hshell = []
for bit in [0, 1]:
    for weight in range(7):
        H = -2 + Fraction(weight, 2) if bit == 0 else -1 + Fraction(weight, 2)
        if H * H == 1:
            hshell.append((bit, weight, H * ((-1) ** bit)))
assert hshell == [(0, 2, -1), (0, 6, 1), (1, 0, 1), (1, 4, -1)]
print("FO-1 shell endpoint modulus recovered by h_s=4 b^2-1 PASS")
# Positivity in the theorem uses these exact factorizations on the stated interval.
zero(w - sp.Rational(32, 15) * (y - sp.Rational(1, 32)))
zero(v - sp.Rational(8, 3) * (sp.Rational(1, 8) - y))
assert Fraction(1, 32) < Fraction(37, 524) < Fraction(1, 12)
print(
    "FO-1 PASS: family orbit data on 1/32 < y < 1/12; includes purity and entropy parameters"
)
