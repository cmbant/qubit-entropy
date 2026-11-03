"""AH-6: exact rational coefficient check for the original joint majorant.

The analytic proof is joint_entropy_analytic_proof.tex. It reduces the original
entropy inequality to a degree-18 univariate polynomial. This script checks
114 fixed coefficients, not a covering of the entropy gap. All numerical
enclosures use directed dyadic rational arithmetic; no Arb or floats are used.
"""

# log: verify_seven_entropy_joint_majorant.txt
from fractions import Fraction as Q
from math import comb
from pathlib import Path
import importlib.util
import sympy as sp

helper = Path(__file__).with_name("rational_intervals.py")
if not helper.is_file():  # Repository layout; the public companion flattens code/.
    helper = (
        Path(__file__).resolve().parents[2]
        / "four_qubit_entropy/code/rational_intervals.py"
    )
spec = importlib.util.spec_from_file_location("entropy_rational_intervals", helper)
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)
Box, interval, log_interval, S = ri.Box, ri.interval, ri.log_interval, ri.SCALE


def add(p, q):
    return [
        (p[i] if i < len(p) else Box(0)) + (q[i] if i < len(q) else Box(0))
        for i in range(max(len(p), len(q)))
    ]


def scale(p, c):
    return [z * c for z in p]


def mul(p, q):
    out = [Box(0)] * (len(p) + len(q) - 1)
    for i, z in enumerate(p):
        for j, w in enumerate(q):
            out[i + j] = out[i + j] + z * w
    return out


def power(p, n):
    out = [Box(1)]
    for _ in range(n):
        out = mul(out, p)
    return out


def ev(p, t):
    out = Box(0)
    for a in reversed(p):
        out = out * t + a
    return out


def der(p):
    return [i * p[i] for i in range(1, len(p))]


def divlin(p, node):
    out = [Box(0)] * (len(p) - 1)
    acc = p[-1]
    out[-1] = acc
    for i in range(len(p) - 2, 0, -1):
        acc = p[i] + node * acc
        out[i - 1] = acc
    return out, p[0] + node * acc


def zero_contains(v):
    assert v.lo <= 0 <= v.hi


def pol(t):
    p, q = t.numerator, t.denominator
    return 2**32 * (q - 2 * p) ** 15 - 3**8 * (q + 8 * p) ** 8 * (q + 28 * p) ** 7


lo, hi = Q("0.0698287976009"), Q("0.0698287976010")
assert pol(lo) > 0 > pol(hi)
for _ in range(180):
    mid = (lo + hi) / 2
    if pol(mid) > 0:
        lo = mid
    else:
        hi = mid
x = interval(lo, hi)
u = (1 - 2 * x) / 30
c, a, d, b = 4 * u, 3 * u + x, u + x, 8 * u
nodes = [8 * c, 8 * a, 8 * d]
e0 = 48 * u
v0 = 256 * u * u
ln2 = ri.LOG2
ln3 = log_interval(Box(3))
f = lambda t: -t * log_interval(t) / ln2
fp = lambda t: -(log_interval(t) + 1) / ln2
# Explicit Hermite basis; avoids an interval linear solve.
p = [Box(0)]
for i, node in enumerate(nodes):
    Li = [Box(1)]
    for j, other in enumerate(nodes):
        if i != j:
            Li = mul(Li, [-other / (node - other), 1 / (node - other)])
    slope = ev(der(Li), node)
    coeff = fp(node / 8) / 8 - 2 * slope * f(node / 8)
    p = add(p, mul(power(Li, 2), [f(node / 8) - coeff * node, coeff]))
for node in nodes:
    zero_contains(ev(p, node) - f(node / 8))
    zero_contains(ev(der(p), node) - fp(node / 8) / 8)
a2 = Box(Q(-115001089107, 10**12))
a3 = Box(Q(18166193766, 10**12))
b1 = Box(Q(14438812336, 10**12))
h = (f(b) + f(c)) / 2
he = (fp(b) + fp(c)) / 16
hv = (fp(b) - fp(c)) / (512 * u)
b0 = (v0 * hv - h + ev(p, e0)) / v0**2 - b1 * e0
a1 = (he - ev(der(p), e0) - v0**2 * b1) / v0 - 2 * a2 * e0 - 3 * a3 * e0**2
a0 = hv - 2 * v0 * (b0 + b1 * e0) - a1 * e0 - a2 * e0**2 - a3 * e0**3
q = [Box(1)]
for node in nodes:
    q = mul(q, [-node, Box(1)])
q0 = ev(q, e0)
k0 = (ev(p, e0) - f(e0 / 8)) / q0**2
k1 = (ev(der(p), e0) - fp(e0 / 8) / 8) / q0**2 - 2 * k0 * ev(der(q), e0) / q0
gamma = -k1 / k0
L = [1 - gamma * e0, gamma]
C = Q(1593, 4) * ln2 - 243 * ln3 - 9
D = Q(81, 2) - Q(9477, 4) * ln2 + 1458 * ln3
assert k0.lo > 0 and gamma.lo > 0 and L[0].lo > 0 and C.lo > 0 and D.lo > 0
print(
    "AH-6a PASS: exact root/logarithm enclosures, original Hermite/contact coefficients and positive kernel constants"
)
AA = scale(power(q, 2), 8 * ln2 * k0)
BB = add([Box(Q(1, 2))], scale([Box(0), a0, a1, a2, a3], 8 * ln2))
CC = add([C], scale([Box(0), Box(0), Box(0), b0, b1], 8 * ln2))
EL = mul([Box(0), Box(1)], L)
# -disc(F_v)/(e^10 L(e)^2), where F=e^5 AA+e^4 L BB v+e^2 L CC v²+D L v³.
R = add(
    add(
        add(
            add(
                scale(mul(power(EL, 2), mul(power(BB, 2), power(CC, 2))), -1),
                scale(mul(EL, mul(AA, power(CC, 3))), 4),
            ),
            scale(mul(power(EL, 2), power(BB, 3)), 4 * D),
        ),
        scale(power(AA, 2), 27 * D**2),
    ),
    scale(mul(EL, mul(AA, mul(BB, CC))), -18 * D),
)
quot, rem0 = divlin(R, e0)
quot, rem1 = divlin(quot, e0)
zero_contains(rem0)
zero_contains(rem1)
# Exact contacts and their first derivatives force the factor; enclosures alone do not.
A, B, CCs, Ds, m, r = sp.symbols("A B C D m r")
ap, bp, cp, dp = sp.symbols("ap bp cp dp")
dis = (
    B**2 * CCs**2
    - 4 * A * CCs**3
    - 4 * B**3 * Ds
    - 27 * A**2 * Ds**2
    + 18 * A * B * CCs * Ds
)
sub = {A: m * r**2, B: Ds * r**2 - 2 * m * r, CCs: m - 2 * Ds * r}
assert sp.expand(dis.subs(sub)) == 0
variation = sum(
    sp.diff(dis, z) * dz for z, dz in zip([A, B, CCs, Ds], [ap, bp, cp, dp])
)
assert sp.expand(variation.subs(sub).subs(ap, -bp * r - cp * r**2 - dp * r**3)) == 0
ee, ll, uu, vv, ww = sp.symbols("e L U V W")
reduced = (
    27 * Ds**2 * uu**2
    + 4 * ee * ll * uu * ww**3
    + 4 * Ds * ee**2 * ll**2 * vv**3
    - ee**2 * ll**2 * vv**2 * ww**2
    - 18 * Ds * ee * ll * uu * vv * ww
)
zero_disc = dis.subs(
    {A: ee**5 * uu, B: ee**4 * ll * vv, CCs: ee**2 * ll * ww, Ds: Ds * ll},
    simultaneous=True,
)
assert sp.expand(zero_disc + ee**10 * ll**2 * reduced) == 0
# The original coefficient formulas force value and both partial contacts.
ep, vp, pp0, pp1, hh, he_, hv_, aa2, aa3, bb1 = sp.symbols(
    "ep vp P0 P1 h he hv alpha2 alpha3 beta1"
)
bb0 = (vp * hv_ - hh + pp0) / vp**2 - bb1 * ep
aa1 = (he_ - pp1 - vp**2 * bb1) / vp - 2 * aa2 * ep - 3 * aa3 * ep**2
aa0 = hv_ - 2 * vp * (bb0 + bb1 * ep) - aa1 * ep - aa2 * ep**2 - aa3 * ep**3
ae = aa0 + aa1 * ep + aa2 * ep**2 + aa3 * ep**3
assert sp.expand(pp0 + vp * ae + vp**2 * (bb0 + bb1 * ep) - hh) == 0
assert sp.expand(ae + 2 * vp * (bb0 + bb1 * ep) - hv_) == 0
assert (
    sp.expand(pp1 + vp * (aa1 + 2 * aa2 * ep + 3 * aa3 * ep**2) + vp**2 * bb1 - he_)
    == 0
)
log2, log3, zz = sp.symbols("log2 log3 z")
binary_c = Q(1593, 4) * log2 - 243 * log3 - 9
binary_d = Q(81, 2) - Q(9477, 4) * log2 + 1458 * log3
minorant = zz / 2 + binary_c * zz**2 + binary_d * zz**3
assert (
    sp.expand(minorant.subs(zz, sp.Rational(1, 9)) - (sp.Rational(5, 3) * log2 - log3))
    == 0
)
assert (
    sp.expand(
        sp.diff(minorant, zz).subs(zz, sp.Rational(1, 9)) - sp.Rational(3, 4) * log2
    )
    == 0
)
assert len(quot) == 19
print(
    "AH-6b PASS: symbolic cubic discriminant identity, double-factor lemma and"
    " binary contacts; enclosed degree-18 quotient"
)


def bern(p, l, h):
    n = len(p) - 1
    transformed = [
        sum(
            (p[j] * comb(j, i) * l ** (j - i) * (h - l) ** i for j in range(i, n + 1)),
            Box(0),
        )
        for i in range(n + 1)
    ]
    return [
        sum((transformed[i] * Q(comb(k, i), comb(n, i)) for i in range(k + 1)), Box(0))
        for k in range(n + 1)
    ]


cut = [Q(0), Q(3, 4), Q(1), Q(5, 4)]
table_bounds = [Q(2, 10**7), Q(5, 10**10), Q(9, 10**11)]
tail_start, tail_bound = Q(5, 4), Q(2, 10**10)
for l, h, bound in zip(cut, cut[1:], table_bounds):
    bc = bern(quot, l, h)
    assert all(z.lo > 0 for z in bc), (l, h, min(z.lo for z in bc))
    assert min(z.lo for z in bc) * bound.denominator > bound.numerator * S
    lower = min(z.lo for z in bc) * 10**14 // S
    assert lower > 0
    print(
        f"AH-6c [{l},{h}] degree 18: all 19 Bernstein coefficients >= {lower}/10^14 > 0"
    )
tail = [
    sum(
        (quot[j] * comb(j, i) * tail_start ** (j - i) for j in range(i, len(quot))), Box(0)
    )
    for i in range(len(quot))
]
assert all(z.lo > 0 for z in tail)
assert min(z.lo for z in tail) * tail_bound.denominator > tail_bound.numerator * S
lower = min(z.lo for z in tail) * 10**14 // S
print(
    f"AH-6c [{tail_start},infinity) degree 18: all 19 shifted power coefficients >= {lower}/10^14 > 0"
)
print(
    "AH-6 PASS: original scalar majorant on the full nonnegative quadrant; fixed univariate polynomial certificate, no entropy-gap subdivision"
)
print(
    "Analytic inputs: Stieltjes Jensen bound, binary Hermite minorant, cubic discriminant and contact factorization. No new quantum compatibility bound is claimed."
)
