#!/usr/bin/env python3
"""EG-1: how close the seven-qubit purity minimiser and entropy maximiser are.

On the icosahedral family psi_x, with u=(1-2x)/30, the 35 three-qubit marginals have
  20 avoiding s:   (4u)^6, (3u+x)^2
  15 containing s: 8u, (4u)^5, (u+x)^2
(thm:ico; these give E_3(psi_x) of seven_qubit_results.tex).  The entropy maximiser x_* is the
unique root in (0,1/2) of g(x)=32 ln2+15 ln(1-2x)-8 ln3-8 ln(1+8x)-7 ln(1+28x), which is strictly
decreasing (prop:entropy-seven-stationarity).  This script
  * checks exactly that the spectra are normalised and that pi_3(psi_{37/524}) = 121/917;
  * encloses x_* by interval bisection with certified signs of g at both ends;
  * encloses E_3(psi_{x_*}), E_3(Psi_ico), their difference, and pi_3(psi_{x_*}) - 121/917,
    all in mpmath interval arithmetic (outward rounding).
"""
from fractions import Fraction as Fr
from mpmath import iv, mp, mpf, nstr

mp.dps = iv.dps = 40


def spectra(x, one):
    u = (one - 2 * x) / 30
    avoid = [4 * u] * 6 + [3 * u + x] * 2
    contain = [8 * u] + [4 * u] * 5 + [u + x] * 2
    return avoid, contain


def mean(fun, x, one):
    a, c = spectra(x, one)
    return (20 * sum(fun(t) for t in a) + 15 * sum(fun(t) for t in c)) / 35


# exact checks at the purity minimiser
xi = Fr(37, 524)
a, c = spectra(xi, Fr(1))
assert sum(a) == 1 and sum(c) == 1
assert mean(lambda t: t * t, xi, Fr(1)) == Fr(121, 917)
print("PASS EG-1: spectra normalised; pi_3(psi_{37/524}) = 121/917 exactly")

LN2 = iv.log(2)


def g(x):
    return 32 * LN2 + 15 * iv.log(1 - 2 * x) - 8 * iv.log(3) - 8 * iv.log(1 + 8 * x) - 7 * iv.log(1 + 28 * x)


lo, hi = mpf("0.06"), mpf("0.08")
assert g(iv.mpf(lo)).a > 0 and g(iv.mpf(hi)).b < 0
for _ in range(110):
    mid = (lo + hi) / 2
    v = g(iv.mpf(mid))
    if v.a > 0:
        lo = mid
    elif v.b < 0:
        hi = mid
    else:
        break
assert g(iv.mpf(lo)).a > 0 and g(iv.mpf(hi)).b < 0
X = iv.mpf([lo, hi])
print(f"PASS EG-1: x_* in [{nstr(lo, 25)}, {nstr(hi, 25)}] (certified signs of g)")


def ent(t):
    return -t * iv.log(t) / LN2


Estar = mean(ent, X, iv.mpf(1))
Eico = mean(ent, iv.mpf(37) / 524, iv.mpf(1))
gap = Estar - Eico
dpur = mean(lambda t: t * t, X, iv.mpf(1)) - iv.mpf(121) / 917
print("E_3(psi_x*)        in", iv.nstr(Estar, 18))
print("E_3(Psi_ico)       in", iv.nstr(Eico, 18))
print("entropy gap        in", iv.nstr(gap, 12))
print("pi_3(psi_x*)-121/917 in", iv.nstr(dpur, 12))
assert 7.15e-6 < gap.a and gap.b < 7.17e-6
assert 1.21e-6 < dpur.a and dpur.b < 1.23e-6
assert 2.9653560294776 < Estar.a and Estar.b < 2.9653560294777
print("PASS EG-1: 7.15e-6 < E_3(psi_x*) - E_3(Psi_ico) < 7.17e-6 bits; "
      "1.21e-6 < pi_3(psi_x*) - 121/917 < 1.23e-6")
