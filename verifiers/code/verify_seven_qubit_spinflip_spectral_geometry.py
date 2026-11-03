#!/usr/bin/env python3
"""SP7-1: exact tightness of the spin-flip spectral pairing.

Proof home: seven_qubit_entropy_compatibility.tex, prop:seven-spinflip-spectral-pair.
The trace-pairing inequality b >= 4v is an analytic rearrangement inequality; this script
checks that the depolarized pure three-qubit (one-spike) spectrum attains it, in exact rationals.
"""
from fractions import Fraction as Q

for s in (Q(1, 3), Q(1, 5), Q(2, 7)):
    e = 7 * s * s; b = 3 * s * s
    lam = [(1 + 7 * s) / 8] + [(1 - s) / 8] * 7
    v = sum((lam[j] + lam[7 - j] - Q(1, 4)) ** 2 for j in range(4))
    assert b == 4 * v
    assert (1 - e + 2 * b) / 8 == sum(lam[j] * lam[7 - j] for j in range(8))
print("PASS SP7-1: exact tightness of spin-flip pairing at one-spike spectra (s = 1/3, 1/5, 2/7)")
