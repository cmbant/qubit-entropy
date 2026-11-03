#!/usr/bin/env python3
"""ST-9: the GHZ_{n-2} (x) Bell witness showing c_n is optimal in the odd-n tradeoff.

Proved in: spinflip_tradeoff.tex, remark after thm:odd-spinflip-tradeoff. For Psi_n = GHZ_{n-2} (x) Phi+, n = 2m+1,
computes in exact arithmetic, for odd n = 5..43:
  * the Pauli weight enumerator as the product of the GHZ_{n-2} enumerator (even-weight Z words C(L,j), and
    2^{L-1} full-support X/Y words) and the Bell-pair enumerator 1 + 3t^2: the only odd weights are
    A_{n-2} = 2^{n-3} and A_n = 3*2^{n-3};
  * O = sum_{odd j} (n-j) A_j / (n 2^{n-1}) = 1/(2n) > 0 (Lemma lem:odd-sector);
  * pi_m independently from the subsystem-purity generating polynomials
    Q_GHZ_L(t) = 1 + t^L + (1/2) sum_{0<j<L} C(L,j) t^j and Q_Bell(t) = 1 + t + t^2;
  * pi_m + c_n O = 1/2 exactly, with c_5 = 1 and c_n = (n+1)/4 for n >= 7.
Hence any coefficient below c_n fails at Psi_n.  Negative control: c_n - 1/100 gives a value below 1/2.
"""
from fractions import Fraction
from math import comb


def weights(n):
    L = n - 2
    ghz = [comb(L, j) if j % 2 == 0 else 0 for j in range(L + 1)]
    ghz[L] += 2 ** (L - 1)
    a = [0] * (n + 1)
    for j, v in enumerate(ghz):
        a[j] += v
        a[j + 2] += 3 * v
    return a


def balanced_purity(n):
    m, L = (n - 1) // 2, n - 2
    qg = [Fraction(1) if j in (0, L) else Fraction(comb(L, j), 2) for j in range(L + 1)]
    total = sum(qg[m - i] for i in range(3) if 0 <= m - i <= L)  # Q_Bell coefficients 1, 1, 1
    return total / comb(n, m)


for n in range(5, 44, 2):
    a = weights(n)
    assert sum(a) == 2 ** n
    odd = {j: v for j, v in enumerate(a) if j % 2 and v}
    assert odd == {n - 2: 2 ** (n - 3), n: 3 * 2 ** (n - 3)}, (n, odd)
    O = sum(Fraction((n - j) * v, n * 2 ** (n - 1)) for j, v in odd.items())
    assert O == Fraction(1, 2 * n)
    c = Fraction(1) if n == 5 else Fraction(n + 1, 4)
    pi = balanced_purity(n)
    assert pi + c * O == Fraction(1, 2), n
    assert pi + (c - Fraction(1, 100)) * O < Fraction(1, 2)
    if n <= 9:
        print(f"n={n}: odd weights {odd}, O = {O}, pi_m = {pi}, pi_m + c_n O = {pi + c * O}")
print("ST-9 PASS: Psi_n = GHZ_{n-2} (x) Bell attains pi_m + c_n O = 1/2 with O = 1/(2n) > 0, odd n = 5..43;"
      " negative control c_n - 1/100 fails")
