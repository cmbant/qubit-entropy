#!/usr/bin/env python3
"""SD-1, SD-2, SD-4: dimensions of the SU(2)^4-invariant polynomials of bidegree (d,d) on four qubits, and of their
qubit-permutation-symmetric and conjugation-even/odd parts.

Statement in invariant_dimensions.tex.

The invariants of bidegree (d,d) are the polynomials z -> <z^{(x)d}, a z^{(x)d}> with a in the commutant of SU(2)^4 on
((C^2)^{(x)d})^{(x)4}, modulo the S_d x S_d symmetry. By Schur-Weyl this commutant is the 4-fold tensor power of
A_d = sum_{lambda in {(d-k,k)}} V_lambda (x) V_lambda^*, so the dimension is a character sum over S_d x S_d (x) S_4 (sites),
and the conjugation psi -> conj(psi) acts by swapping the two S_d factors.

SD-1  (exact) d = 3: 20 invariants, 17 conjugation-even and 3 odd; the qubit-symmetric ones are 5, all even.
SD-2  (exact) independent count by multiplicities: with g_L the multiplicity of the SU(2)^4 irrep L in Sym^d((C^2)^{(x)4}),
      total = sum g^2, odd = sum g(g-1)/2 (odd = conjugation-odd invariants), for d = 1..8, against the swap-trace formula.
SD-4  (exact) the table d = 1..8; dim(qubit-symmetric, conjugation-even) is 5, 16, 30, 85 at d = 3, 4, 5, 6, and the
      degree-six count 85 is the dimension used by the quadratic variance certificate (QG-5).
"""
# log: verify_symmetric_invariant_dimensions.txt
import math
from collections import Counter
from fractions import Fraction
from itertools import product


res = []


def check(tag, ok, msg):
    res.append(bool(ok))
    print(('[PASS] ' if ok else '[FAIL] ') + tag + ' ' + msg)


def partitions(n, maxp=None):
    if maxp is None:
        maxp = n
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxp), 0, -1):
        for p in partitions(n - k, k):
            yield (k,) + p


def class_size(cyc, n):
    c = Counter(cyc)
    den = 1
    for length, m in c.items():
        den *= (length ** m) * math.factorial(m)
    return math.factorial(n) // den


def fixed_ksubsets(cyc, k):
    poly = [0] * (k + 1)
    poly[0] = 1
    for length in cyc:
        new = poly[:]
        for i in range(k + 1 - length):
            new[i + length] += poly[i]
        poly = new
    return poly[k]


def chi(k, cyc):
    """character of the S_d irrep (d-k, k), k <= d/2: fixed k-subsets minus fixed (k-1)-subsets."""
    return fixed_ksubsets(cyc, k) - (fixed_ksubsets(cyc, k - 1) if k > 0 else 0)


def power_cycle(cyc, ell):
    out = []
    for m in cyc:
        g = math.gcd(m, ell)
        out += [m // g] * g
    return tuple(sorted(out, reverse=True))


SITE_PERMS = [((1, 1, 1, 1), 1), ((2, 1, 1), 6), ((2, 2), 3), ((3, 1), 8), ((4,), 6)]


def counts(d, sym):
    classes = [(tuple(p), class_size(p, d)) for p in partitions(d)]
    lams = range(0, d // 2 + 1)
    taus = SITE_PERMS if sym else [((1, 1, 1, 1), 1)]
    ntau = sum(c for _, c in taus)
    fact = math.factorial(d)
    dim = Fraction(0)
    trc = Fraction(0)
    for tcyc, tn in taus:
        for c1, s1 in classes:
            for c2, s2 in classes:
                prod = 1
                for ell in tcyc:
                    prod *= sum(chi(k, power_cycle(c1, ell)) * chi(k, power_cycle(c2, ell)) for k in lams)
                dim += Fraction(tn * s1 * s2 * prod, ntau * fact * fact)
        for c, s in classes:
            prod = 1
            for ell in tcyc:
                if ell % 2 == 1:
                    prod *= sum(chi(k, power_cycle(c, ell)) for k in lams)
                else:
                    cc = power_cycle(c, ell // 2)
                    prod *= sum(chi(k, cc) ** 2 for k in lams)
            trc += Fraction(tn * s * prod, ntau * fact)
    return dim, (dim + trc) / 2, (dim - trc) / 2


TAB = {d: (counts(d, False), counts(d, True)) for d in range(1, 9)}
full3, sym3 = TAB[3]
check('SD-1', full3 == (20, 17, 3) and sym3 == (5, 5, 0),
      'd = 3: invariants 20 = 17 even + 3 odd; qubit-symmetric 5 = 5 even + 0 odd')


# SD-2: independent multiplicity count
def multiplicities(d):
    classes = [(tuple(p), class_size(p, d)) for p in partitions(d)]
    lams = list(range(0, d // 2 + 1))
    g = {}
    for assign in product(lams, repeat=4):
        tot = Fraction(0)
        for c, s in classes:
            pr = 1
            for k in assign:
                pr *= chi(k, c)
            tot += Fraction(s * pr, math.factorial(d))
        g[assign] = tot
    return g


ok2 = True
for d in range(1, 9):
    g = multiplicities(d)
    total = sum(v * v for v in g.values())
    odd = sum(v * (v - 1) / 2 for v in g.values())
    (t, e, o), _ = TAB[d]
    ok2 &= total == t and odd == o and all(v.denominator == 1 for v in g.values())
check('SD-2', ok2, 'for d = 1..8 the swap-trace count agrees with sum g^2 (total) and sum g(g-1)/2 (odd), g the multiplicity of each '
      'SU(2)^4 irrep in Sym^d; at d = 3 the odd count 3 is the three odd invariants at the first odd copy number')

# SD-4
print('    d : full (total, even, odd) | qubit-symmetric (total, even, odd)')
for d in range(1, 9):
    f_, s_ = TAB[d]
    print(f'    {d} : {tuple(int(x) for x in f_)} | {tuple(int(x) for x in s_)}')
check('SD-4', [TAB[d][1][1] for d in (3, 4, 5, 6)] == [5, 16, 30, 85],
      'qubit-symmetric conjugation-even invariants: 5 at d = 3 (n^3, nS1, nP, T, Pi2), 16 at d = 4, 30 at d = 5, '
      '85 at d = 6 (the space of the quadratic variance certificate, QG-5)')
print('All SD checks passed' if all(res) else 'SOME SD CHECK FAILED')
raise SystemExit(0 if all(res) else 1)
