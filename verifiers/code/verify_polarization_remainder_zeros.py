#!/usr/bin/env python3
"""PZ-1--PZ-2: algebra for the analytic remainder-zero proof.

Proved in: sup_general_envelope.tex (supplement), lem:fqe-polarization-remainder-zeros and
its use at tau = 0 in thm:fqe-general-envelope. The support and orbit classifications are analytic.
"""

from itertools import combinations, product

import sympy as s

I = s.I
pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -I], [I, 0]]), s.diag(1, -1)]
identity = s.eye(2)
epsilon = s.Matrix([[0, 1], [-1, 0]])
rr = s.symbols("rx ry rz", real=True)
bloch = sum((v * p for v, p in zip(rr, pauli)), s.zeros(2))
assert s.simplify(bloch * bloch - sum(v**2 for v in rr) * identity) == s.zeros(2)

ti, tj = s.symbols("ti tj", positive=True)
for bits in product(range(2), repeat=4):
    if len(set(bits)) == 1:
        for i, j in combinations(range(4), 2):
            assert (-1) ** bits[i] * ti - (-1) ** bits[j] * tj == (-1) ** bits[i] * (
                ti - tj
            )
    else:
        i = next(i for i in range(4) if bits[i] == 0)
        j = next(j for j in range(4) if bits[j] == 1)
        assert (-1) ** bits[i] * ti - (-1) ** bits[j] * tj == ti + tj
        assert (ti + tj).is_positive
print(
    "PZ-1 PASS: Bloch-operator invertibility and exact positive-Bloch support exclusion"
)

A, B = s.symbols("A B", nonnegative=True)
norm = A + B
P, T, W1, tau = 3 * (A**2 + B**2), 3 * (A**3 + B**3), 4 * (A - B) ** 2, 4 * A * B
F = 17 * norm * P - 12 * T - 12 * norm**3
H = F - s.Rational(3, 4) * norm * W1
assert s.expand(H - 18 * norm * A * B) == 0
assert s.expand(H - s.Rational(9, 2) * norm * tau) == 0
print("PZ-2 PASS: generic GHZ H=(9/2)N tau")
