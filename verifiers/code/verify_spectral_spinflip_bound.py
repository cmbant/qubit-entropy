#!/usr/bin/env python3
"""SCB-1--SCB-5: exact algebra for spectral_spinflip_bound.tex.

The arbitrary-dimensional positivity, trace rearrangement, and inference
of the spectral bound are analytic. This script checks the generic
calibration identity in dimensions 2 through 6, the trace-gap algebra,
the generic two-qubit spin-flip identity, and physical sharpness families.
No numerical eigensolver or fitted certificate is used.
"""

from itertools import permutations

import sympy as s


def norm2(z):
    return s.expand(z * s.conjugate(z))


def frobenius2(matrix):
    return s.expand(sum(norm2(z) for z in matrix))


for n in range(2, 7):
    sig = s.symbols(f"sig0:{n}", real=True)
    lam = [z**2 for z in sig]
    total = sum(lam)
    r = sum(lam[1:])
    d = lam[0] - r
    x = s.symbols(f"x0:{n*n}", real=True)
    y = s.symbols(f"y0:{n*n}", real=True)
    w = s.Matrix(n, n, [a + s.I * b for a, b in zip(x, y)])
    q = sum(sig[i] * w[i, i] for i in range(n))
    t = q - sig[0] * w[0, 0]
    cost = sum((lam[i] + lam[j]) * norm2(w[i, j]) for i in range(n) for j in range(n))
    bform = total * frobenius2(w) + norm2(q) - cost
    weights = [s.S.Zero] + [r - z for z in lam[1:]]
    weighted = sum(
        (weights[i] + weights[j]) * norm2(w[i, j]) for i in range(n) for j in range(n)
    )
    tailnorm = frobenius2(w[1:, 1:])
    remainder = d * (r * tailnorm - norm2(t)) + norm2(r * w[0, 0] + sig[0] * t)
    assert s.expand(r * (bform - weighted) - remainder) == 0
    product = {sig[i]: 0 for i in range(1, n)}
    assert s.expand(bform.subs(product) - sig[0] ** 2 * tailnorm) == 0
print(
    "SCB-1 PASS: generic complex-matrix calibration in dimensions 2--6; rank-one limit"
)

# Telescoping proof of the trace-rearrangement gap. The sign is supplied
# analytically by the increasing weights and the Ky Fan partial-sum bound.
n = 5
weights = s.symbols(f"s0:{n}")
lam = s.symbols(f"l0:{n}")
diag = list(s.symbols(f"a0:{n-1}"))
diag.append(sum(lam) - sum(diag))
gap = sum(weights[i] * (diag[i] - lam[i]) for i in range(n))
telescoping = sum(
    (weights[k + 1] - weights[k]) * sum(lam[i] - diag[i] for i in range(k + 1))
    for k in range(n - 1)
)
assert s.expand(gap - telescoping) == 0
for spec in [(12, 4, 3, 0), (4, 3, 1, 1), (4, 3), (1, 0, 0, 0)]:
    vals = [s.Integer(z) ** 2 for z in spec]
    total, r = sum(vals), sum(vals[1:])
    assert vals[0] >= r
    ws = [s.S.Zero] + [r - z for z in vals[1:]]
    assert all(ws[i] <= ws[i + 1] for i in range(len(ws) - 1))
    minimum = sum(a * b for a, b in zip(ws, vals))
    assert minimum == r**2 - sum(z**2 for z in vals[1:])
    assert all(
        sum(a * b for a, b in zip(ws, perm)) >= minimum for perm in permutations(vals)
    )
    # Exact complex unitary tests use two independently chosen Householders.
    n = len(spec)
    u = s.Matrix([1 + s.I * j for j in range(n)])
    v = s.Matrix([j + 1 + s.I for j in range(n)])
    hu = s.eye(n) - 2 * u * u.conjugate().T / (u.conjugate().T * u)[0]
    hv = s.eye(n) - 2 * v * v.conjugate().T / (v.conjugate().T * v)[0]
    phase = s.diag(*[s.I**j for j in range(n)])
    hu, hv = s.simplify(phase * hu), s.simplify(hv * phase)
    assert s.simplify(hu * hu.conjugate().T - s.eye(n)) == s.zeros(n)
    assert s.simplify(hv * hv.conjugate().T - s.eye(n)) == s.zeros(n)
    dd = s.diag(*spec)
    ww = s.simplify(hu * dd * hv)
    ll = s.diag(*vals)
    bform = (
        total * frobenius2(ww)
        + norm2(s.trace(dd * ww))
        - s.trace(ll * (ww * ww.conjugate().T + ww.conjugate().T * ww))
    )
    assert s.simplify(bform - 2 * minimum) >= 0
    opposite = s.diag(spec[0], *[-z for z in spec[1:]])
    assert s.expand(norm2(s.trace(dd * opposite)) - (vals[0] - r) ** 2) == 0
print(
    "SCB-2 PASS: trace-gap identity, exact unitary regressions, and Schmidt sign-flip sharpness"
)

pauli = [
    s.Matrix([[0, 1], [1, 0]]),
    s.Matrix([[0, -s.I], [s.I, 0]]),
    s.diag(1, -1),
]
identity = s.eye(2)
ra = s.symbols("ra0:3", real=True)
rb = s.symbols("rb0:3", real=True)
corr = s.symbols("c0:9", real=True)
rho = s.eye(4)
for i in range(3):
    rho += ra[i] * s.kronecker_product(pauli[i], identity)
    rho += rb[i] * s.kronecker_product(identity, pauli[i])
    for j in range(3):
        rho += corr[3 * i + j] * s.kronecker_product(pauli[i], pauli[j])
rho /= 4
yy = s.kronecker_product(pauli[1], pauli[1])
diff = s.simplify(rho - yy * rho.T * yy)
assert s.expand(s.trace(diff * diff) - sum(z**2 for z in ra + rb)) == 0
print(
    "SCB-3 PASS: generic two-qubit Pauli spin-flip mismatch equals the two Bloch norms"
)

# Unnormalised physical family (a|00>+b|11>) tensor (|00>+|11>).
a, b = s.symbols("a b", real=True)
state = s.zeros(16, 1)
state[0] = state[3] = a
state[12] = state[15] = b
total = 2 * (a**2 + b**2)
spin = s.kronecker_product(*([pauli[1]] * 4))
tau = norm2((state.T * spin * state)[0])
bloch = 0
for i in range(4):
    for p in pauli:
        factors = [identity] * 4
        factors[i] = p
        val = (state.conjugate().T * s.kronecker_product(*factors) * state)[0]
        bloch += norm2(val)
assert s.expand(tau - 16 * a**2 * b**2) == 0
assert s.expand(bloch - 8 * (a**2 - b**2) ** 2) == 0
assert s.expand(tau + bloch / 2 - total**2) == 0
assert (tau + s.Rational(49, 100) * bloch - total**2).subs({a: 1, b: 0}) < 0
assert (s.Rational(99, 100) * tau + bloch / 2 - total**2).subs({a: 1, b: 1}) < 0
print(
    "SCB-4 PASS: physical sharpness family and negative tests for both smaller coefficients"
)

# These are classical spectra, not compatible physical marginals. They
# satisfy the new per-cut cap with W1=0, tau=2(P-1), as well as F>=0,
# and the critical envelope F>=tau-tau^(3/2), while violating QK.
# Thus these scalar inputs alone cannot replace QK.
spectra = [
    [s.Rational(z, 6000) for z in (3258, 1692, 525, 525)],
    [s.Rational(z, 6000) for z in (3258, 914, 914, 914)],
    [s.Rational(1, 4)] * 4,
]
assert all(sum(row) == 1 and min(row) >= 0 for row in spectra)
purities = [sum(z**2 for z in row) for row in spectra]
purity_sum = sum(purities)
third = sum(z**3 for row in spectra for z in row)
fvalue = 17 * purity_sum - 12 * third - 12
variance = sum(p**2 for p in purities) - purity_sum**2 / 3
assert purity_sum - 1 == s.Rational(4981, 1200000)
assert fvalue == s.Rational(196938001, 3000000000) > 0
assigned_tau = 2 * (purity_sum - 1)
assert 0 < assigned_tau < 1 and fvalue > assigned_tau
assert all(max(0, 2 * max(row) - 1) ** 2 <= 2 * (purity_sum - 1) for row in spectra)
qk = fvalue**2 + s.Rational(9, 2) * (purity_sum - 1) - 6 * variance
assert qk == -s.Rational(43512669144347111, 10**18) < 0
print(
    "SCB-5 PASS: exact classical counterexample satisfies the caps and critical envelope but has QK residual",
    qk,
)
