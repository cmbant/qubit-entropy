#!/usr/bin/env python3
"""ESH: the coefficient 7/8 of the general envelope (GE) is optimal.

Proved in: sextic_identity.tex (thm:fqe-general-envelope and the sentence after it).
On z = c|0000> + s|1111> (c, s real, c^2 + s^2 = 1) compute exactly, from the tensor definitions,
tau = |z^T sigma_y^{x4} z|^2, W1, P, T and F = 17P - 12T - 12, and the slack of
F >= (3/4 + C tau/(1 + sqrt tau)) W1 + tau - tau^{3/2}.  It equals (7/2 - 4C) tau + (1 + 4C) tau^{3/2},
negative for small tau > 0 whenever C > 7/8.
"""
import itertools
import sympy as sp

c, s, C, t = sp.symbols('c s C t', positive=True)
z = sp.zeros(16, 1)
z[0], z[15] = c, s
sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SY = sy
for _ in range(3):
    SY = sp.kronecker_product(SY, sy)
tau = sp.simplify(sp.Abs((z.T * SY * z)[0])**2)
paulis = [sp.Matrix([[0, 1], [1, 0]]), sy, sp.Matrix([[1, 0], [0, -1]])]
I2 = sp.eye(2)


def site_op(op, i):
    m = sp.Matrix([[1]])
    for k in range(4):
        m = sp.kronecker_product(m, op if k == i else I2)
    return m


W1 = sp.simplify(sum(((z.H * site_op(p, i) * z)[0])**2 for i in range(4) for p in paulis))
T4 = sp.MutableDenseNDimArray([z[i] for i in range(16)], (2, 2, 2, 2))


def cut(j):
    keep = [0, j]; rest = [k for k in range(1, 4) if k != j]
    M = sp.zeros(4, 4)
    for a in itertools.product(range(2), repeat=4):
        row = 2 * a[keep[0]] + a[keep[1]]
        for b in itertools.product(range(2), repeat=4):
            if all(a[k] == b[k] for k in rest):
                M[row, 2 * b[keep[0]] + b[keep[1]]] += T4[a] * sp.conjugate(T4[b])
    return M


P = sp.simplify(sum((cut(j)**2).trace() for j in (1, 2, 3)))
T = sp.simplify(sum((cut(j)**3).trace() for j in (1, 2, 3)))
unit = {s: sp.sqrt(1 - c**2)}
F = sp.simplify((17 * P - 12 * T - 12).subs(unit))
tau_u = sp.simplify(tau.subs(unit)); W1_u = sp.simplify(W1.subs(unit))
assert sp.simplify(tau_u - 4 * c**2 * (1 - c**2)) == 0
assert sp.simplify(W1_u - 4 * (1 - tau_u)) == 0
assert sp.simplify(F - (3 + sp.Rational(3, 2) * tau_u)) == 0
print('ESH-1 GHZ family: tau = 4c^2s^2, W1 = 4(1 - tau), F = 3 + 3tau/2 (exact, from the tensor definitions): PASS')
slack = (3 + sp.Rational(3, 2) * t) - ((sp.Rational(3, 4) + C * t / (1 + sp.sqrt(t))) * 4 * (1 - t) + t - t**sp.Rational(3, 2))
target = (sp.Rational(7, 2) - 4 * C) * t + (1 + 4 * C) * t**sp.Rational(3, 2)
assert sp.simplify(slack - target) == 0
print('ESH-2 slack of the envelope with coefficient C: (7/2 - 4C) tau + (1 + 4C) tau^(3/2): PASS')
assert sp.simplify(target.subs(C, sp.Rational(7, 8)) - sp.Rational(9, 2) * t**sp.Rational(3, 2)) == 0
C1 = sp.Rational(7, 8) + sp.Rational(1, 1000)
t1 = sp.Rational(1, 10**8)
assert target.subs({C: C1, t: t1}) < 0
print('ESH-3 C = 7/8 leaves slack (9/2) tau^(3/2); C = 7/8 + 1/1000 fails at tau = 10^-8 (exact): PASS')
print('ALL ESH CHECKS PASSED')
