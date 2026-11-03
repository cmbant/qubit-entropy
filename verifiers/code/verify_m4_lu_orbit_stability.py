#!/usr/bin/env python3
"""OS-1--OS-3: exact local algebra for lu_orbit_stability.tex.

OS-1: full 32-real-variable Taylor form from direct flattenings, block formula,
      spectrum and radial/orbit kernel.
OS-2: explicit local radii and entropy coefficient algebra.
OS-3: HS, conjugate, Bell-product, cluster and normal-direction regressions.
The global bounds of the orbit-stability theorem are checked by
verify_m4_global_overlap.py (GO-1--GO-4); their all-parameter coverage and
positive-root arguments are analytic in the paper.
"""
from itertools import combinations
import sympy as sp

I, R = sp.I, sp.Rational


def zero(expr):
    return sp.expand(expr) == 0


def norm_sq(mat):
    return sp.expand(sum(z * sp.conjugate(z) for z in mat))


def check(tag, condition, message):
    assert condition, tag + ': ' + message
    print('[PASS] ' + tag + ' ' + message, flush=True)


def flatten(z, j):
    out = sp.zeros(4)
    rest = [k for k in (1, 2, 3) if k != j]
    for q in range(16):
        bits = [(q >> (3 - k)) & 1 for k in range(4)]
        out[2 * bits[0] + bits[j], 2 * bits[rest[0]] + bits[rest[1]]] = z[q]
    return out


magic = sp.Matrix([[1, 0, 0, I], [0, I, 1, 0],
                   [0, I, -1, 0], [1, 0, 0, -I]]) / sp.sqrt(2)
K = sp.kronecker_product(magic, magic)
a, b = 1 / sp.sqrt(2), 1 / sp.sqrt(6)
star = sp.diag(a, I * b, I * b, I * b)
zstar = K * sp.Matrix(list(star))
x = sp.Matrix(sp.symbols('x:32', real=True))
H = sp.Matrix(4, 4, lambda i, j: x[4 * i + j] + I * x[16 + 4 * i + j])
zh = K * sp.Matrix(list(H))
P0 = P1 = P2 = T0 = T1 = T2 = 0
for j in (1, 2, 3):
    C, B = flatten(zstar, j), flatten(zh, j)
    rho = sp.simplify(C * C.H)
    L, V = C * B.H + B * C.H, B * B.H
    P0 += sp.trace(rho * rho)
    P1 += 2 * sp.trace(rho * L)
    P2 += sp.trace(L * L + 2 * rho * V)
    T0 += sp.trace(rho * rho * rho)
    T1 += 3 * sp.trace(rho * rho * L)
    T2 += 3 * sp.trace(rho * rho * V + rho * L * L)
n1 = 2 * sp.re((zstar.H * zh)[0])
assert sp.simplify(P0) == 1 and sp.simplify(T0) == R(5, 12)
assert zero(P1 - 2 * n1) and zero(T1 - R(5, 4) * n1)
quad = sp.expand(18 * P2 - 12 * T2 - 21 * sum(t * t for t in x) - 3 * n1**2)
lower = H[1:, 1:]
sym, skew = (lower + lower.T) / 2, (lower - lower.T) / 2
block = 6 * (norm_sq(sp.Matrix([H[0, 0]])) + norm_sq(sym)
             - norm_sq(sp.Matrix([a * H[0, 0] - I * b * sp.trace(sym)])))
block += 20 * norm_sq(sp.re(skew))
block += 30 * sum((a * sp.im(H[0, j]) + b * sp.re(H[j, 0]))**2
                  + (b * sp.re(H[0, j]) + a * sp.im(H[j, 0]))**2
                  for j in (1, 2, 3))
assert zero(quad - block)
operator = sp.simplify(sp.hessian(quad, list(x)) / 2)
assert sp.simplify(operator * (operator - 6 * sp.eye(32))
                   * (operator - 20 * sp.eye(32))) == sp.zeros(32)
assert operator.eigenvals() == {0: 11, 6: 12, 20: 9}


def real_vector(mat):
    return sp.Matrix([sp.re(t) for t in mat] + [sp.im(t) for t in mat])


tangents = [real_vector(I * star)]
for i, j in combinations(range(4), 2):
    rotation = sp.zeros(4)
    rotation[i, j], rotation[j, i] = 1, -1
    tangents += [real_vector(rotation * star), real_vector(star * rotation.T)]
orbit = sp.Matrix.hstack(*tangents)
kernel = sp.Matrix.hstack(real_vector(star), orbit)
assert orbit.rank() == 10 and kernel.rank() == 11
assert sp.simplify(operator * kernel) == sp.zeros(32, kernel.cols)
check('OS-1', True, 'full quadratic Taylor form equals block formula; '
      'quadratic eigenvalues 0^11, 6^12, 20^9; kernel = radial + 10 orbit tangents')

def local_margin(radius):
    return 6 - R(3, 2) * radius**2 - 103 * radius * (20 + 15 * radius + 6 * radius**2 + radius**3)


assert 18 * 3 + 12 * 3 + 13 == 103
assert [sp.binomial(6, i) for i in range(3, 7)] == [20, 15, 6, 1]
assert local_margin(R(1, 1024)) > 3
assert sum(R(7, 10)**i / sp.factorial(i) for i in range(4)) > 2
ln3_lower = 1 + R(1, 12) + R(1, 80)
assert ln3_lower == R(263, 240)
assert (ln3_lower - 1) / (2 * R(7, 10)) > R(1, 15)
assert 9 - R(15, 2) * R(10, 9) > 0  # alpha > beta
check('OS-2', True, 'local sextic Taylor radius 2^-10; coarse k>1/15, alpha>beta (the sharper k>16/225 is GO-4; alpha>6beta follows from ln3<11/10 in the text); exact rational inequalities')


def moments(z):
    n = sp.simplify(norm_sq(z))
    P = T = 0
    for j in (1, 2, 3):
        C = flatten(z, j)
        rho = sp.simplify(C * C.H)
        P += sp.trace(rho**2)
        T += sp.trace(rho**3)
    P, T = sp.simplify(P), sp.simplify(T)
    return P, T, sp.simplify(17*n*P - 12*T - 12*n**3), sp.simplify(18*n*P - 12*T - 13*n**3)


bell = sp.Matrix([R(1, 2) if i in (0, 3, 12, 15) else 0 for i in range(16)])
cluster = bell.copy()
cluster[15] = -R(1, 2)
assert moments(zstar) == (1, R(5, 12), 0, 0)
assert moments(sp.conjugate(zstar)) == (1, R(5, 12), 0, 0)
assert moments(bell) == (R(3, 2), R(9, 8), 0, R(1, 2))
assert moments(cluster) == (1, R(3, 8), R(1, 2), R(1, 2))
normal = sp.zeros(4)
normal[1, 1], normal[2, 2] = 1 / sp.sqrt(2), -1 / sp.sqrt(2)
normal_real = real_vector(normal)
assert (normal_real.T * operator * normal_real)[0] == 6
assert sp.simplify(orbit.T * normal_real) == sp.zeros(orbit.cols, 1)
check('OS-3', True, 'Q=0 at both HS representatives; Bell: F=0,Q=1/2; '
      'cluster: P=1,F=Q=1/2; unit normal has quadratic coefficient 6')
print('All OS checks passed (exact).')
