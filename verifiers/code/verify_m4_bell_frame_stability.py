#!/usr/bin/env python3
"""BF-1--BF-4: exact algebra for the constructive Bell certificate.

Checks full determinant/gradient identities, magic Pauli normalization,
compatible-plane geometry and Bell/HS representatives. The superseded
spectral-gap and cubic-transfer checks are removed with their proof route.
The all-state orbit coefficient is checked independently by GO.
"""
from itertools import combinations, permutations
import sympy as s

I, R = s.I, s.Rational


def zero(expr):
    return s.expand(expr) == 0


def norm2(mat):
    return s.expand(sum(v * s.conjugate(v) for v in mat))


def check(tag, message):
    print('[PASS] ' + tag + ' ' + message, flush=True)


def flatten(z, j):
    out = s.zeros(4)
    rest = [k for k in (1, 2, 3) if k != j]
    for q in range(16):
        b = [(q >> (3 - k)) & 1 for k in range(4)]
        out[2*b[0] + b[j], 2*b[rest[0]] + b[rest[1]]] = z[q]
    return out


def det4(mat):
    out = 0
    for p in permutations(range(4)):
        inversions = sum(p[i] > p[j] for i, j in combinations(range(4), 2))
        out += (-1)**inversions * s.prod(mat[i, p[i]] for i in range(4))
    return s.expand(out)


magic = s.Matrix([[1, 0, 0, I], [0, I, 1, 0],
                  [0, I, -1, 0], [1, 0, 0, -I]]) / s.sqrt(2)
T = s.kronecker_product(magic, magic)
z = s.symbols('z:16')
Z = s.Matrix(4, 4, z)
comp = T * s.Matrix(z)
ss = s.trace(Z.T * Z)
J = s.trace((Z.T * Z)**2) / 4 - ss**2 / 8
D = det4(Z)
cut_dets = [det4(flatten(comp, j)) for j in (1, 2, 3)]
assert zero(cut_dets[0] - D)
assert zero(cut_dets[1] - (D + J)/2)
assert zero(cut_dets[2] - (-D + J)/2)
N = Z * Z.T * Z - ss * Z / 2
assert all(zero(s.diff(J, t) - v) for t, v in zip(z, N))
u, v, ub, vb = s.symbols('u v ub vb')
assert zero(u*ub + (u+v)*(ub+vb)/4 + (-u+v)*(-ub+vb)/4
            - R(3, 2)*u*ub - R(1, 2)*v*vb)
check('BF-1', 'all 16-variable quartic determinant coefficients and cubic gradients; '
      'C = (3/2)||adj Z||^2 + (1/2)||ZZ^T Z-sZ/2||^2')

pauli = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -I], [I, 0]]), s.diag(1, -1)]
cs = s.symbols('c:6', real=True)
A = s.zeros(4)
for value, (i, j) in zip(cs, combinations(range(4), 2)):
    A[i, j], A[j, i] = value, -value
ops = [s.simplify(magic.H * q * magic) for p in pauli
       for q in (s.kronecker_product(p, s.eye(2)), s.kronecker_product(s.eye(2), p))]
assert all(q == -s.conjugate(q) and q.T == -q for q in ops)
assert zero(sum(s.trace(I*A*q)**2 for q in ops) - 4*norm2(A))
check('BF-2', 'magic Pauli basis and arbitrary skew-matrix normalization for W1')

c12, c13, c23, lam, E = s.symbols('c12 c13 c23 lam E', real=True)
G = s.Matrix([[1, c12, c13], [c12, 1, c23], [c13, c23, 1]])
deltas = sum(R(1, 4)-c*c for c in (c12, c13, c23))
assert zero(deltas - (R(9, 2)-s.trace(G*G))/2)
assert zero((3-lam)**2/2+lam*lam - (R(9, 2)-3*lam) - R(3, 2)*lam*lam)
c = s.symbols('c', real=True)
Delta = R(1, 4)-c*c
comparison = 1-R(7, 4)*Delta
positive = (1-2*c)*(13+c*(525-462*c)+588*c**3*(1-2*c))/256
assert zero(1+c-3*(1-c)*comparison**2-positive)
assert comparison.subs(c, 0) > 0
# 0<=c<=1/2: every displayed factor in positive is nonnegative.
assert 525-462*R(1, 2) > 0
assert 4*R(1, 3)+R(7, 8) == R(53, 24)
assert R(14, 3) < (R(9, 2)-R(53, 24))**2
check('BF-3', 'Gram deficit, exact compatible-plane cosine sqrt((1+c)/(3(1-c))), '
      'null-ray correction <=7Delta/4; Bell-frame constant 9/2')

def computational_bell(pairing):
    out = s.zeros(16, 1)
    for i in range(16):
        bits = [(i >> (3-k)) & 1 for k in range(4)]
        if all(bits[a] != bits[b] for a, b in pairing):
            out[i] = (-1)**sum(bits[a] for a, b in pairing)*R(1, 2)
    return out


pairings = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
bells = [s.simplify(T.H*computational_bell(p)) for p in pairings]
for i, j in combinations(range(3), 2):
    assert abs((bells[i].H*bells[j])[0]) == R(1, 2)
assert s.Matrix.hstack(*bells).rank() == 2
inner = (bells[0].T*bells[1])[0]
plane_y = (bells[1]-inner*bells[0])/s.sqrt(1-inner**2)
plane_null = (bells[0]+I*plane_y)/s.sqrt(2)
assert s.simplify((plane_null.T*plane_null)[0]) == 0
assert all(s.simplify(1-2*abs((bell.H*plane_null)[0])**2) == 0 for bell in bells)
plane_P = plane_T = 0
for j in (1, 2, 3):
    A = flatten(T*plane_null, j)
    rho = s.simplify(A*A.H)
    plane_P += s.trace(rho*rho)
    plane_T += s.trace(rho*rho*rho)
assert s.simplify(18*plane_P-12*plane_T-13) == 0
for bell in bells[1:]:
    B = s.Matrix(4, 4, list(bell))
    assert B == s.conjugate(B) and 4*B*B.T == s.eye(4)
assert sorted(s.simplify((2*s.Matrix(4, 4, list(bell))).det()) for bell in bells[1:]) == [-1, 1]
star = s.diag(1/s.sqrt(2), I/s.sqrt(6), I/s.sqrt(6), I/s.sqrt(6))
cs_star = T*s.Matrix(list(star))
pol = tr3 = Csum = 0
for j in (1, 2, 3):
    A = flatten(cs_star, j)
    rho = s.simplify(A*A.H)
    assert rho.eigenvals() == {R(1, 2): 1, R(1, 6): 3}
    pol += s.trace(rho*rho)
    tr3 += s.trace(rho*rho*rho)
assert s.simplify(pol) == 1 and s.simplify(tr3) == R(5, 12)
assert s.simplify(18*pol-12*tr3-13) == 0
assert abs(star[0, 0])**2 == R(1, 2)  # attainable leading Bell-product overlap
check('BF-4', 'standard Bell products form a trine, alternative-pairing '
      'magic matrices are O/2, HS null plane has Q=0 and Bell overlap^2=1/2')
print('All BF checks passed (exact); analytic inequalities are proved in the paper.')
