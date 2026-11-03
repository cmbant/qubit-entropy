#!/usr/bin/env python3
"""NC: exact two-Bloch-sphere geometry for null critical four-qubit states.

Checks the auxiliary product reduction, all 32 HS cube pairs, moment formula,
scalar remainders, cluster equality, three-cut fourth moments and the positive entropy-remainder kernel.
The unit-sphere inequalities and signed-Birkhoff orbit reduction are analytic
in critical_orbit_stability.tex; no numerical optimization is a proof input.
"""
from itertools import product
import sympy as s

R = s.Rational
M = s.Matrix([[1, 0, 0, s.I], [0, s.I, 1, 0],
              [0, s.I, -1, 0], [1, 0, 0, -s.I]])/s.sqrt(2)
assert M.H*M == s.eye(4)
z = s.Matrix(s.symbols('z0:4'))
assert s.expand(s.Matrix(2, 2, list(M*z)).det()-sum(v*v for v in z)/2) == 0
print('[PASS] NC-1 auxiliary magic transform is unitary; determinant=sum(v_j^2)/2')
paulis = [s.Matrix([[0, 1], [1, 0]]),
          s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
frame, cube_pairs = [], set()
for i in range(4):
    for signs in product((-1, 1), repeat=3):
        eta = s.zeros(4, 1)
        eta[i] = 1/s.sqrt(2)
        for j, sign in zip([j for j in range(4) if j != i], signs):
            eta[j] = sign*s.I/s.sqrt(6)
        psi = M*eta
        assert s.simplify(s.Matrix(2, 2, list(psi)).det()) == 0
        r = tuple(s.simplify(s.sqrt(3)*(psi.H*s.kronecker_product(p, s.eye(2))*psi)[0]) for p in paulis)
        t = tuple(s.simplify(s.sqrt(3)*(psi.H*s.kronecker_product(s.eye(2), p)*psi)[0]) for p in paulis)
        assert all(v in (-1, 1) for v in r+t) and s.prod(r+t) == -1
        cube_pairs.add((r, t))
        frame.append(eta)
expected = {(r, t) for r in product((-1, 1), repeat=3)
            for t in product((-1, 1), repeat=3) if s.prod(r+t) == -1}
assert cube_pairs == expected and len(cube_pairs) == 32
print('[PASS] NC-2 all 32 HS rays are exactly the opposite-parity auxiliary cube pairs')
chi = [(1, -1, 1), (1, 1, -1), (-1, -1, -1), (-1, 1, 1)]
for i, signs in enumerate(chi):
    projector = (s.eye(4)+sum((sign*s.kronecker_product(p, p)
                 for sign, p in zip(signs, paulis)), s.zeros(4)))/4
    assert s.simplify(M[:, i]*M[:, i].H-projector) == s.zeros(4)
c = s.symbols('c0:3', real=True)
p = [(1+sum(a*b for a, b in zip(signs, c)))/4 for signs in chi]
assert s.expand(sum(v*v for v in p)-(1+sum(v*v for v in c))/4) == 0
assert s.expand(sum(v**3 for v in p)-(1+3*sum(v*v for v in c)-6*s.prod(c))/16) == 0
Q = 27*sum(v*v for v in p)-36*sum(v**3 for v in p)-4
assert s.expand(Q-R(1, 2)-R(27, 2)*s.prod(c)) == 0
print('[PASS] NC-3 all Bell projectors and the exact Q=(1+27 prod(r_j s_j))/2 identity')
a, b, c = s.symbols('a b c', nonnegative=True)
A, B = 1-1/s.sqrt(3), 1+1/s.sqrt(3)
f0 = (2+s.sqrt(3))/6
fg = (1+(a+b+c)/s.sqrt(3))*(1+(a+b-c)/s.sqrt(3))
rem = 2*(a+b-1)/s.sqrt(3)+R(2, 3)*(a*b-c*c)
assert s.simplify(fg-B*B-rem-(a*a+b*b+c*c-1)/3) == 0
assert s.expand((a+b)**2-1-(a*a+b*b+c*c-1)-2*a*b+c*c) == 0
assert s.simplify(6*s.sqrt(3)/(s.sqrt(3)+1)-(9-3*s.sqrt(3))) == 0
u, v = s.symbols('u v', nonnegative=True)
P = (B+A*u)*(B+A*v)
assert s.simplify(P-B*B-(4-B*B)*u*v-A*B*(u*(1-v)+v*(1-u))) == 0
assert s.simplify(1/(2*(1-f0))-3/(4-s.sqrt(3))) == 0
assert s.simplify(s.sqrt(f0)-B/2) == 0
assert s.simplify(3/(4-s.sqrt(3))*(1+B/2)/2-(3+s.sqrt(3))/4) == 0   # distance constant, pointwise from Phi >= f0
cluster = s.Matrix([1, s.I, 0, 0])/s.sqrt(2)
values = [s.simplify((eta.H*cluster)[0]*s.conjugate((eta.H*cluster)[0])) for eta in frame]
assert f0 in values and all(s.simplify(f0-value).is_nonnegative for value in values)
assert 27*R(1, 2)-36*R(1, 4)-4 == R(1, 2)
assert s.simplify(B*B/4-f0) == 0
print('[PASS] NC-4 exact scalar positive remainders, sharp constants and cluster equality')

# Derive every cut directly in the physical magic basis, rather than infer
# the fourth moment from the single-cut Q identity.
W = s.kronecker_product(M, M)
embed = s.zeros(16, 4)
for i in range(4):
    embed[5*i, i] = 1
r, t = s.symbols('r0:3', real=True), s.symbols('s0:3', real=True)
probabilities = []
for axes in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
    permutation = s.zeros(16)
    for i in range(16):
        bits = [(i >> j) & 1 for j in (3, 2, 1, 0)]
        out = sum(bits[axes[j]] << (3-j) for j in range(4))
        permutation[out, i] = 1
    transformed = s.simplify(W.H*permutation*W*embed)
    assert all(transformed[i, j] == 0 for i in range(16)
               for j in range(4) if i not in (0, 5, 10, 15))
    H = transformed.extract((0, 5, 10, 15), range(4))
    assert H.T*H == s.eye(4)
    for i in range(4):
        bell = M*H.T[:, i]
        projector = bell*bell.H
        assert all(s.simplify(s.trace(projector*s.kronecker_product(p, s.eye(2)))) == 0
                   and s.simplify(s.trace(projector*s.kronecker_product(s.eye(2), p))) == 0
                   for p in paulis)
        C = s.Matrix([[s.simplify(s.trace(projector*s.kronecker_product(p, q)))
                       for q in paulis] for p in paulis])
        assert C.T*C == s.eye(3) and C.det() == -1
        probabilities.append((1+sum(r[j]*C[j, k]*t[k]
                                    for j in range(3) for k in range(3)))/4)
rr, ss = sum(x*x for x in r), sum(x*x for x in t)
Ar, As = sum(x**4 for x in r), sum(x**4 for x in t)
chirality = s.prod(r)*s.prod(t)
assert s.expand(sum(x*x for x in probabilities)-(3+rr*ss)/4) == 0
assert s.expand(sum(x**3 for x in probabilities)-(3+3*rr*ss-18*chirality)/16) == 0
U4 = (3+6*rr*ss+Ar*As+R(3, 2)*(rr**2-Ar)*(ss**2-As)-72*chirality)/64
assert s.expand(sum(x**4 for x in probabilities)-U4) == 0
X, Y, q = s.symbols('X Y Q', real=True)
U4unit = (21-3*(Ar+As)+5*Ar*As-144*chirality)/128
assert s.expand(U4-U4unit-((rr*ss-1)*6
                +R(3, 2)*((rr**2-Ar)*(ss**2-As)-(1-Ar)*(1-As)))/64) == 0
aa, bb, cc = s.symbols('aa bb cc')
U4xy = ((21-3*(aa+bb)+5*aa*bb-144*cc)/128).subs(
    {aa: (X+1)/3, bb: (Y+1)/3, cc: (2*q-1)/27})
Xi = s.expand(U4xy+q/9-R(7, 36))
assert s.expand(Xi-q/36+(4*X+4*Y-5*X*Y)/1152) == 0
assert s.expand(Xi-q/72-X*Y/384-(q-(X+Y)/4+X*Y/8)/72) == 0
x, y, z = s.symbols('x y z', nonnegative=True)
assert s.expand((x*y+y*z+z*x)**2-3*x*y*z*(x+y+z)
                -((x*y-y*z)**2+(y*z-z*x)**2+(z*x-x*y)**2)/2) == 0
print('[PASS] NC-5 all three physical cut transforms, fourth moments and Schur remainder')

x, z = s.symbols('x z', positive=True)
nodes = (R(1, 6), R(1, 2))
f = -x*s.log(x)
coeff = s.symbols('h0:4')
h = sum(coeff[i]*x**i for i in range(4))
def hermite_solution(function):
    return s.solve([(h-function).subs(x, a) for a in nodes]
                   + [s.diff(h-function, x).subs(x, a) for a in nodes], coeff)
scalar = x/(x+z)
interpolant = h.subs(hermite_solution(scalar))
weight = (x-nodes[0])**2*(x-nodes[1])**2
kernel = z/((x+z)*(z+nodes[0])**2*(z+nodes[1])**2)
assert s.factor(interpolant-scalar-weight*kernel) == 0
print('[PASS] NC-6 exact positive Hermite-remainder kernel')
print('All NC checks passed; two-sphere geometry and the kernel argument are analytic.')
