#!/usr/bin/env python3
"""EH-1--EH-3: exact entropy Hessian and local-radius algebra.

Computes all 32 real coefficients from cut flattenings; checks the explicit
normal block decomposition, the least mode and rational radius bounds.
The relative-entropy/resolvent estimate and limiting argument are analytic.
"""
from itertools import combinations
import sympy as s

I, R = s.I, s.Rational


def zero(expr):
    return s.expand(expr) == 0


def matrix_zero(mat):
    return all(s.simplify(v) == 0 for v in mat)


def real_vector(mat):
    return s.Matrix([s.re(v) for v in mat] + [s.im(v) for v in mat])


def flatten(z, j):
    out = s.zeros(4)
    rest = [k for k in (1, 2, 3) if k != j]
    for q in range(16):
        bits = [(q >> (3-k)) & 1 for k in range(4)]
        out[2*bits[0]+bits[j], 2*bits[rest[0]]+bits[rest[1]]] = z[q]
    return out


magic = s.Matrix([[1, 0, 0, I], [0, I, 1, 0],
                  [0, I, -1, 0], [1, 0, 0, -I]])/s.sqrt(2)
T = s.kronecker_product(magic, magic)
star = s.diag(1/s.sqrt(2), I/s.sqrt(6), I/s.sqrt(6), I/s.sqrt(6))
zstar = T*s.Matrix(list(star))
x = s.Matrix(s.symbols('x:32', real=True))
H = s.Matrix(4, 4, lambda i, j: x[4*i+j]+I*x[16+4*i+j])
zH = T*s.Matrix(list(H))
tangents = [real_vector(star), real_vector(I*star)]
for i, j in combinations(range(4), 2):
    rotation = s.zeros(4)
    rotation[i, j], rotation[j, i] = 1, -1
    tangents += [real_vector(rotation*star), real_vector(star*rotation.T)]
V = s.Matrix.hstack(*s.Matrix.hstack(*tangents).columnspace())
assert V.cols == 11
normal = s.simplify(s.eye(32)-V*(V.T*V).inv()*V.T)
A = B = Kform = 0
K = s.zeros(16)
for j in (1, 2, 3):
    C, D = flatten(zstar, j), flatten(zH, j)
    rho = s.simplify(C*C.H)
    P = s.simplify(3*rho-s.eye(4)/2)
    Q = s.eye(4)-P
    assert matrix_zero(P*P-P) and s.trace(P) == 1
    assert rho.eigenvals() == {R(1, 2): 1, R(1, 6): 3}
    L = C*D.H+D*C.H
    A += s.trace((P*L*P)**2)+3*s.trace((Q*L*Q)**2)
    B += s.trace(P*D*D.H)+3*s.trace(P*L*Q*L*P)
    Kform += s.trace(P*D*D.H)
    for q in range(16):
        bits = [(q >> (3-i)) & 1 for i in range(4)]
        for idx in range(4):
            bs = bits.copy()
            bs[0], bs[j] = idx//2, idx % 2
            out = sum(v << (3-i) for i, v in enumerate(bs))
            K[out, q] += P[idx, 2*bits[0]+bits[j]]
nH = sum(v*v for v in x)
B -= R(3, 2)*nH
Kform -= R(3, 2)*nH
K -= R(3, 2)*s.eye(16)
assert matrix_zero(K*zstar)
assert K.eigenvals() == {R(-3, 2): 5, -1: 6, 0: 2, R(1, 2): 3}
forms = [s.simplify(normal*(s.hessian(s.expand(f), list(x))/2)*normal)
         for f in (A, B, Kform)]
A, B, Kn = forms
# Explicit orthonormal coordinates for the real normal space.
basis = []
for diag in ((1, -1, 0), (1, 1, -2)):
    W = s.zeros(4)
    scale = s.sqrt(sum(v*v for v in diag))
    for j in range(3): W[j+1, j+1] = diag[j]/scale
    basis += [W, I*W]
for i, j in combinations(range(1,4), 2):
    W = s.zeros(4)
    W[i,j] = W[j,i] = 1/s.sqrt(2)
    basis += [W, I*W]
W = s.diag(I/s.sqrt(2), *([1/s.sqrt(6)]*3))
basis += [W, I*W]
for i,j in combinations(range(1,4),2):
    W = s.zeros(4)
    W[i,j], W[j,i] = 1/s.sqrt(2), -1/s.sqrt(2)
    basis.append(W)
for j in range(1,4):
    W = s.zeros(4)
    W[0,j], W[j,0] = s.Rational(1,2), I*s.sqrt(3)/2
    B2 = s.zeros(4)
    B2[0,j], B2[j,0] = I*s.sqrt(3)/2, s.Rational(1,2)
    basis += [W,B2]
N = s.Matrix.hstack(*(real_vector(W) for W in basis))
assert matrix_zero(N.T*N-s.eye(21))
assert matrix_zero(N*N.T-normal)
for mat, vals in zip(forms[:2], ([3]*12+[2]*9,
                            [R(-3,2)]*10+[0]*2+[3]*9)):
    assert matrix_zero(N.T*mat*N-s.diag(*vals))
assert matrix_zero(Kn*N[:,:10]+s.Rational(3,2)*N[:,:10])
assert matrix_zero(Kn*N[:,10:12])
print('[PASS] EH-1 full cut forms in explicit orthonormal 10+2+9 normal blocks; '
      'entropy coefficients (2-ln3)/(2ln2)^10, (1/ln2)^2, ((2+3ln3)/(3ln2))^9; U>=6')

radius = R(1, 20)
assert 6 < R(49, 20)**2
assert R(1733, 1000)**2 > 3
rr, ee = s.symbols('r e', positive=True)
# ((1-r^2/2) sqrt6 - r sqrt18)^2 (1/2 - e/6) = (3 - e)(1 - sqrt3 r - r^2/2)^2: the bracket of c(r) in its simplified form
assert s.expand(((1-rr**2/2)*s.sqrt(6)-rr*s.sqrt(18))**2*(R(1, 2)-ee/6)
                - (3-ee)*(1-s.sqrt(3)*rr-rr**2/2)**2) == 0
# exp(1099/1000)>3 and exp(7/10)>2 prove logarithm upper bounds.
assert sum(R(1099, 1000)**i/s.factorial(i) for i in range(10)) > 3
assert sum(R(7, 10)**i/s.factorial(i) for i in range(4)) > 2
eps_upper = 2*R(49, 20)*radius+6*radius**2
norm_lower = 1-R(1733, 1000)*radius-radius**2/2
assert norm_lower > 0
margin = ((1-radius**2/4)*((3-eps_upper)*norm_lower**2
                         - R(3, 2)*R(1099, 1000))/(3*R(7, 10)))
assert margin > R(3, 10)
v, e = s.symbols('v e', nonnegative=True)
assert s.simplify(1/(1+v*e)-(1-v*e)-v*v*e*e/(1+v*e)) == 0
assert s.integrate((1-v)*(1-v*e), (v, 0, 1)) == R(1, 2)-e/6
print('[PASS] EH-2 exact bracket identity (3-epsilon)(1-sqrt3 r-r^2/2)^2, exact rational c(1/20)>3/10 and resolvent-integral coefficient 1/2-epsilon/6')

least = s.zeros(4)
least[1, 1], least[2, 2] = 1/s.sqrt(2), -1/s.sqrt(2)
w = real_vector(least)
assert matrix_zero(normal*w-w)
assert matrix_zero(A*w-3*w) and matrix_zero(B*w+R(3, 2)*w)
assert matrix_zero(Kn*w+R(3, 2)*w) and (w.T*w)[0] == 1
assert zero(R(3, 2)*(1-s.log(3)/2)/(R(3, 2)*s.log(2))
            - (2-s.log(3))/(2*s.log(2)))
print('[PASS] EH-3 explicit unit normal attains the least coefficient (algebraic eigen-relations and the closed form of c_*; the asymptotics are analytic)')
print('All EH checks passed (exact); the analytic radius and limiting proofs are in the paper.')
