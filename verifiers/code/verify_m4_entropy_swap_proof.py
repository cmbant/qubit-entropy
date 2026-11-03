#!/usr/bin/env python3
"""Exact elementary contraction checks for sextic_identity.tex (the moment inequality, Theorem 4 of the paper).

SP-1: local epsilon maps, spin-flip transpose and qubit antisymmetry.
SP-2: the generic two-qubit swap/cycle contractions and marginal identities, and the pair norm
      ||(A(x)1-1(x)B)z||^2 = tr A^3 + tr B^3 - 2 Pi_ij in the marginal invariants (Pr).
SP-3: the eleven orbit coefficients and the two nontrivial class reductions.
SP-4: all three contractions and 8F = 6nW1 + 3 sum||V_i-8U_i||^2 + 12 sum_{i<j}||(R_i-R_j)z||^2.
SP-5: local polarization tensor norm, including degenerate Schmidt values.
No state sampling, invariant-dimension count or SOS search is used.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations, product
from math import comb
import sympy as s


def zero(expr):
    return s.expand(expr) == 0


def matrix_zero(A):
    return A.applyfunc(s.simplify) == s.zeros(*A.shape)


def perm_matrix(perm):
    out = s.zeros(8)
    for i in range(8):
        bits = [(i >> k) & 1 for k in (2, 1, 0)]
        row = 4*bits[perm[0]]+2*bits[perm[1]]+bits[perm[2]]
        out[row, i] = 1
    return out


ident = (0, 1, 2)
p, q, r, c = (1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0)
I, P, Q, R, C = [perm_matrix(a) for a in (ident, p, q, r, c)]
E = s.Matrix([[0, 1], [-1, 0]])
L, M = s.zeros(2, 8), s.zeros(2, 8)
for x, y, t in product(range(2), repeat=3):
    L[x, 4*x+2*y+t] = E[y, t]
    M[y, 4*x+2*y+t] = E[x, t]
S = sum((perm_matrix(a) for a in permutations(range(3))), s.zeros(8))/6
J = s.kronecker_product(s.eye(4), E)


def twist(A):
    A = J.T*A*J
    return s.Matrix(8, 8, lambda i, j: A[(i//2)*2+j%2, (j//2)*2+i%2])


assert I-P-Q-R+C+C.T == s.zeros(8)
assert S == (P+Q+R)/3 and C*R == Q
assert twist(S) == (2*I+P-Q-R)/3
assert twist((L+M).T*(L+M)) == -I+P+2*Q+2*R
assert twist(L.T*(L+M)) == R+C and twist(L.T*L) == R
assert Q*C*Q == C.T
# Link to the independent mixed-cubic checker, before using identical copies.
sm, am = (L+M)/s.sqrt(6), (M-L)/s.sqrt(2)
Amap = s.kronecker_product(sm, sm, sm)
Bmap = sum((s.kronecker_product(*[sm if j == i else am for j in range(3)])
            for i in range(3)), s.zeros(8,512))
Vmap = s.kronecker_product(L+M,L+M,L+M)
E_map = s.kronecker_product(L,L,L)+s.kronecker_product(M,M,M)
assert matrix_zero(9*Amap+19*Bmap+4/s.sqrt(6)*(2*Vmap-s.Rational(19,2)*E_map))
print('SP-1 PASS: exact local swap, antisymmetry and spin-flip transpose identities')

# Generic matrix: these are polynomial identities, not sampled density matrices.
Z = s.Matrix(4, 4, s.symbols('z0:16'))
A = s.Matrix(2, 2, lambda i,j: sum(Z[2*i+b, 2*j+b] for b in range(2)))
B = s.Matrix(2, 2, lambda i,j: sum(Z[2*a+i, 2*a+j] for a in range(2)))
n = s.trace(Z)
Pi = s.trace(Z*s.kronecker_product(A, B))
PT = s.Matrix(4, 4, lambda i,j: Z[2*(j//2)+i%2, 2*(i//2)+j%2])


def contraction(left, right):
    total = 0
    for ix in product(range(4), repeat=3):
        ai, bi = [j//2 for j in ix], [j%2 for j in ix]
        total += s.prod(Z[ix[k], 2*ai[left[k]]+bi[right[k]]] for k in range(3))
    return s.expand(total)


assert zero(contraction(p, ident)-n*s.trace(A*A))
assert zero(contraction(p, p)-n*s.trace(Z*Z))
assert zero(contraction(p, q)-Pi)
assert zero(contraction(p, c)-s.trace(Z*Z*s.kronecker_product(s.eye(2), B)))
assert zero(contraction(c, c)-s.trace(Z**3))
assert zero(contraction(c, tuple(c.index(k) for k in range(3)))-s.trace(PT**3))
assert zero(s.trace(Z*Z*s.kronecker_product(A, s.eye(2)))
            -n*s.trace(Z*Z)/2-Pi+n*s.trace(B*B)/2)
assert zero(s.trace(PT**3)+s.trace(Z**3)-n**3/2
            -3*n*s.trace(Z*Z)/2-3*Pi+3*n*(s.trace(A*A)+s.trace(B*B))/2)
assert zero(s.trace(A**3)-3*n*s.trace(A*A)/2+n**3/2)
assert zero(s.trace(B**3)-3*n*s.trace(B*B)/2+n**3/2)
I2 = s.eye(2)
D = s.kronecker_product(A, I2)-s.kronecker_product(I2, B)
pair = s.trace(Z*D*D)
assert zero(pair-s.trace(A**3)-s.trace(B**3)+2*Pi)
assert zero(pair-(3*n*(s.trace(A*A)+s.trace(B*B))-2*n**3)/2+2*Pi)
print('SP-2 PASS: generic two-qubit swap/cycle, partial-transpose and marginal trace identities; pair norm (Pr)')

# The seventh type reduces to the last by qubit antisymmetry and global cycles.
# Multiplying a cycle plus three distinct transpositions by the inverse global
# cycle gives identity plus three distinct transpositions again.
for cycle in (C, C.T):
    transformed = [T*cycle.T for T in (P, Q, R)]
    assert all(T in transformed for T in (P, Q, R))


def typ(a, b, c_, d):
    return (a,)+tuple(sorted((b, c_, d), reverse=True))


coeff = defaultdict(Fraction)
sv, kv = [Fraction(2,3),Fraction(1,3),Fraction(-1,3),Fraction(-1,3)], [-1,1,2,2]
for word in product(range(4), repeat=4):
    key = typ(*(word.count(j) for j in range(4)))
    coeff[key] += sum(sv[word[i]]*s.prod(kv[word[j]] for j in range(4) if j != i) for i in range(4))
expected = {(4,0,0,0):-s.Rational(8,3), (3,1,0,0):s.Rational(124,3),
 (2,2,0,0):-84, (2,1,1,0):-136, (1,3,0,0):s.Rational(220,3),
 (1,2,1,0):296, (1,1,1,1):64, (0,4,0,0):-20,
 (0,3,1,0):-s.Rational(248,3), (0,2,2,0):-48, (0,2,1,1):-16}
assert dict(coeff) == expected
print('SP-3 PASS: eleven exact orbit coefficients and the qubit-antisymmetry class reduction')

n3, nS, nP, T, Pi = s.symbols('n3 nS nP T Pi')
Jstar = (n3+nP+Pi)/2-3*nS/4-T/3
values = {(4,0,0,0):n3, (3,1,0,0):nS/4, (2,2,0,0):nP/3,
 (2,1,1,0):Pi/6, (1,3,0,0):nS/4, (1,2,1,0):(nP+Pi)/6-nS/8,
 (1,1,1,1):Jstar, (0,4,0,0):n3, (0,3,1,0):3*nS/8-n3/2,
 (0,2,2,0):T/3, (0,2,1,1):Jstar}
V2 = sum(coeff[key]*values[key] for key in coeff)
UV = sum(s.Rational(comb(3,m),3)*(-4*values[typ(4-m,m,0,0)]
          +8*values[typ(3-m,m,1,0)]) for m in range(4))
U2 = (2*nS-4*n3)/3
assert zero(V2-(128*n3-226*nS+136*nP+152*Pi)/3+32*T)
assert zero(UV-(-8*n3-2*nS+8*Pi)/3)
nW = 2*nS-4*n3
res = V2-16*UV+64*U2
assert zero(res-(s.Rational(136, 3)*nP-32*T-22*nS+8*Pi))
pairs = s.Rational(9, 2)*nS-6*n3-2*Pi          # sum over the six pairs of SP-2's pair norm
assert zero(6*nW+3*res+12*pairs-8*(17*nP-12*T-12*n3))
print('SP-4 PASS: all three contraction formulas; 8F = 6nW1 + 3||V-8U||^2 + 12 sum||(R_i-R_j)z||^2')

lam, mu = s.symbols('lam mu', nonnegative=True)
RE = s.diag(lam, mu)*E.T
col = RE.reshape(4,1)
U = S*s.Matrix.hstack(s.sqrt(lam)*s.kronecker_product(s.Matrix([1,0]), col),
                     s.sqrt(mu)*s.kronecker_product(s.Matrix([0,1]), col))
assert zero(s.trace(U.T*U)-(lam+mu)*(lam-mu)**2/3)
print('SP-5 PASS: local tensor norm n(lambda-mu)^2/3; no nondegeneracy assumption')
print('ALL SP CHECKS PASS')
