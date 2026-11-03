#!/usr/bin/env python3
"""RG-1--RG-3: exact Pauli geometry and three-copy obstruction for the M4 sextic.

RG-1: arbitrary two-qubit Pauli formulas for tr(R^2), tr(R^3).
RG-2: the equivalent four-qubit determinant-monogamy identity, using ME-3.
RG-3: negative eigenvalues of the natural swap/cycle witness on an S3-fixed
      spin-1/2 multiplicity sector. This does NOT contradict F(z) >= 0.
"""
from itertools import product
import sympy as sp

n = sp.symbols("n", real=True)
a = sp.symbols("a0:3", real=True)
b = sp.symbols("b0:3", real=True)
c = sp.symbols("c0:9", real=True)
C = sp.Matrix(3, 3, c)
I = sp.eye(2)
sig = [sp.Matrix([[0, 1], [1, 0]]),
       sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1)]
R = (n * sp.eye(4)
     + sum((a[k] * sp.kronecker_product(sig[k], I)
            + b[k] * sp.kronecker_product(I, sig[k])
            for k in range(3)), sp.zeros(4))
     + sum((C[i, j] * sp.kronecker_product(sig[i], sig[j])
            for i in range(3) for j in range(3)), sp.zeros(4))) / 4
u = sum(x*x for x in a) + sum(x*x for x in b) + sum(x*x for x in c)
coupling = (sp.Matrix(a).T * C * sp.Matrix(b))[0]
assert sp.expand(sp.trace(R**2) - (n*n+u)/4) == 0
assert sp.expand(sp.trace(R**3) -
                 (n**3 + 3*n*u + 6*coupling - 6*C.det())/16) == 0
print("RG-1 PASS: exact two-qubit Pauli trace formulas")

W1, tau, B, D = sp.symbols("W1 tau B D", real=True)
# ME-3 gives P=n^2+(W1+tau)/2, so sum_j u_1j=n^2+2(W1+tau).
usum = n*n + 2*(W1+tau)
P = (3*n*n + usum)/4
T = (3*n**3 + 3*n*usum + 6*B - 6*D)/16
F = 17*n*P - 12*T - 12*n**3
assert sp.expand(2*F -
                 (n**3 + 8*n*(W1+tau) - 9*B + 9*D)) == 0
print("RG-2 PASS: 2F=n^3+8n(W1+tau)-9B+9D")

p = sp.diag(1, -1)  # (12) on std(S3)
r = sp.Matrix([[-sp.Rational(1,2), -sp.sqrt(3)/2],
               [sp.sqrt(3)/2, -sp.Rational(1,2)]])  # (123)
assert p*r*p == r.T and r**3 == I

def tensor(mats):
    out = sp.eye(1)
    for m in mats:
        out = sp.kronecker_product(out, m)
    return out

def on_pair(op, j):
    return tensor([op if i in (0,j) else I for i in range(4)])

# F(z)=<z^{tensor 3}, K z^{tensor 3}>.
K = -12*sp.eye(16)
for j in (1,2,3):
    K += 17*on_pair(p,j)-6*(on_pair(r,j)+on_pair(r.T,j))

# Three S3-invariant pairings in std(S3)^{tensor 4}.
vectors = []
for which in range(3):
    v = []
    for x,y,z,t in product(range(2), repeat=4):
        v.append(int((x==y and z==t) if which==0 else
                     ((x==z and y==t) if which==1 else
                      (x==t and y==z))))
    vectors.append(sp.Matrix(v))
G = sp.Matrix([[v.dot(w) for w in vectors] for v in vectors])
M = sp.Matrix([[(v.T*K*w)[0] for w in vectors] for v in vectors])
assert G == sp.Matrix([[4,2,2],[2,4,2],[2,2,4]])
assert M == sp.Matrix([[-52,42,42],[42,-52,42],[42,42,-52]])
assert (G.inv()*M).eigenvals() == {sp.Integer(4):1, sp.Integer(-47):2}
print("RG-3 PASS: copy-symmetric witness eigenvalues 4, -47, -47")
print("ALL RG CHECKS PASS")
