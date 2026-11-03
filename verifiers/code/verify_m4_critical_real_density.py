#!/usr/bin/env python3
"""RP: exact real-density/Bures model for the complete critical locus.

Checks the real measurement frame, tomography, all 16 HS planes, Bures
compression algebra, purity/spin flip and the entropy-gradient identity.
The rank-two parametrization and entropy/Bures arguments are analytic in
critical_orbit_stability.tex. No numerical optimizer is a proof input.
"""
from itertools import product
import sympy as s

R = s.Rational
H = [s.eye(4), s.Matrix([[1, -1, 1, -1], [-1, 1, 1, -1],
                       [1, 1, 1, 1], [-1, -1, 1, 1]])/2,
     s.Matrix([[1, -1, -1, -1], [-1, 1, -1, -1],
               [1, 1, -1, 1], [-1, -1, -1, 1]])/2]
projectors = []
for i, h in enumerate(H):
    assert h*h.T == s.eye(4)
    for k in range(i):
        assert all(x*x == R(1, 4) for x in h*H[k].T)
    projectors.extend(h[j, :].T*h[j, :] for j in range(4))
a = s.symbols('a0:10', real=True)
A = s.zeros(4)
for value, (i, j) in zip(a, ((i, j) for j in range(4) for i in range(j+1))):
    A[i, j] = A[j, i] = value
probabilities = [s.trace(P*A) for P in projectors]
assert s.simplify(sum((p*P for p, P in zip(probabilities, projectors)), s.zeros(4))
                  -A-s.trace(A)*s.eye(4)/2) == s.zeros(4)
assert s.expand(sum(p*p for p in probabilities)-s.trace(A*A)-s.trace(A)**2/2) == 0
print('[PASS] RP-1 three real MUBs, exact tomography and tight quadratic frame')

x, y = s.Matrix(s.symbols('x0:4', real=True)), s.Matrix(s.symbols('y0:4', real=True))
v = x+s.I*y
Ar = x*x.T+y*y.T
n = (x.T*x+y.T*y)[0]
spin = (x.T*x-y.T*y)[0]**2+4*(x.T*y)[0]**2
assert s.expand(spin-2*s.trace(Ar*Ar)+n*n) == 0
assert s.expand(sum(s.trace(P*Ar)**2 for P in projectors)-n*n-spin/2) == 0
count = 0
frame, overlaps = [], []
for i in range(4):
    remaining = [j for j in range(4) if j != i]
    e = s.eye(4)[:, i]
    for signs in product((-1, 1), repeat=2):
        u = s.zeros(4, 1)
        for j, sign in zip(remaining, (1,)+signs):
            u[j] = sign/s.sqrt(3)
        P = e*e.T+u*u.T
        A0 = P/2
        eta = (e+s.I*u)/s.sqrt(2)
        assert s.simplify(s.re(eta*eta.H)-A0) == s.zeros(4)
        s0 = s.expand(s.trace(P*Ar))
        overlap = s.expand((eta.H*v)[0]*s.conjugate((eta.H*v)[0]))
        opposite = s.expand((eta.T*v)[0]*s.conjugate((eta.T*v)[0]))
        compression = s.Matrix.hstack(e, u).T*Ar*s.Matrix.hstack(e, u)
        assert s.simplify(overlap+opposite-s0) == 0
        assert s.simplify((overlap-opposite)**2/4-compression.det()) == 0
        assert s.simplify(s.trace((Ar-A0)**2)
                          -(s.trace(Ar*Ar)+R(1, 2)-s0)) == 0
        reference = [s.simplify(s.trace(B*A0)) for B in projectors]
        assert reference.count(R(1, 2)) == 3 and reference.count(R(1, 6)) == 9
        selected = sum((B for B, q in zip(projectors, reference)
                        if q == R(1, 2)), s.zeros(4))
        assert s.simplify(selected-R(3, 2)*P) == s.zeros(4)
        frame.extend([eta, s.conjugate(eta)])
        overlaps.extend([overlap, opposite])
        count += 1
assert count == 16
print('[PASS] RP-2 all 16 HS planes, paired overlaps and Bures compression determinants')
assert s.expand(sum(overlaps)-8*n) == 0
assert s.expand(sum(q*q for q in overlaps)-R(8, 9)*(4*n*n-spin)) == 0
ghz = s.Matrix([1, 1, 0, 0])/s.sqrt(2)
values = [s.simplify((eta.H*ghz)[0]*s.conjugate((eta.H*ghz)[0])) for eta in frame]
assert max(values) == R(1, 3)
print('[PASS] RP-3 spin flip, entropy gradient, exact fourth frame moment and sharp GHZ floor')

# Exact normal family: any correction G4+beta*tau >= 0 needs beta >= 3/4.
z = s.symbols('z', real=True)
c = (s.sqrt(3)-1)/2
v0 = s.Matrix([1, c*(1-s.I), c*(1+s.I), -s.I])
n0 = s.simplify((v0.H*v0)[0])
w = v0+z*s.conjugate(v0)
nw = s.simplify((w.H*w)[0])
p = [s.expand(q*s.conjugate(q)) for h in H for q in h*w]
G4 = s.expand(72*sum(q**4 for q in p)-84*nw*sum(q**3 for q in p)
               +30*nw*nw*sum(q*q for q in p)-9*nw**4)
assert s.simplify(G4/n0**4+3*z*z*(z**4-19*z*z+1)) == 0
assert s.simplify(sum(q*q for q in w)*s.conjugate(sum(q*q for q in w))/nw**2
                  -4*z*z/(1+z*z)**2) == 0
assert s.simplify(s.limit(G4/(nw**2*sum(q*q for q in w)
                                   *s.conjugate(sum(q*q for q in w))), z, 0)) == -R(3, 4)
print('[PASS] RP-4 exact rank-two normal family forces quartic correction beta >= 3/4')
print('All RP checks passed; full critical entropy/orbit reduction is analytic.')
