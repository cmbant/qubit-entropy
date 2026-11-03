#!/usr/bin/env python3
"""IC-1--IC-6: exact checks for the icosahedral seven-qubit state.

Statement: thm:ico (in icosahedral_state.tex)

Qubit 0 is the special qubit s; qubits 1..6 carry the labels 0..5 of the
icosahedral two-graph (Seidel signs s_ij = -1 on the pentagon 1-3-2-4-5-1,
+1 otherwise).  For weights alpha^2 = (1-2x)/30, beta^2 = x,

  psi_x = |0>( alpha sum_{i<j} s_ij |e_i+e_j> + beta |111111> )
        + |1>( beta |000000> - alpha sum_{i<j} s_ij |111111 - e_i - e_j> ).

All arithmetic is exact: amplitudes are integer multiples of alpha or beta,
so every reduced matrix is R = alpha^2 R_AA + beta^2 R_BB + alpha*beta*C with
integer matrices, and purities are exact polynomials in alpha^2, beta^2 and
alpha*beta.  Floats are used only on small integers (checked integral).
"""
import itertools
from fractions import Fraction as Fr
import numpy as np
import sympy as sym

PENT = {(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)}
s = {p: (-1 if p in PENT else 1) for p in itertools.combinations(range(6), 2)}
A = np.zeros([2]*7, dtype=np.int64)   # coefficient of alpha
B = np.zeros([2]*7, dtype=np.int64)   # coefficient of beta
def idx(sb, ones):
    v = [0]*7; v[0] = sb
    for q in ones: v[1+q] = 1
    return tuple(v)
for p, sg in s.items():
    A[idx(0, p)] = sg
    A[idx(1, tuple(q for q in range(6) if q not in p))] = -sg
B[idx(0, tuple(range(6)))] = 1
B[idx(1, ())] = 1

def red(T, S, U=None):
    U = T if U is None else U
    rest = [i for i in range(7) if i not in S]
    M = np.transpose(T, list(S)+rest).reshape(2**len(S), -1)
    N = np.transpose(U, list(S)+rest).reshape(2**len(S), -1)
    return M @ N.T

def purity_poly(S):
    """tr R_S^2 as integer coefficients of (a^4, b^4, a^2 b^2, a^3 b, a b^3)."""
    Raa, Rbb = red(A, S), red(B, S)
    C = red(A, S, B) + red(B, S, A)
    tr = lambda X, Y: int(np.trace(X @ Y))
    return (tr(Raa, Raa), tr(Rbb, Rbb), 2*tr(Raa, Rbb) + tr(C, C),
            2*tr(Raa, C), 2*tr(Rbb, C))

# IC-1: normalization; alpha^2*30 + beta^2*2 = 1 is the stated weight constraint.
assert int((A*A).sum()) == 30 and int((B*B).sum()) == 2 and int((A*B).sum()) == 0
print("IC-1 PASS: support sizes 30 and 2, disjoint; psi_x is a unit vector for every x in [0,1/2]")

# IC-2: one-body marginals are I/2 for every x (all odd cross terms vanish too).
for i in range(7):
    for X in (A, B):
        R = red(X, (i,))
        assert R[0, 1] == 0 and R[1, 0] == 0 and R[0, 0] == R[1, 1]
    assert np.all(red(A, (i,), B) + red(B, (i,), A) == 0)
print("IC-2 PASS: every one-qubit marginal is I/2 along the whole family")

# IC-3: the alpha*beta-odd terms of every purity vanish; mean 3-body purity is
# a quadratic in x.
polys = {S: purity_poly(S) for S in itertools.combinations(range(7), 3)}
assert all(p[3] == 0 and p[4] == 0 for p in polys.values())
c4a = sum(p[0] for p in polys.values()); c4b = sum(p[1] for p in polys.values())
cab = sum(p[2] for p in polys.values())
def mean_purity(x):
    a2, b2 = (1-2*x)/30, x
    return (c4a*a2*a2 + c4b*b2*b2 + cab*a2*b2)/35
print("IC-3 PASS: mean 3-body purity = (%d a^4 + %d b^4 + %d a^2 b^2)/35, odd terms zero" % (c4a, c4b, cab))

# IC-4: exact minimum on the family at x = 37/524, value 121/917.
x = [Fr(0), Fr(1, 2)]
# quadratic q(x) = q0 + q1 x + q2 x^2 from three exact evaluations
v0, v1, v2 = (mean_purity(Fr(t, 4)) for t in (0, 1, 2))
q2 = (v2 - 2*v1 + v0)*8; q1 = (v1 - v0)*4 - q2/4; q0 = v0
xstar = -q1/(2*q2)
assert q2 > 0 and xstar == Fr(37, 524)
assert mean_purity(xstar) == Fr(121, 917)
assert Fr(121, 917) < Fr(19, 140)
print("IC-4 PASS: family minimum at x = 37/524, mean 3-body purity exactly 121/917 < 19/140")

# IC-5: the per-subset purities at x = 37/524: 20 triples avoiding s, 15 containing s.
a2, b2 = Fr(15, 524), Fr(37, 524)
vals = {}
for S, p in polys.items():
    v = p[0]*a2*a2 + p[1]*b2*b2 + p[2]*a2*b2
    vals.setdefault((0 in S, v), []).append(S)
summary = sorted((k[0], str(k[1]), len(v)) for k, v in vals.items())
assert summary == [(False, '4381/34322', 20), (True, '2363/17161', 15)], summary
pairs = {}
for S in itertools.combinations(range(7), 2):
    p = purity_poly(S); assert p[3] == p[4] == 0
    v = p[0]*a2*a2 + p[1]*b2*b2 + p[2]*a2*b2
    pairs.setdefault((0 in S, v), 0); pairs[(0 in S, v)] += 1
assert sorted((k[0], str(k[1]), n) for k, n in pairs.items()) == [(False, '8641/34322', 15), (True, '8761/34322', 6)]
print("IC-5 PASS: triple purities 4381/34322 (x20, avoiding s) and 2363/17161 (x15, containing s);"
      " pair purities 8641/34322 (x15, avoiding s) and 8761/34322 (x6, containing s)")

# IC-6: exact spectra throughout the entire family, without eigenvalue
# approximation. For alpha=1,beta=t the annihilating polynomial has
# degree at most six in t. Seven integer samples therefore certify the
# identity for every t; homogeneity also covers alpha=0. Trace powers
# fix the multiplicities when the roots are distinct; use continuity
# at collisions. Python integers avoid machine-integer overflow.
conference = sym.Matrix([[0 if i == j else s[tuple(sorted((i, j)))]
                         for j in range(6)] for i in range(6)])
assert conference**2 == 5*sym.eye(6) and conference.det() == -125
aa4, ab4, ac4, bc4, bd4, cd4 = sym.symbols('a b c d e f')
generic4 = sym.Matrix([[0, aa4, ab4, ac4], [aa4, 0, bc4, bd4],
                      [ab4, bc4, 0, cd4], [ac4, bd4, cd4, 0]])
matchings = [aa4*cd4, ab4*bd4, ac4*bc4]
assert sym.expand(generic4.det()+sum(matchings)**2
                  -2*sum(v*v for v in matchings)) == 0
for K in itertools.combinations(range(6), 4):
    i, j, k, l = K
    haf = s[i, j]*s[k, l]+s[i, k]*s[j, l]+s[i, l]*s[j, k]
    assert conference.extract(K, K).det() == 5 and haf*haf == 1
print('IC-6 PASS: conference equation, complementary minors and generic hafnian-determinant identity')
I8=np.eye(8,dtype=object)
for S in polys:
    aa,bb=red(A,S).astype(object),red(B,S).astype(object)
    cross=(red(A,S,B)+red(B,S,A)).astype(object)
    for t in range(7):
        R=aa+t*t*bb+t*cross
        pairs=[(8,1),(4,5),(1+t*t,2)] if 0 in S else [(4,6),(3+t*t,2)]
        annihilator=I8
        for root,multiplicity in pairs:
            annihilator=annihilator@(R-root*I8)
        assert not annihilator.any(),(S,t)
        for degree in range(len(pairs)):
            assert np.trace(np.linalg.matrix_power(R,degree))==sum(
                multiplicity*root**degree for root,multiplicity in pairs)
# The derivative at x*=37/524 is 4/(35 ln2) times log of this ratio.
# This integer comparison proves entropy is not stationary there.
assert 30**10*15**5 < 13**7*41**8
print("IC-6 PASS: exact all-family triple spectra, 35 cuts, degree-six polynomial identities")
print("IC-6 PASS: entropy derivative at the purity minimum is negative (exact integer sign)")
# E_3 = (195 f(4u) + 40 f(3u+x) + 15 f(8u) + 30 f(u+x))/35 with f(t) = -t log2 t and u = (1-2x)/30.
# The eigenvalues sum to 1 for every x, so 35 ln2 E_3'(x) = ln[(4u)^52 (8u)^8 / ((3u+x)^32 (u+x)^28)].
def dlog_ratio(xv):
    u = (1-2*xv)/30
    return (4*u)**52*(8*u)**8/((3*u+xv)**32*(u+xv)**28)
assert 195*4+40*3+15*8+30 == 30*35 and 40+30 == 2*35          # trace 1 for every x
assert dlog_ratio(Fr(37, 524)) == (Fr(30**10*15**5, 13**7*41**8))**4
assert dlog_ratio(Fr(0)) == Fr(16, 3)**32 and Fr(16, 3) > 1
us = (1-2*Fr(37, 524))/30; xs_ = Fr(37, 524)
assert (4*us, 3*us+xs_, 8*us, us+xs_) == (Fr(15, 131), Fr(41, 262), Fr(30, 131), Fr(13, 131))
# Exact family polynomials for pi_2(psi_x), pi_3(psi_x) and mean 3-body third moment b_3(psi_x):
pair_polys = [purity_poly(S) for S in itertools.combinations(range(7), 2)]
for k in range(5):
    xv = Fr(k, 8)
    uv = (1 - 2*xv)/30
    pi2_val = sum(p[0]*uv*uv + p[1]*xv*xv + p[2]*uv*xv for p in pair_polys) / 21
    pi3_spec = (20*(6*(4*uv)**2 + 2*(3*uv+xv)**2) + 15*((8*uv)**2 + 5*(4*uv)**2 + 2*(uv+xv)**2)) / 35
    b3_spec = (20*(6*(4*uv)**3 + 2*(3*uv+xv)**3) + 15*((8*uv)**3 + 5*(4*uv)**3 + 2*(uv+xv)**3)) / 35
    assert pi2_val == (832*xv**2 - 112*xv + 163) / 630
    assert pi3_spec == mean_purity(xv) == (2096*xv**2 - 296*xv + 149) / 1050
    assert b3_spec == (35008*xv**3 + 17328*xv**2 - 3084*xv + 709) / 31500
b3_slope_ico = (3*35008*xs_**2 + 2*17328*xs_ - 3084) / 31500
assert b3_slope_ico == Fr(-432, 120127)
print("IC-6 PASS: E_3'(37/524) = 4/(35 ln2) ln(30^10 15^5/(13^7 41^8)) < 0 and E_3'(0) = 32 ln(16/3)/(35 ln2) > 0 (exact)")
print("IC-6 PASS: Psi_ico triple spectra (41/262)^2 (15/131)^6 avoiding s; 30/131, (15/131)^5, (13/131)^2 containing s; b_3'(37/524) = -432/120127")
