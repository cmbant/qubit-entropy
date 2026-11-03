"""DP5-1: exact probe-space dimension, coset count, and target calibrations.

Uses integer S5 characters and complete-symmetric-function recurrences.
Does not provide an evaluation matrix of full rank or a sharp PSD dual.
"""
import itertools
import math
from collections import Counter
from fractions import Fraction as F

import numpy as np
import sympy as sp

import verify_five_copy_site_algebra as s5


def cycles(p):
    seen, out = set(), []
    for i in range(5):
        if i in seen:
            continue
        n, j = 0, i
        while j not in seen:
            seen.add(j)
            n += 1
            j = p[j]
        out.append(n)
    return tuple(sorted(out, reverse=True))


def power(p, k):
    q = tuple(range(5))
    for _ in range(k):
        q = tuple(p[q[i]] for i in range(5))
    return q


def h7(traces):
    h = [F(1)]
    for n in range(1, 8):
        h.append(sum(F(traces[k-1])*h[n-k] for k in range(1, n+1))/n)
    return h[7]


classes = Counter(cycles(p) for p in s5.PERMS)
representatives = {c: next(p for p in s5.PERMS if cycles(p) == c) for c in classes}
chars = {}
for p in s5.PERMS:
    k = s5.PINDEX[p]
    chars[p] = [int(sp.trace(s5.RHO[m][k])) for m in range(3)]
    assert sum((6-2*m)*chars[p][m] for m in range(3)) == 2**len(cycles(p))

all_dim = F(0)
for c, nc in classes.items():
    for d, nd in classes.items():
        p, q = representatives[c], representatives[d]
        traces = [sum(a*b for a, b in zip(chars[power(p, k)], chars[power(q, k)]))
                  for k in range(1, 8)]
        all_dim += nc*nd*h7(traces)/120**2
transpose_trace = F(0)
for c, nc in classes.items():
    p = representatives[c]
    traces = [sum(chars[power(p, k)]) if k % 2 else
              sum(x*x for x in chars[power(p, k//2)]) for k in range(1, 8)]
    transpose_trace += nc*h7(traces)/120
assert all_dim == 5844 and transpose_trace == 726
assert (all_dim+transpose_trace)/2 == 3285
print('DP5-1.1 PASS: exact character count 5844, transpose trace 726, even dimension 3285')

x = sp.Symbol('x')
u = (1-2*x)/30
purity_family = (4470*u*u+70*x*x+300*u*x)/35
pair_family = 6*((1-8*x)/3)**2+15*((32*x-1)/15)**2
star = sp.Rational(37, 524)
assert purity_family.subs(x, star) == sp.Rational(121, 917)
assert sp.diff(purity_family, x).subs(x, star) == 0
assert sp.diff(pair_family, x).subs(x, star) == sp.Rational(96, 131)
print('DP5-1.1 PASS: independent value/derivative functionals; critical subspace dimension 3283')

E = np.array(s5.EMB_NUM, dtype=np.int64).reshape(42, -1)
assert 1024*int(np.abs(E).max())**2 < 2**63
W = sp.Matrix(E@E.T)
for j, flipped in enumerate(((), (4,), (3, 4))):
    H = [p for p in s5.PERMS if set(p[5-j:]) == set(range(5-j, 5))]
    assert len(H) == math.factorial(5-j)*math.factorial(j)
    assert (120//len(H))**2 == (1, 25, 100)[j]
    T = sp.eye(42) if not j else s5.flip_map(flipped)
    assert T.T*W == W*T
print('DP5-1.2 PASS: 1/25/100 coset terms for zero/one/two flips; local maps trace-self-adjoint')

transpositions = [p for p in s5.PERMS if cycles(p) == (2, 1, 1, 1)]
assert len(transpositions) == 10
for scalar in (False, True):
    a = {p: 6 if scalar else 2**len(cycles(p)) for p in s5.PERMS}
    norm = F(sum(v**7 for v in a.values()), 120)
    purity = F(sum(a[tuple(t[p[i]] for i in range(5))]**3*a[p]**4
                   for p in s5.PERMS for t in transpositions), 1200)
    if scalar:
        assert norm == 6**7 and purity == norm
    else:
        assert norm == math.comb(132, 5) and purity/norm == F(8, 43)
    assert purity-F(121, 917)*norm > 0
print('DP5-1.3 PASS: target calibrated at identity (Haar purity 8/43) and scalar sector (product purity 1)')
print('DP5-1 PASS; full-rank probe matrix and verified PSD repair remain OPEN')
