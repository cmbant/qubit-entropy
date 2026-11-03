#!/usr/bin/env python3
"""FP: sharp critical HS covering profile, exact rational certificate.

Checks every coefficient and positive Gram of the four-point cubic inequality
in all 23 ordering classes, plus the fusion-frame identity and matching family.
The ordering coverage, area-weighted Bures argument and spin-flip/Bloch bridge
are analytic in critical_covering_geometry.tex; NG and RP supply their stated
normal-form inputs. No optimization solver or floating-point proof input.
"""
import json
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from math import lcm
from pathlib import Path
import sympy as s

R = s.Rational
variables = s.symbols('r0:3 s0:3')
E = s.Matrix([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
assert E.T*E == 4*s.eye(3) and E.T*s.ones(4, 1) == s.zeros(3, 1)
X, Y = list(E*s.Matrix(variables[:3])), list(E*s.Matrix(variables[3:]))
spheres = [sum(x*x for x in variables[:3])-1, sum(x*x for x in variables[3:])-1]
point = dict(zip(variables, (-1, 0, 0, -1, 0, 0)))

def canonical(word):
    swapped = word.translate(str.maketrans('XY', 'YX'))
    return min(word, word[::-1], swapped, swapped[::-1])

words = {''.join('X' if i in indices else 'Y' for i in range(8))
         for indices in combinations(range(8), 4)}
classes = {canonical(word) for word in words}
assert len(words) == 70 and len(classes) == 23
print('[PASS] FP-1 all 70 interleavings reduce to exactly 23 ordering classes')

@lru_cache(None)
def monos(degree):
    out = []
    for d in range(degree+1):
        for indices in combinations_with_replacement(range(6), d):
            out.append(tuple(sum(i == j for j in indices) for i in range(6)))
    return tuple(out)

@lru_cache(None)
def basis(degree, face):
    monomials = s.Matrix([s.prod(x**a for x, a in zip(variables, mm)) for mm in monos(degree)])
    if face:
        rows = [[b.subs(point) for b in monomials]]
        for j in face[1:]:
            rows.append([s.diff(b, variables[j]).subs(point) for b in monomials])
        kernel = s.Matrix.hstack(*s.Matrix(rows).nullspace())
        monomials = kernel.T*monomials
    out = []
    for b in monomials:
        p = s.Poly(b, *variables).as_dict()
        assert all(c.is_Integer for c in p.values())
        out.append({mm: int(c) for mm, c in p.items()})
    return out

@lru_cache(None)
def multiply(a, b):
    out = defaultdict(int)
    for ma, ca in a:
        for mb, cb in b:
            out[tuple(x+y for x, y in zip(ma, mb))] += ca*cb
    return tuple((mm, c) for mm, c in out.items() if c)

raw = json.loads((Path(__file__).parent/'data/m4_critical_covering_profile.json').read_text())
assert raw['format'] == 'four-point third moment SOS v1'
assert len(raw['cases']) == 23 and {c['word'] for c in raw['cases']} == classes
count = 0
for case in raw['cases']:
    word = case['word']
    assert canonical(word) == word
    positions_x, positions_y = [i for i, c in enumerate(word) if c == 'X'], [i for i, c in enumerate(word) if c == 'Y']
    i = j = 0
    ordered = []
    for letter in word:
        if letter == 'X':
            ordered.append(X[i]); i += 1
        else:
            ordered.append(Y[j]); j += 1
    gaps = [s.expand(ordered[i+1]-ordered[i]) for i in range(7)]
    target = s.Poly(s.expand(sum((1 if positions_x[i] > positions_y[j] else -1)
                                *(X[i]-Y[j])**3 for i in range(4) for j in range(4))/16-4), *variables)
    binary = word[:4].count('X') == 2
    bases = [basis(2, (0, 1, 2, 4, 5) if binary else ())]
    for g in gaps:
        face = (0, 1, 2, 4, 5) if binary and g.subs(point) > 0 else ((0,) if binary else ())
        bases.append(basis(1, face))
    den, ld = int(case['den']), case['cholesky_grid']
    assert den > 0 and ld > 0 and len(case['blocks']) == 8
    residual = defaultdict(int)
    for mm, c in target.terms():
        assert (c*den).is_Integer
        residual[mm] = int(c*den)
    for block, B, g in zip(case['blocks'], bases, [s.Integer(1)]+gaps):
        dim = len(B)
        assert block['size'] == dim and len(block['upper']) == dim*(dim+1)//2
        multip = tuple((mm, int(c)) for mm, c in s.Poly(g, *variables).terms())
        matrix = [[0]*dim for _ in range(dim)]
        k = 0
        for j in range(dim):
            for i in range(j+1):
                value = int(block['upper'][k]); k += 1
                matrix[i][j] = matrix[j][i] = value
                polynomial = multiply(multiply(tuple(B[i].items()), tuple(B[j].items())), multip)
                for mm, coefficient in polynomial:
                    residual[mm] -= (1 if i == j else 2)*value*coefficient
        assert len(block['cholesky_lower']) == dim*(dim+1)//2
        chol = [[0]*dim for _ in range(dim)]
        k = 0
        for i in range(dim):
            for j in range(i+1):
                chol[i][j] = block['cholesky_lower'][k]; k += 1
        common = lcm(den, ld*ld)
        fg, fl = common//den, common//(ld*ld)
        for i in range(dim):
            diagonal, radius = None, 0
            for j in range(dim):
                value = matrix[i][j]*fg-sum(chol[i][k]*chol[j][k] for k in range(min(i, j)+1))*fl
                if i == j:
                    diagonal = value
                else:
                    radius += abs(value)
            assert diagonal > radius, (word, i)
        count += 1
    assert len(case['ideal']) == 2*len(monos(2))
    k = 0
    for sphere in spheres:
        pp = s.Poly(sphere, *variables).as_dict()
        for mm in monos(2):
            coefficient = int(case['ideal'][k]); k += 1
            for exponent, value in pp.items():
                residual[tuple(a+b for a, b in zip(mm, exponent))] -= coefficient*int(value)
    assert set(residual) <= set(monos(4))
    assert all(residual[mm] == 0 for mm in monos(4)), word
assert count == 184
print('[PASS] FP-2 all 23 rational identities, 4830 coefficients and 184 positive Gram matrices')

# Unrestricted homogeneous fusion-frame identity, before imposing orthonormality.
u, v = s.Matrix(s.symbols('u0:4', real=True)), s.Matrix(s.symbols('v0:4', real=True))
u2, v2, uv = (u.T*u)[0], (v.T*v)[0], (u.T*v)[0]
area2 = s.expand(u2*v2-uv*uv)
K = v2*u*u.T+u2*v*v.T-uv*(u*v.T+v*u.T)
T = s.zeros(4)
area_sum = 0
frames = []
for i in range(4):
    remaining = [j for j in range(4) if j != i]
    for signs in product((-1, 1), repeat=2):
        q = s.zeros(4, 1)
        for j, sign in zip(remaining, (1,)+signs):
            q[j] = sign/s.sqrt(3)
        P = s.eye(4)[:, i]*s.eye(4)[i, :]+q*q.T
        d = s.expand((u[i]*(q.T*v)[0]-v[i]*(q.T*u)[0])**2)
        T += d*P
        area_sum += d
        frames.append((s.eye(4)[:, i]+s.I*q)/s.sqrt(2))
assert s.expand(area_sum-R(8, 3)*area2) == 0
assert all(s.expand(T[i, j]-R(8, 9)*(area2*s.eye(4)[i, j]+K[i, j])) == 0
           for i in range(4) for j in range(4))
M = s.Matrix([[1, 0, 0, s.I], [0, s.I, 1, 0],
              [0, s.I, -1, 0], [1, 0, 0, -s.I]])/s.sqrt(2)
epsilon = s.Matrix([[0, 1], [-1, 0]])
assert s.kronecker_product(epsilon, epsilon)*s.conjugate(M) == M
print('[PASS] FP-3 exact area-weighted plane identity and magic spin-flip compatibility')

a, b = s.symbols('a b', nonnegative=True)
normal = a*a+b*b
family = s.Matrix([a+s.I*b, -a+s.I*b, 0, 0])
profile = R(1, 3)+a*b/(s.sqrt(3)*normal)
tau = (a*a-b*b)**2/normal**2
assert s.simplify((profile-R(1, 3))**2-(1-tau)/12) == 0
values = []
for eta in frames+[s.conjugate(eta) for eta in frames]:
    q = s.simplify((eta.H*family)[0]*s.conjugate((eta.H*family)[0])/(2*normal))
    difference = s.Poly(s.expand(s.simplify((profile-q)*normal)), a, b)
    assert all(c.is_nonnegative for c in difference.coeffs())
    values.append(s.simplify(profile-q))
assert s.Integer(0) in values
print('[PASS] FP-4 matching cluster-to-GHZ family attains the profile for every spin flip')
print('All FP checks passed; sharp covering follows by the analytic area-weighted Bures argument.')
