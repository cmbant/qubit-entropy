#!/usr/bin/env python3
"""PM: exact finite identities behind the sharp P=1 coefficient (p1_sharp_coefficient.tex).

PM-1  the twelve Bell correlation matrices C^{ji} of the three physical cuts are signed
      permutations: cut j has spectrum (1+sigma.u^(j))/4, u^(j)_a = r_{a+k_j} s_a, sigma in the
      four odd-minus sign vectors T, with k_j = 0,-1,+1 (computed from the physical magic basis);
PM-2  sum_j |u^(j)|^2 = |r|^2 |s|^2; the symmetry generators (cyclic shift of r alone or s alone,
      even sign flips) permute the twelve probabilities, and the 32 opposite-parity cube pairs are
      invariant under every permutation of r alone, of s alone, and under even flips;
PM-4  the equality state: spectra, HS overlap F*, and C* from its closed form.
"""
from itertools import product, combinations
import sympy as s

M = s.Matrix([[1, 0, 0, s.I], [0, s.I, 1, 0], [0, s.I, -1, 0], [1, 0, 0, -s.I]])/s.sqrt(2)
paulis = [s.Matrix([[0, 1], [1, 0]]), s.Matrix([[0, -s.I], [s.I, 0]]), s.diag(1, -1)]
W = s.kronecker_product(M, M)
embed = s.zeros(16, 4)
for i in range(4):
    embed[5*i, i] = 1
T = [(-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]
shifts = []
for axes in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
    perm = s.zeros(16)
    for i in range(16):
        bits = [(i >> j) & 1 for j in (3, 2, 1, 0)]
        out = sum(bits[axes[j]] << (3-j) for j in range(4))
        perm[out, i] = 1
    tr = s.simplify(W.H*perm*W*embed)
    H = tr.extract((0, 5, 10, 15), range(4))
    assert H.T*H == s.eye(4)
    Cs = []
    for i in range(4):
        bell = M*H.T[:, i]
        P = bell*bell.H
        assert all(s.simplify(s.trace(P*s.kronecker_product(p, s.eye(2)))) == 0
                   and s.simplify(s.trace(P*s.kronecker_product(s.eye(2), p))) == 0 for p in paulis)
        Cs.append(s.Matrix([[s.simplify(s.trace(P*s.kronecker_product(p, q))) for q in paulis]
                            for p in paulis]))
    found = None
    for k in (0, 1, 2):          # C[alpha, beta] = sigma_beta * [alpha == beta + k mod 3]
        sig = []
        for C in Cs:
            row = [C[(b+k) % 3, b] for b in range(3)]
            if all(C[a, b] == 0 for a in range(3) for b in range(3) if a != (b+k) % 3) and \
               all(x in (-1, 1) for x in row):
                sig.append(tuple(row))
        if len(sig) == 4:
            found = k
            assert sorted(sig) == sorted(T)
    assert found is not None
    shifts.append(found)
assert sorted(shifts) == [0, 1, 2]
print('[PASS] PM-1 all 12 cut correlation matrices are signed permutations; u^(j)_a=r_(a+k_j) s_a, '
      'k_j=%s, sigma ranges over the 4 odd-minus vectors' % shifts)

r = s.symbols('r0:3', real=True); t = s.symbols('s0:3', real=True)
def probs(r, t):
    out = []
    for k in range(3):
        for sg in T:
            out.append(s.expand((1+sum(sg[a]*r[(a+k) % 3]*t[a] for a in range(3)))/4))
    return out
base = probs(r, t)
assert s.expand(sum(sum(r[(a+k) % 3]**2*t[a]**2 for a in range(3)) for k in range(3))
                - sum(x*x for x in r)*sum(x*x for x in t)) == 0
def same(a, b): return sorted(map(str, a)) == sorted(map(str, b))
gens = {'cycle r': ([r[1], r[2], r[0]], list(t)), 'cycle s': (list(r), [t[1], t[2], t[0]]), 'swap r,s': (list(t), list(r))}
for (i, j) in ((0, 1), (0, 2), (1, 2)):
    rr = list(r); rr[i] = -rr[i]; rr[j] = -rr[j]; gens['flip r%d r%d' % (i, j)] = (rr, list(t))
    tt = list(t); tt[i] = -tt[i]; tt[j] = -tt[j]; gens['flip s%d s%d' % (i, j)] = (list(r), tt)
for i, j in product(range(3), repeat=2):
    rr = list(r); rr[i] = -rr[i]; tt = list(t); tt[j] = -tt[j]
    gens['flip r%d s%d' % (i, j)] = (rr, tt)
for name, (rr, tt) in gens.items():
    assert same(probs(rr, tt), base), name
pairs = {(e, f) for e in product((1, -1), repeat=3) for f in product((1, -1), repeat=3)
         if s.prod(e)*s.prod(f) == -1}
for perm in __import__('itertools').permutations(range(3)):
    assert {(tuple(e[p] for p in perm), f) for e, f in pairs} == pairs
    assert {(e, tuple(f[p] for p in perm)) for e, f in pairs} == pairs
assert {(f, e) for e, f in pairs} == pairs
for fl in product((1, -1), repeat=6):
    if s.prod(fl) == 1:
        assert {(tuple(a*b for a, b in zip(e, fl[:3])), tuple(a*b for a, b in zip(f, fl[3:])))
                for e, f in pairs} == pairs
print('[PASS] PM-2 sum_j|u_j|^2=|r|^2|s|^2; delta-generators (cycle r, cycle s, swap r,s, even flips) permute the 12 '
      'probabilities; the 32 admissible cube pairs are invariant under S3 on r, S3 on s, swap and even flips')

# Exact equality data: algebraic signs, not floating-point coincidences.
p = (2+s.sqrt(2))/4
Fs = (3+s.sqrt(3)+s.sqrt(2)+s.sqrt(6))/12
rv = (s.Integer(1), s.Integer(0), s.Integer(0))
sv = (1/s.sqrt(2), s.Integer(0), 1/s.sqrt(2))
values = probs(rv, sv)
expected = [p/2, p/2, (1-p)/2, (1-p)/2]
for k in range(3):
    target = [s.Rational(1,4)]*4 if k == 2 else expected
    assert same([s.simplify(x) for x in values[4*k:4*k+4]],
                [s.simplify(x) for x in target])
q = [s.simplify((1+sum(e[a]*rv[a] for a in range(3))/s.sqrt(3))*
               (1+sum(f[a]*sv[a] for a in range(3))/s.sqrt(3))/4) for e,f in pairs]
assert all(s.simplify(Fs-x).is_nonnegative is True for x in q)
assert sum(s.simplify(Fs-x) == 0 for x in q) == 4
assert s.simplify(4*Fs-(1+1/s.sqrt(3))*(1+s.sqrt(s.Rational(2,3)))) == 0
# Check that the stated equality set is exactly one orbit of the proved symmetries.
axes = {tuple(s.Integer(sign) if j == i else s.Integer(0) for j in range(3))
        for i in range(3) for sign in (-1,1)}
edges = {tuple(s.Integer(0) if j == i else signs[j]/s.sqrt(2) for j in range(3))
         for i in range(3) for signs in product((-1,1), repeat=3)}
equality = {(r,t) for r in axes for t in edges} | {(t,r) for r in axes for t in edges}
orbit, todo = {(rv,sv)}, [(rv,sv)]
while todo:
    r0,t0 = todo.pop()
    next_pairs = [(r0[1:]+r0[:1],t0), (r0,t0[1:]+t0[:1]), (t0,r0)]
    z = r0+t0
    for i,j in combinations(range(6),2):
        zz = tuple(-x if k in (i,j) else x for k,x in enumerate(z))
        next_pairs.append((zz[:3],zz[3:]))
    for pair in next_pairs:
        if pair not in orbit:
            orbit.add(pair); todo.append(pair)
assert len(orbit) == 144 and orbit == equality
H2 = -(p*s.log(p)+(1-p)*s.log(1-p))/s.log(2)
cstar = (s.log(3)/(2*s.log(2))-(1+2*H2)/3)/(2-2*s.sqrt(Fs))
print('[PASS] PM-4 exact equality spectra, maximum HS fidelity (3+sqrt3+sqrt2+sqrt6)/12, '
      'four tied pairs and all 144 symmetry-related coordinate pairs')
print('[EVALUATION] C* = '+str(s.N(cstar, 25)))
