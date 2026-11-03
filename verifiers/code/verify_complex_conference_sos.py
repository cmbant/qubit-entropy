#!/usr/bin/env python3
"""CC-1: exact formal sum-of-squares identity for the complex 7-qubit pair/complement ansatz.

The pair amplitudes t_ij and their conjugates are independent formal variables;
b is real.  All arithmetic is rational, without numerical interpolation.
The state has amplitudes:
  |0>(sum_ij t_ij |ij> + b |111111>)
 +|1>(b |000000> - sum_ij conjugate(t_ij) |complement(ij)>).
Its squared norm is N = 2*(sum_ij |t_ij|^2 + b^2).
The identity is homogeneous, so one may subsequently impose N=1.
"""
from collections import defaultdict
from fractions import Fraction as Q
from itertools import combinations

PAIRS = list(combinations(range(6), 2))
ID = {pair: i for i, pair in enumerate(PAIRS)}
B = 30   # 0..14: t_ij, 15..29: conjugate(t_ij), 30: real b

def add(*ps):
    d = defaultdict(Q)
    for p in ps:
        for monomial, coefficient in p.items():
            d[monomial] += coefficient
    return {m: c for m, c in d.items() if c}

def scale(p, c):
    return {m: v*c for m, v in p.items() if v*c}

def mul(a, b):
    out = defaultdict(Q)
    for x, ax in a.items():
        for y, by in b.items():
            out[tuple(sorted(x+y))] += ax*by
    return {m: c for m, c in out.items() if c}

def sq(p):
    return mul(p, p)

def conj(p):
    def c(i):
        return i+15 if i < 15 else (i-15 if i < 30 else i)
    return {tuple(sorted(c(i) for i in m)): v for m, v in p.items()}

def mon(*ids):
    return {tuple(sorted(ids)): Q(1)}

def cj(i):
    return i+15 if i < 15 else (i-15 if i < 30 else i)

# Sparse state tensor: (seven-bit basis index, formal variable index, sign).
amplitudes = []
for (i, j), v in ID.items():
    a = [0]*7
    a[1+i] = a[1+j] = 1
    amplitudes.append((tuple(a), v, 1))
    a = [1]*7
    a[1+i] = a[1+j] = 0
    amplitudes.append((tuple(a), v+15, -1))
amplitudes += [((0,1,1,1,1,1,1), B, 1),
               ((1,0,0,0,0,0,0), B, 1)]
assert len(amplitudes) == 32

# Calculate sum_{|S|=3} tr(rho_S^2) from the actual state amplitudes.
purity = defaultdict(Q)
for S in combinations(range(7), 3):
    rest = [i for i in range(7) if i not in S]
    columns = defaultdict(dict)
    for bits, var, sign in amplitudes:
        row = tuple(bits[i] for i in S)
        column = tuple(bits[i] for i in rest)
        columns[column][row] = (var, sign)
    for entries_a in columns.values():
        for entries_b in columns.values():
            shared = entries_a.keys() & entries_b.keys()
            for row in shared:
                v0, s0 = entries_a[row]
                v1, s1 = entries_b[row]
                for col in shared:
                    v2, s2 = entries_a[col]
                    v3, s3 = entries_b[col]
                    purity[tuple(sorted((cj(v0), v1, v2, cj(v3))))] += s0*s1*s2*s3
purity = {m: c for m, c in purity.items() if c}

norm_pairs = add(*(mon(i, i+15) for i in range(15)))
b2 = mon(B, B)
N = scale(add(norm_pairs, b2), 2)
degree = [
    add(*(mon(ID[tuple(sorted((i,j)))],
               ID[tuple(sorted((i,j)))]+15) for j in range(6) if j != i))
    for i in range(6)
]

# 35*(pi_3 - (121/917)*N^2), computed from the proposed SOS formula.
rhs = scale(sq(add(scale(b2,524),scale(N,-37))), Q(35,137550))
rhs = add(rhs, scale(add(*(sq(add(d,scale(norm_pairs,Q(-1,3))))
                              for d in degree)), Q(4)))
rhs = add(rhs, scale(add(*(sq(add(mon(i,i+15),
                                   scale(norm_pairs,Q(-1,15))))
                              for i in range(15))), Q(48)))
for i, j in PAIRS:
    # (T T^dagger)_ij = sum_{k not in {i,j}} t_ik conjugate(t_jk).
    term = add(*(mon(ID[tuple(sorted((i,k)))],
                     ID[tuple(sorted((j,k)))]+15)
                 for k in range(6) if k not in (i, j)))
    rhs = add(rhs, scale(mul(term, conj(term)), Q(24)))

# Aggregate form used in the text: 35 pi_3 = 7/2 N^2 - 4 b^2 N + 64 b^4
#   + 4 sum p_i^2 + 48 sum |t_ij|^4 + 24 sum |(T T^dagger)_ij|^2.
aggregate = add(scale(sq(N), Q(7, 2)), scale(mul(b2, N), -4), scale(sq(b2), 64),
                scale(add(*(sq(d) for d in degree)), 4),
                scale(add(*(sq(mon(i, i+15)) for i in range(15))), 48))
for i, j in PAIRS:
    term = add(*(mon(ID[tuple(sorted((i,k)))],
                     ID[tuple(sorted((j,k)))]+15)
                 for k in range(6) if k not in (i, j)))
    aggregate = add(aggregate, scale(mul(term, conj(term)), Q(24)))
assert not add(purity, scale(aggregate, -1))
# Scalar reduction (a quadratic in x, so three points decide it):
# 7/2 - 4x + 64x^2 = 35*121/917 + 7(524x-37)^2/27510 - (88/15)(1/2-x)^2.
for x in (Q(0), Q(1), Q(2)):
    assert Q(7,2)-4*x+64*x*x == Q(35*121,917)+Q(7,27510)*(524*x-37)**2-Q(88,15)*(Q(1,2)-x)**2
print("CC-1 aggregate: 35 pi_3 = 7/2 - 4x + 64x^2 + 4 sum p_i^2 + 48 sum |t_ij|^4"
      " + 24 sum |(TT^dag)_ij|^2 (x = b^2, unit norm); scalar completion identity exact")

residual = add(purity, scale(sq(N), Q(-35*121,917)), scale(rhs,-1))
assert not residual, list(residual.items())[:10]
print("CC-1 PASS: exact complex SOS identity, %d nonzero quartic monomials"
      % len(purity))
print("For normalized states: pi_3 >= 121/917 throughout the pair/complement family.")
