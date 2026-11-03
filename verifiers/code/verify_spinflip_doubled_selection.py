#!/usr/bin/env python3
"""SDS-1--SDS-3: exact Pauli selection in spinflip-doubled states.

Gaussian integer amplitudes and Pauli matrices with entries in
Z[i] provide exact checks, without normalization or floating point.
"""
from itertools import combinations, product

def spinflip(amp, m):
    out = {}
    for x, z in amp.items():
        y = tuple(1-v for v in x)
        out[y] = (-1)**sum(v == 0 for v in x)*z.conjugate()
    return out

def pauli_expectation(amp, word):
    total = 0j
    for x, z in amp.items():
        y = list(x)
        phase = 1+0j
        for i, p in enumerate(word):
            if p == 'X':
                y[i] = 1-x[i]
            elif p == 'Y':
                y[i] = 1-x[i]
                phase *= 1j if x[i] == 0 else -1j
            elif p == 'Z':
                phase *= 1 if x[i] == 0 else -1
        total += amp.get(tuple(y), 0).conjugate()*phase*z
    return total

def doubled(m, w):
    # |0>(u+v) + |1>J(v-u), u in weight-w, v on |1^m>.
    u = {}
    for k, S in enumerate(combinations(range(m), w)):
        bits = tuple(int(i in S) for i in range(m))
        u[bits] = complex(((3*k+1)%7)-3, ((5*k+2)%9)-4)
    v = {(1,)*m: 4+0j}
    a = {**u, **v}
    b = spinflip({**{x: -z for x, z in u.items()}, **v}, m)
    return a, b

def reduced_purity(a, b, m, S):
    # rho_S, with the distinguished qubit traced out.
    R = [i for i in range(m) if i not in S]
    size = 2**len(S)
    rho = [[0j]*size for _ in range(size)]
    for branch in (a, b):
        columns = {}
        for bits, z in branch.items():
            rest = tuple(bits[i] for i in R)
            local = sum(bits[i] << r for r, i in enumerate(S))
            columns.setdefault(rest, []).append((local, z))
        for items in columns.values():
            for i, vi in items:
                for j, vj in items:
                    rho[i][j] += vi*vj.conjugate()
    p = sum(abs(rho[i][j])**2
            for i in range(size) for j in range(size))
    assert p == int(p), p
    return int(p)

def check(m, w):
    a, b = doubled(m, w)
    distance = m-w
    assert 0 < w < m and 2*w != m and distance >= 2
    N = sum(abs(z)**2 for branch in (a, b)
            for z in branch.values())
    assert N == int(N)
    N = int(N)
    tested = 0
    # Exhaust every ordinary Pauli word of odd weight at most 3.
    for q in range(1, min(distance, 4), 2):
        for S in combinations(range(m), q):
            for labels in product('XYZ', repeat=q):
                word = ['I']*m
                for i, p in zip(S, labels):
                    word[i] = p
                assert pauli_expectation(a, word)+pauli_expectation(b, word) == 0j
                tested += 1
    # Equal branch norms and disjoint support make the distinguished
    # one-qubit marginal exactly I/2.
    assert not (set(a) & set(b))
    assert sum(abs(z)**2 for z in a.values()) == N//2
    assert sum(abs(z)**2 for z in b.values()) == N//2
    if (m, w) == (6, 2):
        for I in combinations(range(m), 3):
            lhs = 4*reduced_purity(a, b, m, I)
            rhs = 2*sum(reduced_purity(a, b, m, J)
                        for J in combinations(I, 2))-N*N
            assert lhs == rhs, (I, lhs, rhs)
        print('SDS-2 PASS: all 20 ordinary triple-purity relations')
    print('SDS-1 PASS: m=%d w=%d, %d exact Pauli zeros below distance %d' %
          (m, w, tested, distance))

if __name__ == '__main__':
    for m, w in ((5, 1), (6, 2), (7, 2), (8, 2)):
        check(m, w)
    print('SDS-3 PASS: one-uniform spinflip-doubled families')
