"""Exact local Pauli-algebra bases for stabilizer-code marginal compression.

All matrix entries in basis() are Gaussian integers of modulus zero or one.
The integer operations performed in NumPy stay exactly representable.
"""

import numpy as np
from math import gcd
from functools import reduce


def pairing(v, w, q):
    mask = (1 << q) - 1
    return (
        ((v & mask) & (w >> q)).bit_count() + ((v >> q) & (w & mask)).bit_count()
    ) & 1


def rank(words):
    p = {}
    for w in words:
        while w:
            j = w.bit_length() - 1
            if j in p:
                w ^= p[j]
            else:
                p[j] = w
                break
    return len(p)


def basis(restrictions, q):
    rest = list(restrictions)
    xs = []
    zs = []
    assert rank(rest) == len(rest)
    while True:
        pairs = [
            (i, j)
            for i in range(len(rest))
            for j in range(i + 1, len(rest))
            if pairing(rest[i], rest[j], q)
        ]
        if not pairs:
            break
        i, j = pairs[0]
        u, v = rest[i], rest[j]
        xs.append(u)
        zs.append(v)
        rest = [
            w ^ (u if pairing(w, v, q) else 0) ^ (v if pairing(w, u, q) else 0)
            for k, w in enumerate(rest)
            if k not in (i, j)
        ]
    e = len(xs)
    c = len(rest)
    zs += rest
    canonical = xs + zs
    assert rank(canonical) == len(restrictions)
    assert all(rank(canonical + [w]) == len(canonical) for w in restrictions)
    for w in range(1, 1 << (2 * q)):
        if len(zs) == q:
            break
        if all(not pairing(w, a, q) for a in zs + xs) and rank(zs + [w]) > len(zs):
            zs.append(w)
    assert len(zs) == q
    for j in range(e, q):
        for w in range(1, 1 << (2 * q)):
            if all(pairing(w, a, q) == int(k == j) for k, a in enumerate(zs)) and all(
                not pairing(w, a, q) for a in xs
            ):
                xs.append(w)
                break
    assert len(xs) == q
    size = 1 << q
    idx = np.arange(size)

    def apply(v, a):
        x, z = v & (size - 1), v >> q
        signs = np.array([(-1) ** ((int(j) & z).bit_count()) for j in idx])
        out = np.empty_like(a)
        out[idx ^ x] = (1j ** ((x & z).bit_count())) * signs * a
        return out

    for j in range(size):
        a = np.zeros(size, complex)
        a[j] = 1
        for v in zs:
            a += apply(v, a)
        if np.any(a):
            break
    common = reduce(gcd, [abs(int(t)) for t in np.r_[a.real, a.imag] if t])
    a /= common
    gram = int(np.vdot(a, a).real)
    V = []
    for b in range(size):
        v = a.copy()
        for j, x in enumerate(xs):
            if b >> j & 1:
                v = apply(x, v)
        V.append(v)
    V = np.array(V).T
    assert set(np.ravel(abs(V) ** 2)) <= {0.0, 1.0}
    assert gram > 0 and gram & (gram - 1) == 0
    assert np.array_equal(V.conj().T @ V, gram * np.eye(size))
    for j, (x, z) in enumerate(zip(xs, zs)):

        def matrix_apply(word):
            a, b = word & (size - 1), word >> q
            signs = np.array([(-1) ** ((int(t) & b).bit_count()) for t in idx])
            out = np.empty_like(V)
            out[idx ^ a, :] = (1j ** ((a & b).bit_count())) * signs[:, None] * V
            return out

        assert np.array_equal(
            matrix_apply(z),
            V * np.array([(-1) ** ((b >> j) & 1) for b in idx])[None, :],
        )
        assert np.array_equal(matrix_apply(x), V[:, idx ^ (1 << j)])
    blocks = [
        [
            sum(((u >> j) & 1) << (e + j) for j in range(c))
            + sum(((v >> j) & 1) << (e + c + j) for j in range(q - e - c))
            for v in range(1 << (q - e - c))
        ]
        for u in range(1 << c)
    ]
    return V, gram, e, c, blocks
