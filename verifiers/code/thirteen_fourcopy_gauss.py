"""Exact four-copy graph contractions in spin-zero/spin-one covectors.

Each local covector has affine support of dimension two and a linear sign.
Substitution into four graph-state quadratic phases leaves a binary quadratic
Gauss sum on 2n variables. Numba accelerates integer operations only.
The callers use n <= 13, so every integer sum is bounded by 2^26.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import itertools
import numpy as np
from numba import njit
from verify_balanced_purity_13_15 import GRAPH14, graph_rows

E = np.array([[0, 1], [-1, 0]], np.int64)
T = []
F = []
for a, b, c, d in itertools.product(range(2), repeat=4):
    x = [a, b, c, d]
    T.append([E[a, b] * E[c, d], E[a, c] * E[b, d]])
    F.append(
        [
            [
                E[x[0], x[r]]
                * ((1 if x[s[0]] == 0 else -1) if x[s[0]] == x[s[1]] else 0)
                for r in [1, 2, 3]
                for s in [[i for i in range(4) if i not in [0, r]]]
            ],
            [
                E[x[0], x[r]] * (1 if x[s[0]] == x[s[1]] else 0)
                for r in [1, 2, 3]
                for s in [[i for i in range(4) if i not in [0, r]]]
            ],
            [
                E[x[0], x[r]] * (1 if x[s[0]] != x[s[1]] else 0)
                for r in [1, 2, 3]
                for s in [[i for i in range(4) if i not in [0, r]]]
            ],
        ]
    )
T = np.array(T)
F = np.array(F)
assert all(np.array_equal(F[:, a].T @ F[:, a], F[:, 0].T @ F[:, 0]) for a in range(3))


def affine(v):
    idx = np.flatnonzero(v)
    assert len(idx) == 4
    a = int(idx[0])
    b = int(idx[1]) ^ a
    c = int(idx[2]) ^ a
    assert a ^ b ^ c == idx[3]
    e0 = int(v[a] < 0)
    e1 = int(v[a ^ b] < 0) ^ e0
    e2 = int(v[a ^ c] < 0) ^ e0
    assert int(v[a ^ b ^ c] < 0) == e0 ^ e1 ^ e2
    return [a, b, c, e0, e1, e2]


TV = np.array([affine(T[:, i]) for i in range(2)], np.int64)
TF = np.array([[affine(F[:, a, i]) for i in range(3)] for a in range(3)], np.int64)
PAR = np.array([i.bit_count() % 2 for i in range(16)], np.int64)


@njit
def least(mask):
    i = 0
    while (mask & 1) == 0:
        mask >>= 1
        i += 1
    return i


@njit
def gauss(rows, l, c, n):
    active = (np.uint64(1) << n) - np.uint64(1)
    pairs = 0
    while True:
        i = -1
        for k in range(n):
            if active >> k & 1 and rows[k] & active:
                i = k
                break
        if i < 0:
            break
        j = least(rows[i] & active)
        active ^= (np.uint64(1) << i) | (np.uint64(1) << j)
        ai = rows[i] & active
        aj = rows[j] & active
        li = (l >> i) & 1
        lj = (l >> j) & 1
        c ^= li & lj
        if li:
            l ^= aj
        if lj:
            l ^= ai
        l ^= ai & aj
        l &= active
        for k in range(n):
            if active >> k & 1:
                if ai >> k & 1:
                    rows[k] ^= aj
                if aj >> k & 1:
                    rows[k] ^= ai
                rows[k] &= active
        pairs += 1
    if l & active:
        return np.int64(0)
    return np.int64(1 if c == 0 else -1) * np.int64(1 << (n - pairs))


@njit
def value(local, edges, p, jflip, qone):
    n = len(local)
    rows = np.zeros(2 * n, np.uint64)
    l = np.uint64(0)
    c = (jflip % 2) * qone
    J = 0 if jflip == 0 else (1 if jflip == 1 else 3)
    for i in range(n):
        a, b, d, e0, e1, e2 = local[i]
        c ^= e0
        ll1 = e1
        ll2 = e2
        if p >> i & 1:
            c ^= PAR[a & J]
            ll1 ^= PAR[b & J]
            ll2 ^= PAR[d & J]
        l ^= np.uint64(ll1) << (2 * i)
        l ^= np.uint64(ll2) << (2 * i + 1)
    for e in range(len(edges)):
        i, k = edges[e]
        a, b, d = local[i, :3]
        aa, bb, dd = local[k, :3]
        c ^= PAR[a & aa]
        l ^= np.uint64(PAR[b & aa]) << (2 * i)
        l ^= np.uint64(PAR[d & aa]) << (2 * i + 1)
        l ^= np.uint64(PAR[bb & a]) << (2 * k)
        l ^= np.uint64(PAR[dd & a]) << (2 * k + 1)
        for ii in range(2):
            for kk in range(2):
                v = PAR[local[i, ii + 1] & local[k, kk + 1]]
                rows[2 * i + ii] ^= np.uint64(v) << (2 * k + kk)
                rows[2 * k + kk] ^= np.uint64(v) << (2 * i + ii)
    return gauss(rows, l, c, 2 * n)


@njit
def tensor(G, sites, axes, jflip):
    n = len(G)
    b = len(sites)
    radix = np.full(n, 2, np.int64)
    axis = np.full(n, -1, np.int64)
    for i in range(b):
        radix[sites[i]] = 3
        axis[sites[i]] = axes[i]
    size = 1
    for r in radix:
        size *= r
    edges = np.array(
        [(i, k) for i in range(n) for k in range(i + 1, n) if G[i] >> k & 1], np.int64
    ).reshape(-1, 2)
    p = 0
    qone = len(edges) % 2
    for i in range(n):
        deg = 0
        for k in range(n):
            deg += G[i] >> k & 1
        p |= (1 ^ (deg % 2)) << i
    out = np.empty(size, np.int64)
    local = np.empty((n, 6), np.int64)
    for x in range(size):
        tmp = x
        for i in range(n - 1, -1, -1):
            r = tmp % radix[i]
            tmp //= radix[i]
            local[i] = TV[r] if axis[i] < 0 else TF[axis[i], r]
        out[x] = value(local, edges, p, jflip, qone)
    return out


def stabilizers(G, shadow=False, w=5):
    n = len(G)
    p = sum((1 ^ (r.bit_count() % 2)) << i for i, r in enumerate(G))
    out = []
    for x in range(1 << n):
        z = p if shadow else 0
        for i in range(n):
            if x >> i & 1:
                z ^= G[i]
        if (x | z).bit_count() != w:
            continue
        sites = [i for i in range(n) if (x | z) >> i & 1]
        axes = [1 if (x & z) >> i & 1 else (0 if x >> i & 1 else 2) for i in sites]
        out.append((sites, axes, x, z))
    return out


def graph13():
    """The canonical pivot-and-delete representative independently checked by SH-1."""

    def lc(G, v):
        G = G[:]
        ne = [i for i in range(len(G)) if G[v] >> i & 1]
        for i, j in itertools.combinations(ne, 2):
            G[i] ^= 1 << j
            G[j] ^= 1 << i
        return G

    G = graph_rows(GRAPH14)
    G = lc(lc(lc(G, 4), 1), 4)
    return [
        sum((G[i] >> j & 1) << (j - int(j > 4)) for j in range(14) if j != 4)
        for i in range(14)
        if i != 4
    ]
