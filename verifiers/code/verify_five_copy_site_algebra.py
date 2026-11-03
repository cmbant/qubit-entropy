"""Exact five-copy one-site Schur algebra and spin-flip maps (S5-1).

The weight-m highest-weight vectors are the kernel of the qubit raising
operator.  For m=1,2 their S5 modules have dimensions 4,5.  We construct
matrix units by finite-group Fourier inversion, apply the physical spin-flip
as an integer operation on 32x32 matrices, and check the rational result.
This is a local algebra check; it does not certify the seven-site SDP dual.
"""
import itertools

import numpy as np
import sympy as sp


R = 5
PERMS = list(itertools.permutations(range(R)))
PINDEX = {p: i for i, p in enumerate(PERMS)}
SUBSETS = {m: list(itertools.combinations(range(R), m)) for m in range(3)}


def perm_action(p, m):
    ss = SUBSETS[m]
    index = {s: i for i, s in enumerate(ss)}
    M = sp.zeros(len(ss))
    for j, s in enumerate(ss):
        M[index[tuple(sorted(p[k] for k in s))], j] = 1
    return M


def highest_weight(m):
    if m == 0:
        return sp.ones(1, 1)
    rows = SUBSETS[m - 1]
    cols = SUBSETS[m]
    J = sp.zeros(len(rows), len(cols))
    row_index = {s: i for i, s in enumerate(rows)}
    for j, s in enumerate(cols):
        for k in s:
            J[row_index[tuple(x for x in s if x != k)], j] = 1
    return sp.Matrix.hstack(*J.nullspace())


V = {m: highest_weight(m) for m in range(3)}
D = {m: V[m].cols for m in V}
UDIM = {m: R - 2 * m + 1 for m in V}
assert D == {0: 1, 1: 4, 2: 5}
assert sum(D[m] * UDIM[m] for m in V) == 32
L = {m: (V[m].T * V[m]).inv() * V[m].T for m in V}
RHO = {m: [L[m] * perm_action(p, m) * V[m] for p in PERMS] for m in V}
for m in V:
    for k, p in enumerate(PERMS):
        assert V[m] * RHO[m][k] == perm_action(p, m) * V[m]
        assert all(x.q == 1 for x in RHO[m][k])
print("dimensions", D, "weight multiplicities", UDIM, flush=True)
print("Gram determinants", {m: (V[m].T * V[m]).det() for m in V}, flush=True)

COORDS = [(m, i, j) for m in V for i in range(D[m]) for j in range(D[m])]
N = len(COORDS)
assert N == 42


def op(p):
    M = np.zeros((32, 32), dtype=np.int64)
    for x in range(32):
        y = sum(((x >> (R - 1 - i)) & 1) << (R - 1 - p[i]) for i in range(R))
        M[y, x] = 1
    return M


OPS = np.array([op(p) for p in PERMS])
PINV = [PINDEX[tuple(np.argsort(p))] for p in PERMS]


def expand_weight(Vm, m):
    arr = sp.zeros(32, Vm.cols)
    for j, s in enumerate(SUBSETS[m]):
        arr[sum(1 << (R - 1 - k) for k in s), :] = Vm[j, :]
    return arr


V32 = {m: expand_weight(V[m], m) for m in V}
L32 = {m: (V32[m].T * V32[m]).inv() * V32[m].T for m in V}
EMB_NUM = []
for m, i, j in COORDS:
    coeff = np.array([int(D[m] * RHO[m][PINV[k]][j, i]) for k in range(120)], dtype=np.int64)
    E = np.einsum("k,kij->ij", coeff, OPS, optimize=True)
    EMB_NUM.append(E)
    for mm in V:
        got = L32[mm] * sp.Matrix(E) * V32[mm]
        want = 120 * (sp.eye(D[m])[:, i] * sp.eye(D[m])[j, :]) if mm == m else sp.zeros(D[mm])
        assert got == want
print("exact matrix units checked", N, flush=True)


def flip(M, copies):
    eps = np.array([[0, 1], [-1, 0]], dtype=np.int64)
    U = np.ones((1, 1), dtype=np.int64)
    for c in range(R):
        U = np.kron(U, eps if c in copies else np.eye(2, dtype=np.int64))
    G = U.T @ M @ U
    axes = list(range(2 * R))
    for c in copies:
        axes[c], axes[R + c] = axes[R + c], axes[c]
    return G.reshape((2,) * (2 * R)).transpose(axes).reshape(32, 32)


def flip_map(copies):
    out = sp.zeros(N)
    for col, E in enumerate(EMB_NUM):
        A = sp.Matrix(flip(E, copies))
        for row, (m, i, j) in enumerate(COORDS):
            out[row, col] = (L32[m] * A * V32[m])[i, j] / 120
    return out


for copies in [(4,), (3, 4)]:
    T = flip_map(copies)
    den = sp.ilcm(*[int(x.q) for x in T])
    assert den == 30
    assert T * T == sp.eye(N)
    Ti = np.array([[int(den * x) for x in row] for row in T.tolist()], dtype=np.int64)
    E = np.stack(EMB_NUM)
    for col in range(N):
        lhs = np.einsum("k,kij->ij", Ti[:, col], E, optimize=True)
        rhs = int(den) * flip(E[col], copies)
        assert np.array_equal(lhs, rhs)
    print("copies", copies, "denominator", den, "max numerator", np.max(np.abs(Ti)),
          "physical reconstruction PASS", "involution PASS", flush=True)
print("S5-1 PASS", flush=True)
