"""Y5-1: exact rational Young models for the leading five-copy two-flip blocks.

Uses [5,1] = ker(Sym^5 V ⊗ V -> Sym^6 V) and
[4,2] = ker(Sym^4 V ⊗ Sym^2 V -> Sym^5 V ⊗ V), V=Q^5.
Integer generator matrices are selected by a prime-field rank check.
"""
import itertools
import os
import sys

import flint
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_five_copy_site_algebra as s5

P = 2147483629
D = 5


def mons(k):
    out = []
    for c in itertools.combinations_with_replacement(range(D), k):
        e = [0] * D
        for i in c:
            e[i] += 1
        out.append(tuple(e))
    return out


def add(*es):
    return tuple(sum(e[i] for e in es) for i in range(D))


e = [tuple(int(i == j) for i in range(D)) for j in range(D)]
M2, M4, M5, M6 = [mons(k) for k in (2, 4, 5, 6)]
i2, i4, i5, i6 = [{x: j for j, x in enumerate(M)} for M in (M2, M4, M5, M6)]


def rank_mod(A):
    B = flint.nmod_mat(A.shape[0], A.shape[1], [int(x) % P for x in A.ravel()], P)
    return B.rank()


def select_cols(A):
    B = flint.nmod_mat(A.shape[0], A.shape[1], [int(x) % P for x in A.ravel()], P)
    R, r = B.rref()
    piv = [next(j for j in range(A.shape[1]) if int(R[i, j])) for i in range(r)]
    out = A[:, piv].copy()
    assert rank_mod(out) == r
    return out


def model51():
    gen = np.zeros((len(M5) * D, len(M4) * 10), dtype=np.int64)
    col = 0
    for f in M4:
        for i, j in itertools.combinations(range(D), 2):
            gen[i5[add(f, e[i])] * D + j, col] += 1
            gen[i5[add(f, e[j])] * D + i, col] -= 1
            col += 1
    assert col == gen.shape[1] == 700
    # Multiplication map. Checking the proposed generators are killed is exact.
    mul = np.zeros((len(M6), len(M5) * D), dtype=np.int64)
    for f in M5:
        for i in range(D):
            mul[i6[add(f, e[i])], i5[f] * D + i] = 1
    assert np.array_equal(mul @ gen, np.zeros((len(M6), gen.shape[1]), dtype=np.int64))
    # Multiplication is surjective: each target monomial has a variable factor.
    assert all(any(mul[j]) for j in range(len(M6)))
    rank = rank_mod(gen)
    assert rank == 420 == gen.shape[0] - mul.shape[0]
    return rank, select_cols(gen)


def model42():
    wedges = list(itertools.combinations(range(D), 2))
    pairs = list(itertools.combinations_with_replacement(wedges, 2))
    gen = np.zeros((len(M4) * len(M2), len(M2) * len(pairs)), dtype=np.int64)
    col = 0
    for f in M2:
        for (i, j), (k, l) in pairs:
            terms = [((i, k), (j, l), 1), ((i, l), (j, k), -1),
                     ((j, k), (i, l), -1), ((j, l), (i, k), 1)]
            for (a, b), (c, d), sign in terms:
                gen[i4[add(f, e[a], e[b])] * len(M2) + i2[add(e[c], e[d])], col] += sign
            col += 1
    assert col == gen.shape[1] == 825
    # Polarisation: delta(x^f ⊗ y^g) = sum_i g_i x^(f+e_i) ⊗ y^(g-e_i).
    polar = np.zeros((len(M5) * D, len(M4) * len(M2)), dtype=np.int64)
    for f in M4:
        for g in M2:
            for i in range(D):
                if not g[i]:
                    continue
                rem = tuple(g[j] - (i == j) for j in range(D))
                j = rem.index(1)
                polar[i5[add(f, e[i])] * D + j, i4[f] * len(M2) + i2[g]] += g[i]
    assert np.array_equal(polar @ gen, np.zeros((polar.shape[0], gen.shape[1]), dtype=np.int64))
    rank = rank_mod(gen)
    assert rank == 420
    assert rank_mod(polar) == 630  # hence ker(delta) has dimension 420
    return rank, select_cols(gen)


def complete_traces(A, kmax=6):
    power = sp.eye(A.rows)
    p = [0]
    for k in range(1, kmax + 1):
        power = power * A
        p.append(sp.trace(power))
    h = [sp.Integer(1)]
    for k in range(1, kmax + 1):
        h.append(sum(p[i] * h[k - i] for i in range(1, k + 1)) / k)
    return h


def invariant_dims():
    H = [k for k, p in enumerate(s5.PERMS) if set(p[3:]) == {3, 4}]
    assert len(H) == 12
    tr51 = tr42 = 0
    for k in H:
        R4 = s5.RHO[1][k]
        R5 = s5.RHO[2][k]
        h = complete_traces(R5)
        tr51 += sp.trace(R4) * (h[5] * sp.trace(R5) - h[6])
        tr42 += sp.trace(R4) * (h[4] * h[2] - h[5] * sp.trace(R5))
    assert tr51 / 12 == 137
    assert tr42 / 12 == 148
    return int(tr51 / 12), int(tr42 / 12)


def sym_power(R, degree):
    """Integer action on monomials in Sym^degree Q^5, x_j -> sum_i R_ij x_i."""
    M = mons(degree)
    ind = {v: i for i, v in enumerate(M)}
    S = np.zeros((len(M), len(M)), dtype=np.int64)
    zero = (0,) * D
    for col, exp in enumerate(M):
        terms = {zero: 1}
        for j in range(D):
            for _ in range(exp[j]):
                new = {}
                for mon, coeff in terms.items():
                    for i in range(D):
                        c = int(R[i, j])
                        if c:
                            key = add(mon, e[i])
                            new[key] = new.get(key, 0) + c * coeff
                terms = new
        for mon, coeff in terms.items():
            S[ind[mon], col] = coeff
    return S


def apply_action(X, R4, S1, S2):
    """Act on tensor (Q4)⊗Sym^p(Q5)⊗Sym^q(Q5), with q=1 or2."""
    # The intermediate arrays are small (at most 4*1050*164 int64 entries).
    Y = np.einsum('ab,bijk->aijk', R4, X, optimize=True)
    Y = np.einsum('ij,ajbk->aibk', S1, Y, optimize=True)
    Y = np.einsum('ij,abjk->abik', S2, Y, optimize=True)
    return Y


def invariant_basis(B, degree1, degree2, target_dim, seed):
    """Rational H-invariants of Q4⊗ker(delta), in ambient polynomial coordinates."""
    H = [k for k, p in enumerate(s5.PERMS) if set(p[3:]) == {3, 4}]
    assert len(H) == 12
    n1, n2 = len(mons(degree1)), len(mons(degree2))
    assert B.shape == (n1 * n2, 420)
    rng = np.random.default_rng(seed)
    K = target_dim + 16
    Q = rng.integers(-2, 3, size=(4, 420, K), dtype=np.int64)
    X = np.einsum('pj,ajk->apk', B, Q, optimize=True).reshape(4, n1, n2, K)
    Y = np.zeros_like(X)
    cache = {}
    for k in H:
        R4 = np.array(s5.RHO[1][k]).astype(np.int64)
        R5 = np.array(s5.RHO[2][k]).astype(np.int64)
        S1 = sym_power(R5, degree1)
        S2 = R5 if degree2 == 1 else sym_power(R5, degree2)
        cache[k] = (R4, S1, S2)
        Y += apply_action(X, R4, S1, S2)
    A = Y.reshape(4 * n1 * n2, K)
    print('  projected candidate shape', A.shape, 'max abs', np.abs(A).max(), flush=True)
    assert rank_mod(A) == target_dim
    U = select_cols(A)
    assert U.shape[1] == target_dim
    # Exact invariance under all 12 group elements; integer equality, no tolerance.
    U4 = U.reshape(4, n1, n2, target_dim)
    for R4, S1, S2 in cache.values():
        assert np.array_equal(apply_action(U4, R4, S1, S2), U4)
    return U


r51, B51 = model51()
print('[5,1] exact kernel rank', r51, flush=True)
r42, B42 = model42()
print('[4,2] exact kernel rank', r42, flush=True)
m51, m42 = invariant_dims()
print('H invariant multiplicities', (m51, m42), flush=True)
U51 = invariant_basis(B51, 5, 1, m51, 51)
print('[5,1] exact invariant basis', U51.shape, flush=True)
U42 = invariant_basis(B42, 4, 2, m42, 42)
print('[4,2] exact invariant basis', U42.shape, flush=True)
print('Y5-1 PASS', flush=True)
