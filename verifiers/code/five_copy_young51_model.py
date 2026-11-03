"""Rational [5,1] GL(5) model and H=S3xS2 invariant basis for five copies.

This helper is portable when placed beside verify_five_copy_site_algebra.py.
"""
import itertools

import flint
import numpy as np
import sympy as sp

import verify_five_copy_site_algebra as s5

P = 2147483629
D = 5
assert sp.isprime(P)


def monomials(k):
    out = []
    for c in itertools.combinations_with_replacement(range(D), k):
        e = [0] * D
        for i in c: e[i] += 1
        out.append(tuple(e))
    return out


def select_columns(A):
    F = flint.nmod_mat(A.shape[0], A.shape[1], [int(x) % P for x in A.ravel()], P)
    R, r = F.rref()
    piv = [next(j for j in range(A.shape[1]) if int(R[i, j])) for i in range(r)]
    out = A[:, piv].copy()
    assert flint.nmod_mat(out.shape[0], out.shape[1],
                          [int(x) % P for x in out.ravel()], P).rank() == r
    return out


def sym_power(R, degree):
    mons = monomials(degree)
    idx = {v: i for i, v in enumerate(mons)}
    e = [tuple(int(i == j) for i in range(D)) for j in range(D)]
    S = np.zeros((len(mons), len(mons)), dtype=np.int64)
    for col, exp in enumerate(mons):
        terms = {(0,) * D: 1}
        for j in range(D):
            for _ in range(exp[j]):
                new = {}
                for a, coeff in terms.items():
                    for i in range(D):
                        c = int(R[i, j])
                        if c:
                            b = tuple(a[h] + e[i][h] for h in range(D))
                            new[b] = new.get(b, 0) + c * coeff
                terms = new
        for a, coeff in terms.items(): S[idx[a], col] = coeff
    return S


def apply_action(X, R4, S5, R5):
    Y = np.einsum('ab,bijk->aijk', R4, X, optimize=True)
    Y = np.einsum('ij,ajbk->aibk', S5, Y, optimize=True)
    return np.einsum('ij,abjk->abik', R5, Y, optimize=True)


def invariant_basis():
    """Return 2520x137 integer basis of H-invariants in Q4⊗[5,1](Q5)."""
    m4, m5, m6 = [monomials(k) for k in (4, 5, 6)]
    i5 = {v: i for i, v in enumerate(m5)}
    i6 = {v: i for i, v in enumerate(m6)}
    e = [tuple(int(i == j) for i in range(D)) for j in range(D)]
    add = lambda a, b: tuple(a[i] + b[i] for i in range(D))
    gen = np.zeros((630, 700), dtype=np.int64)
    col = 0
    for f in m4:
        for i, j in itertools.combinations(range(D), 2):
            gen[i5[add(f, e[i])] * D + j, col] += 1
            gen[i5[add(f, e[j])] * D + i, col] -= 1
            col += 1
    mul = np.zeros((210, 630), dtype=np.int64)
    for f in m5:
        for i in range(D): mul[i6[add(f, e[i])], i5[f] * D + i] = 1
    assert np.array_equal(mul @ gen, np.zeros((210, 700), dtype=np.int64))
    B = select_columns(gen)
    assert B.shape == (630, 420)  # ker(mul), since mul is surjective
    H = [k for k, p in enumerate(s5.PERMS) if set(p[3:]) == {3, 4}]
    assert len(H) == 12
    # Trace of [5,1](R5) = h5(R5) tr(R5) - h6(R5), the kernel
    # character of Sym^5(V)⊗V -> Sym^6(V).
    trace_sum = 0
    for k in H:
        R4 = s5.RHO[1][k]
        R5 = s5.RHO[2][k]
        powers = [0]
        Z = sp.eye(5)
        for j in range(1, 7):
            Z = Z * R5
            powers.append(sp.trace(Z))
        h = [sp.Integer(1)]
        for j in range(1, 7):
            h.append(sum(powers[i] * h[j-i] for i in range(1,j+1))/j)
        trace_sum += sp.trace(R4) * (h[5]*sp.trace(R5)-h[6])
    assert trace_sum/12 == 137
    rng = np.random.default_rng(51)
    q = rng.integers(-2, 3, size=(4, 420, 153), dtype=np.int64)
    X = np.einsum('pj,ajk->apk', B, q, optimize=True).reshape(4, 126, 5, 153)
    Y = np.zeros_like(X)
    actions = []
    for k in H:
        R4 = np.asarray(s5.RHO[1][k], dtype=np.int64)
        R5 = np.asarray(s5.RHO[2][k], dtype=np.int64)
        S5 = sym_power(R5, 5)
        actions.append((R4, S5, R5))
        Y += apply_action(X, R4, S5, R5)
    U = select_columns(Y.reshape(2520, 153))
    assert U.shape == (2520, 137)
    U4 = U.reshape(4, 126, 5, 137)
    for R4, S5, R5 in actions:
        assert np.array_equal(apply_action(U4, R4, S5, R5), U4)
    for a in range(4): assert np.array_equal(mul @ U4[a].reshape(630, 137), np.zeros((210, 137), dtype=np.int64))
    return U
