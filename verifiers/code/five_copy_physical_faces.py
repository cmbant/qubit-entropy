"""Rational physical ICO faces for the four exact two-flip feature models.

Only independent feature columns are transported.  Reconstructing their
ambient columns first avoids forming the dense rational matrix F.  The
physical kernel is then obtained by integer HNF and lattice reduction.
"""
import itertools
import math

import flint
import numpy as np

import verify_five_copy_site_algebra as s5
from five_copy_symmetric_models import invariant_bases
from five_copy_young51_model import invariant_basis, monomials, sym_power
from five_copy_young42_model import model_and_projector, invariant_basis as invariant42


def symmetric_gram_numerator(n):
    G = np.asarray(s5.V[2].T*s5.V[2], dtype=np.int64)
    factorials = np.array([math.prod(math.factorial(v) for v in e)
                           for e in monomials(n)], dtype=object)
    H = factorials[:, None]*sym_power(G, n).astype(object)
    assert np.array_equal(H, H.T)
    return H


def models():
    U0, U12 = invariant_bases()
    U51 = invariant_basis()
    B, _, E = model_and_projector()
    U42 = invariant42(B, E)
    G4 = np.asarray(s5.V[1].T*s5.V[1], dtype=object)
    G5 = np.asarray(s5.V[2].T*s5.V[2], dtype=object)
    return [
        ('7', U0, [symmetric_gram_numerator(7)], math.factorial(7), 2),
        ('6', U12, [G4, symmetric_gram_numerator(6)], math.factorial(6), 8),
        ('51', U51, [G4, symmetric_gram_numerator(5), G5], math.factorial(5), 12),
        ('42', U42, [G4, symmetric_gram_numerator(4), symmetric_gram_numerator(2)],
         math.factorial(4)*math.factorial(2), 20),
    ]


def integer_matrix(A):
    return flint.fmpz_mat(A.shape[0], A.shape[1], [int(v) for v in A.ravel()])


def rational_matrix(A):
    return flint.fmpq_mat(A.shape[0], A.shape[1], [int(v) for v in A.ravel()])


def independent_columns(A, rank):
    p = 2147483629
    R, r = flint.nmod_mat(A.shape[0], A.shape[1],
                         [int(v) % p for v in A.ravel()], p).rref()
    assert r == rank
    return [next(j for j in range(A.shape[1]) if R[i, j]) for i in range(rank)]


def apply_gram(Y, factors):
    X = Y.reshape(*[G.shape[1] for G in factors], Y.shape[1]).astype(object)
    for i, G in enumerate(factors):
        X = np.moveaxis(np.tensordot(G, X, axes=([1], [i])), 0, i)
    return X.reshape(Y.shape)


def physical_columns(U, factors, Q):
    """Return the numerator of U^T H U (U^T U)^-1 Q over the model's denominator."""
    Uq = rational_matrix(U)
    Eu = Uq.transpose()*Uq
    Yq = Uq*Eu.solve(rational_matrix(Q))
    # The stored feature columns came from integral ambient columns.  Their
    # reconstruction in the complete invariant space must recover integers.
    assert all(v.denom() == 1 for v in Yq.entries())
    Y = np.array([[int(Yq[i, j]) for j in range(Yq.ncols())]
                  for i in range(Yq.nrows())], dtype=object)
    assert np.array_equal(U.astype(object).T@Y, Q.astype(object))
    return integer_matrix(U.astype(object).T@apply_gram(Y, factors))


def short_kernel(C, rank):
    n = C.nrows()
    H, T = C.hnf(transform=True)
    assert T*C == H and abs(T.det()) == 1
    K = flint.fmpz_mat(n-rank, n,
                      [T[i, j] for i in range(rank, n) for j in range(n)])
    assert K*C == flint.fmpz_mat(n-rank, rank)
    L, S = K.lll(transform=True)
    assert S*K == L and abs(S.det()) == 1
    return L
