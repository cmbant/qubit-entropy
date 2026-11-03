"""Complete rational H-invariant Young models [6,1] and [5,2].

The models are kernels of the Pieri multiplication/polarization maps.
A modular minor gives an exact lower rank bound, while the surjective
map and the exact H-character average give matching upper bounds.
"""
import itertools
import numpy as np
import sympy as sp
import scipy.sparse as ss
from functools import lru_cache
import verify_five_copy_site_algebra as f
from five_copy_young51_model import monomials, sym_power, select_columns
import flint
P = 2147483629

@lru_cache(None)
def invariant_basis(k, l):
    assert (k, l) in ((6, 1), (5, 2))
    mk, ml = (monomials(k), monomials(l))
    ik, il = ({m: i for i, m in enumerate(mk)}, {m: i for i, m in enumerate(ml)})
    e = [tuple((int(i == j) for i in range(5))) for j in range(5)]
    add = lambda *xs: tuple((sum((x[i] for x in xs)) for i in range(5)))
    if l == 1:
        small = monomials(k - 1)
        G = np.zeros((len(mk) * 5, len(small) * 10), np.int64)
        c = 0
        for a in small:
            for i, j in itertools.combinations(range(5), 2):
                G[ik[add(a, e[i])] * 5 + j, c] += 1
                G[ik[add(a, e[j])] * 5 + i, c] -= 1
                c += 1
        high = monomials(k + 1)
        ih = {m: i for i, m in enumerate(high)}
        E = ss.lil_matrix((len(high), len(mk) * 5), dtype=np.int64)
        for i, a in enumerate(mk):
            for j in range(5):
                E[ih[add(a, e[j])], i * 5 + j] = 1
        E = E.tocsr()
        expected = 5 * len(mk) - len(high)
    else:
        small = monomials(k - 2)
        wedges = list(itertools.combinations(range(5), 2))
        G = np.zeros((len(mk) * len(ml), len(small) * 55), np.int64)
        c = 0
        for a in small:
            for (i, j), (u, v) in itertools.combinations_with_replacement(wedges, 2):
                for (aa, bb), (cc, dd), sg in [((i, u), (j, v), 1), ((i, v), (j, u), -1), ((j, u), (i, v), -1), ((j, v), (i, u), 1)]:
                    G[ik[add(a, e[aa], e[bb])] * len(ml) + il[add(e[cc], e[dd])], c] += sg
                c += 1
        high = monomials(k + 1)
        ih = {m: i for i, m in enumerate(high)}
        E = ss.lil_matrix((len(high) * 5, len(mk) * len(ml)), dtype=np.int64)
        for i, a in enumerate(mk):
            for j, b in enumerate(ml):
                for u in range(5):
                    if b[u]:
                        v = list(b)
                        v[u] -= 1
                        qq = v.index(1)
                        E[ih[add(a, e[u])] * 5 + qq, i * len(ml) + j] += b[u]
        E = E.tocsr()
        expected = len(mk) * len(ml) - len(high) * 5
    assert not np.any(E @ G)
    B = select_columns(G)
    assert B.shape[1] == expected
    assert flint.nmod_mat(E.shape[0], E.shape[1], [int(v) % P for v in E.toarray().ravel()], P).rank() == E.shape[0]
    H = [i for i, p in enumerate(f.PERMS) if set(p[3:]) == {3, 4}]
    dim = 60 if l == 1 else 75
    trace_sum = 0
    for i in H:
        R = f.RHO[2][i]
        Z = sp.eye(5)
        tr = [0]
        for j in range(1, 8):
            Z = Z * R
            tr.append(sp.trace(Z))
        h = [sp.Integer(1)]
        for j in range(1, 8):
            h.append(sum((tr[a] * h[j - a] for a in range(1, j + 1))) / j)
        trace_sum += h[k] * h[l] - h[k + 1] * h[l - 1]
    assert trace_sum == 12 * dim
    rng = np.random.default_rng(61 if l == 1 else 52)
    assert 2 * B.shape[1] * int(abs(B).max()) < 2**63
    X = (B @ rng.integers(-2, 3, size=(B.shape[1], dim + 16), dtype=np.int64)).reshape(len(mk), len(ml), dim + 16)
    Y = np.zeros_like(X)
    actions = []
    for i in H:
        R = np.asarray(f.RHO[2][i], dtype=np.int64)
        S0 = sym_power(R, k)
        S1 = sym_power(R, l)
        assert 12 * len(mk) * len(ml) * int(abs(S0).max()) * int(abs(S1).max()) * int(abs(X).max()) < 2**63
        actions.append((S0, S1))
        Y += np.einsum('ij,jbk,ab->iak', S0, X, S1, optimize=True)
    U = select_columns(Y.reshape(len(mk) * len(ml), dim + 16))
    assert U.shape[1] == dim and (not np.any(E @ U))
    for A, C in actions:
        assert len(mk) * len(ml) * int(abs(A).max()) * int(abs(C).max()) * int(abs(U).max()) < 2**63
        yy = np.einsum('ij,jbk,ab->iak', A, U.reshape(len(mk), len(ml), dim), C, optimize=True)
        assert np.array_equal(yy.reshape(U.shape), U)
    return U
