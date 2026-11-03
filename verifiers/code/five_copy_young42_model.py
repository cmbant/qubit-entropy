"""Exact rational [4,2] model and H-invariant basis for the five-copy block."""
import itertools

import flint
import numpy as np
import scipy.sparse as ss
import sympy as sp

import verify_five_copy_site_algebra as s5
from five_copy_young51_model import P as PRIME, monomials, select_columns, sym_power, apply_action

D = 5


def model_and_projector():
    m2, m4, m5 = [monomials(k) for k in (2, 4, 5)]
    i2, i4, i5 = [{e: j for j, e in enumerate(m)} for m in (m2, m4, m5)]
    e = [tuple(int(i == j) for i in range(D)) for j in range(D)]
    add = lambda *a: tuple(sum(v[i] for v in a) for i in range(D))
    sub = lambda a, b: tuple(x-y for x, y in zip(a, b))
    wedge = list(itertools.combinations(range(D), 2))
    gen = np.zeros((1050, 825), dtype=np.int64)
    col = 0
    for f in m2:
        for (i, j), (k, l) in itertools.combinations_with_replacement(wedge, 2):
            for (a, b), (c, d), sign in (
                ((i,k),(j,l),1), ((i,l),(j,k),-1),
                ((j,k),(i,l),-1), ((j,l),(i,k),1)):
                gen[i4[add(f,e[a],e[b])]*15+i2[add(e[c],e[d])],col] += sign
            col += 1
    assert col == 825
    er, ec, ev, fr, fc, fv = [], [], [], [], [], []
    for f in m4:
        for g in m2:
            col = i4[f]*15+i2[g]
            for i in range(D):
                if g[i]:
                    j = sub(g,e[i]).index(1)
                    er.append(i5[add(f,e[i])]*5+j); ec.append(col); ev.append(g[i])
    for f in m5:
        for j in range(D):
            col = i5[f]*5+j
            for i in range(D):
                if f[i]:
                    fr.append(i4[sub(f,e[i])]*15+i2[add(e[j],e[i])]); fc.append(col); fv.append(f[i])
    E = ss.csr_matrix((np.array(ev,dtype=np.int64),(er,ec)),shape=(630,1050))
    F = ss.csr_matrix((np.array(fv,dtype=np.int64),(fr,fc)),shape=(1050,630))
    assert not np.any(E@gen)
    assert flint.nmod_mat(630,1050,[int(x)%PRIME for x in E.toarray().ravel()],PRIME).rank()==630
    B = select_columns(gen)
    assert B.shape == (1050,420)
    N = F@E
    P40 = 40*ss.eye(1050,dtype=np.int64,format='csr')-14*N+N@N
    assert (P40@P40-40*P40).nnz == 0 and (E@P40).nnz == 0
    assert P40.nnz == 6115 and int(np.asarray(abs(P40).sum(axis=1)).max()) == 96
    return B, P40, E


def invariant_basis(B,E):
    """Return 4200x148 integer basis of Q4 tensor [4,2](Q5) H-invariants."""
    H = [k for k,p in enumerate(s5.PERMS) if set(p[3:]) == {3,4}]
    assert len(H) == 12
    tr = 0
    actions = []
    for k in H:
        R4, R5 = s5.RHO[1][k], s5.RHO[2][k]
        p = [0]
        Z = sp.eye(5)
        for j in range(1,7):
            Z = Z*R5
            p.append(sp.trace(Z))
        h = [sp.Integer(1)]
        for j in range(1,7):
            h.append(sum(p[i]*h[j-i] for i in range(1,j+1))/j)
        tr += sp.trace(R4)*(h[4]*h[2]-h[5]*sp.trace(R5))
        actions.append((np.asarray(R4,dtype=np.int64),
                        sym_power(np.asarray(R5,dtype=np.int64),4),
                        sym_power(np.asarray(R5,dtype=np.int64),2)))
    assert tr/12 == 148
    rng = np.random.default_rng(42)
    q = rng.integers(-2,3,size=(4,420,164),dtype=np.int64)
    X = np.einsum('pj,ajk->apk',B,q,optimize=True).reshape(4,70,15,164)
    Y = np.zeros_like(X)
    for R4,S4,S2 in actions:
        Y += apply_action(X,R4,S4,S2)
    U = select_columns(Y.reshape(4200,164))
    assert U.shape == (4200,148)
    U4 = U.reshape(4,70,15,148)
    for R4,S4,S2 in actions:
        assert np.array_equal(apply_action(U4,R4,S4,S2),U4)
    for a in range(4):
        assert not np.any(E@U4[a].reshape(1050,148))
    return U
