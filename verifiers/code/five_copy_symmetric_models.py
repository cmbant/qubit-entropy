"""Exact H-invariant models for the g34 all-symmetric [7] and [6] blocks."""
import itertools
import numpy as np
import sympy as sp

import verify_five_copy_site_algebra as s5
from five_copy_young51_model import monomials, select_columns, sym_power


def _complete_traces(R,kmax):
    powers=[0]; X=sp.eye(R.rows)
    for k in range(1,kmax+1):
        X=X*R; powers.append(sp.trace(X))
    h=[sp.Integer(1)]
    for k in range(1,kmax+1):
        h.append(sum(powers[i]*h[k-i] for i in range(1,k+1))/k)
    return h


def invariant_bases():
    """Return integer U0:330x33 and U12:840x77, complete H invariants."""
    H=[k for k,p in enumerate(s5.PERMS) if set(p[3:])=={3,4}]
    assert len(H)==12
    actions=[]; dim0=dim12=0
    for k in H:
        R4=np.asarray(s5.RHO[1][k],dtype=np.int64)
        R5=np.asarray(s5.RHO[2][k],dtype=np.int64)
        S6=sym_power(R5,6);S7=sym_power(R5,7)
        actions.append((R4,S6,S7))
        h=_complete_traces(s5.RHO[2][k],7)
        dim0 += h[7]
        dim12 += sp.trace(s5.RHO[1][k])*h[6]
    assert dim0/12==33 and dim12/12==77

    rng=np.random.default_rng(7)
    X0=rng.integers(-2,3,size=(330,49),dtype=np.int64)
    Y0=np.zeros_like(X0)
    for _,_,S7 in actions: Y0 += S7@X0
    U0=select_columns(Y0)
    assert U0.shape==(330,33)

    rng=np.random.default_rng(6)
    X12=rng.integers(-2,3,size=(4,210,93),dtype=np.int64)
    Y12=np.zeros_like(X12)
    for R4,S6,_ in actions:
        Z=np.einsum('ab,bik->aik',R4,X12,optimize=True)
        Y12 += np.einsum('ij,ajk->aik',S6,Z,optimize=True)
    U12=select_columns(Y12.reshape(840,93))
    assert U12.shape==(840,77)
    for R4,S6,S7 in actions:
        assert np.array_equal(S7@U0,U0)
        Z=np.einsum('ab,bik->aik',R4,U12.reshape(4,210,77),optimize=True)
        assert np.array_equal(np.einsum('ij,ajk->aik',S6,Z,optimize=True).reshape(840,77),U12)
    return U0,U12


def symmetric_lookup(k):
    mons=monomials(k); lookup={m:i for i,m in enumerate(mons)}
    q=np.array(list(itertools.product(range(5),repeat=k)),dtype=np.int8)
    group=np.array([lookup[tuple(int(np.sum(row==j)) for j in range(5))] for row in q],dtype=np.int32)
    order=np.argsort(group,kind='stable')
    starts=np.r_[0,np.where(np.diff(group[order]))[0]+1]
    assert len(starts)==len(mons)
    return order,starts


def sym7_image(T,order,starts):
    T=T.reshape(*([5,2]*7))
    T=T.transpose([2*j for j in range(7)]+[2*j+1 for j in range(7)]).reshape(5**7,128)
    # At most 7! permutations contribute to each commutative monomial.
    assert 5040*int(np.abs(T).max())<2**63
    Y=np.add.reduceat(T[order],starts,axis=0)
    assert Y.shape==(330,128)
    return Y


def sym6_image(T,pattern,order,starts):
    sites=[pattern.index(1)]+[i for i in range(7) if i!=pattern.index(1)]
    T=T.transpose(sites).reshape(4,4,*([5,2]*6))
    T=T.transpose([0]+[2+2*j for j in range(6)]+[1]+[3+2*j for j in range(6)])
    T=T.reshape(4,5**6,256)
    # At most 6! permutations contribute to each commutative monomial.
    assert 720*int(np.abs(T).max())<2**63
    return np.add.reduceat(T[:,order,:],starts,axis=1).reshape(840,256)


def exact_projection(U,Y):
    maxu,maxy=int(np.abs(U).max()),int(np.abs(Y).max())
    assert U.shape[0]*maxu*maxy<2**63
    chunk=max(1,min(U.shape[0],(2**52)//max(1,maxu*maxy)))
    X=np.zeros((U.shape[1],Y.shape[1]),dtype=np.int64)
    for i in range(0,U.shape[0],chunk):
        j=min(U.shape[0],i+chunk)
        Q=U[i:j].T.astype(float)@Y[i:j].astype(float)
        assert np.array_equal(Q,np.rint(Q))
        X+=Q.astype(np.int64)
    return X
